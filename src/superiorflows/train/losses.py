from typing import Callable, Optional

import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp

from superiorflows import Flow

__all__ = [
    "MaximumLikelihoodLoss",
    "EnergyBasedLoss",
    "KullbackLeiblerLoss",
    "StochasticInterpolantLoss",
]


class MaximumLikelihoodLoss(eqx.Module):
    """Negative log-likelihood loss for flow training.

    Computes `-mean(log p(x))` where `p` is the flow density. The batch `x`
    should contain samples from the target distribution.

    Attributes:
        base_distribution: The base (prior) distribution for the flow.
        flow_kwargs: Additional keyword arguments passed to `Flow(...)`.

    Example:
        >>> loss_fn = MaximumLikelihoodLoss(base_dist, dt0=0.1)
        >>> loss = loss_fn(velocity_field, batch, key=jax.random.key(0))
    """

    base_distribution: dsx.Distribution
    flow_kwargs: dict = eqx.field(static=True)

    def __init__(self, base_distribution, **flow_kwargs):
        """Initialize the loss module.

        Args:
            base_distribution: A distrax Distribution for the flow base.
            **flow_kwargs: Passed to `Flow(...)` (e.g., `dt0`, `hutchinson_samples`).
        """
        self.base_distribution = base_distribution
        self.flow_kwargs = flow_kwargs

    @eqx.filter_jit
    def __call__(self, velocity_field, batch, key=None):
        """Compute the NLL loss.

        Args:
            velocity_field: The velocity field module (trainable).
            batch: Target samples with shape `(batch_size, *event_shape)`.
            key: Optional PRNG key for Hutchinson estimator.

        Returns:
            Scalar loss value.
        """
        flow = Flow(
            velocity_field=velocity_field,
            base_distribution=self.base_distribution,
            **self.flow_kwargs,
        )
        return -jnp.mean(flow.log_prob(batch, key=key))


class EnergyBasedLoss(eqx.Module):
    """Energy-based (reverse KL) loss for flow training.

    Minimizes `E_q[log q(x) - log p(x)]` where `q` is the pushforward of the
    base distribution through the flow, and `p` is the target.

    **Important**: The batch should contain samples from the BASE distribution,
    not the target.

    Attributes:
        base_distribution: The base distribution for the flow.
        target_distribution: The target distribution with known `log_prob`.
        flow_kwargs: Additional keyword arguments passed to `Flow(...)`.

    Example:
        >>> loss_fn = EnergyBasedLoss(base_dist, target_dist)
        >>> loss = loss_fn(velocity_field, base_samples, key=jax.random.key(0))
    """

    base_distribution: dsx.Distribution
    target_distribution: dsx.Distribution
    flow_kwargs: dict = eqx.field(static=True)

    def __init__(self, base_distribution, target_distribution, **flow_kwargs):
        """Initialize the loss module.

        Args:
            base_distribution: A distrax Distribution for the flow base.
            target_distribution: A distrax Distribution for the target.
            **flow_kwargs: Passed to `Flow(...)`.
        """
        self.base_distribution = base_distribution
        self.target_distribution = target_distribution
        self.flow_kwargs = flow_kwargs

    @eqx.filter_jit
    def __call__(self, velocity_field, batch, key=None):
        """Compute the energy-based loss.

        Args:
            velocity_field: The velocity field module (trainable).
            batch: Base distribution samples with shape `(batch_size, *event_shape)`.
            key: Optional PRNG key for Hutchinson estimator.

        Returns:
            Scalar loss value.
        """
        flow = Flow(
            velocity_field=velocity_field,
            base_distribution=self.base_distribution,
            **self.flow_kwargs,
        )

        x0 = batch

        if key is not None:
            batch_size = jax.tree.leaves(batch)[0].shape[0]
            keys = jax.random.split(key, batch_size)
            x1, logq = jax.vmap(lambda x, k: flow.apply_map_and_log_prob(x, key=k))(x0, keys)
        else:
            x1, logq = jax.vmap(flow.apply_map_and_log_prob)(x0)

        logp = jax.vmap(self.target_distribution.log_prob)(x1)
        return jnp.mean(logq - logp)


class KullbackLeiblerLoss(eqx.Module):
    """Hybrid forward/reverse KL loss for flow training.

    Combines maximum likelihood (forward KL) and energy-based (reverse KL)
    losses with a blending coefficient alpha:

        loss = alpha * NLL + (1 - alpha) * EnergyLoss

    - alpha=1.0: Pure maximum likelihood (forward KL).
    - alpha=0.0: Pure energy-based (reverse KL).

    **Note**: The batch should contain samples from the TARGET distribution.
    Base samples for the energy term are generated internally.

    Attributes:
        mle_loss: The MaximumLikelihoodLoss component.
        energy_loss: The EnergyBasedLoss component.
        base_distribution: The base distribution (for internal sampling).
        alpha: Blending coefficient in [0, 1].

    Example:
        >>> loss_fn = KullbackLeiblerLoss(base_dist, target_dist, alpha=0.5)
        >>> loss = loss_fn(velocity_field, target_samples, key=jax.random.key(0))
    """

    mle_loss: MaximumLikelihoodLoss
    energy_loss: EnergyBasedLoss
    base_distribution: dsx.Distribution
    alpha: float

    def __init__(self, base_distribution, target_distribution, alpha=0.5, **flow_kwargs):
        """Initialize the hybrid loss.

        Args:
            base_distribution: A distrax Distribution for the flow base.
            target_distribution: A distrax Distribution for the target.
            alpha: Blending coefficient. 1.0 = pure MLE, 0.0 = pure energy-based.
            **flow_kwargs: Passed to both component losses.
        """
        self.mle_loss = MaximumLikelihoodLoss(base_distribution, **flow_kwargs)
        self.energy_loss = EnergyBasedLoss(base_distribution, target_distribution, **flow_kwargs)
        self.base_distribution = base_distribution
        self.alpha = alpha

    @eqx.filter_jit
    def __call__(self, velocity_field, batch, key):
        """Compute the hybrid KL loss.

        Args:
            velocity_field: The velocity field module (trainable).
            batch: Target distribution samples.
            key: PRNG key (required for internal sampling and Hutchinson).

        Returns:
            Scalar loss value.
        """
        x1 = batch
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        key1, key2, key3 = jax.random.split(key, 3)
        x0 = self.base_distribution.sample(seed=key1, sample_shape=(batch_size,))
        mle_term = self.mle_loss(velocity_field, x1, key=key2)
        energy_term = self.energy_loss(velocity_field, x0, key=key3)
        return self.alpha * mle_term + (1 - self.alpha) * energy_term


class StochasticInterpolantLoss(eqx.Module):
    """Stochastic Interpolant loss for flow matching training.

    Given coupled samples ``(x0, x1)`` from a coupling ``ν(dx0, dx1)``
    that marginalises onto the base and target, constructs interpolants

        ``xt = I(t, x0, x1) + γ(t) · z``

    and trains the velocity field to match the optimal transport velocity:

        ``L = E_{t, x0, x1, z}[||v(t, xt) - (∂_t I(t, x0, x1) + ∂_t γ(t) · z)||²]``

    The interpolation function ``I`` must satisfy ``I(0, x0, x1) = x0`` and
    ``I(1, x0, x1) = x1``. The optional noise schedule ``γ(t)`` must satisfy
    ``γ(0) = γ(1) = 0`` and ``γ(t) > 0`` for ``t ∈ (0, 1)``.

    Time derivatives ``∂_t I`` and ``∂_t γ`` are computed automatically
    via ``jax.jvp`` at initialisation for efficiency.

    If ``gamma`` is ``None``, the deterministic (noiseless) interpolant is used.

    The per-sample loss is defined on single (unbatched) elements and
    ``jax.vmap``-ed over the batch, making it robust to arbitrary pytree
    data structures.

    Attributes:
        interpolant: Function ``I(t, x0, x1)`` mapping scalar ``t`` and
            single samples ``x0``, ``x1`` to the interpolated point.
        gamma: Optional noise schedule ``γ(t)``, or ``None``.
        dynamic_mask: Function ``mask(x)`` that returns a pytree with the same
            structure as ``x`` but with boolean arrays indicating which
            components are part of the state.
        velocity_kwargs: Keyword arguments passed to the velocity field.
        dt_interpolant: Time derivative ``∂_t I(t, x0, x1)`` (via autodiff).
        dt_gamma: Time derivative ``∂_t γ(t)`` (via autodiff), or ``None``.

    Example:
        >>> interpolant = lambda t, x0, x1: (1 - t) * x0 + t * x1
        >>> gamma = lambda t: jnp.sqrt(2 * t * (1 - t))
        >>> loss_fn = StochasticInterpolantLoss(interpolant, gamma=gamma)
        >>> loss = loss_fn(velocity_field, (x0_batch, x1_batch), key=jax.random.key(0))
    """

    interpolant: Callable = eqx.field(static=True)
    gamma: Optional[Callable] = eqx.field(static=True)
    dynamic_mask: Callable = eqx.field(
        default=lambda x: jax.tree.map(eqx.is_inexact_array, x),
        static=True,
    )
    velocity_kwargs: dict = eqx.field(static=True)
    dt_interpolant: Callable = eqx.field(static=True)
    dt_gamma: Optional[Callable] = eqx.field(static=True)

    def __init__(
        self,
        interpolant: Callable,
        gamma: Optional[Callable] = None,
        dynamic_mask: Optional[Callable] = None,
        **velocity_kwargs,
    ):
        """Initialize the Stochastic Interpolant loss.

        Precomputes time derivatives of the interpolant and (optionally)
        gamma via ``jax.jvp``.

        Args:
            interpolant: Function ``I(t, x0, x1)`` operating on scalar ``t``
                and single (unbatched) samples. Must satisfy
                ``I(0, x0, x1) = x0`` and ``I(1, x0, x1) = x1``.
            gamma: Optional noise schedule ``γ(t)`` operating on scalar ``t``.
                Must satisfy ``γ(0) = γ(1) = 0`` and ``γ(t) > 0`` for
                ``t ∈ (0, 1)``. If ``None``, uses the deterministic interpolant.
        """
        self.interpolant = interpolant
        self.gamma = gamma
        if dynamic_mask is not None:
            self.dynamic_mask = dynamic_mask
        else:
            self.dynamic_mask = lambda x: jax.tree.map(eqx.is_inexact_array, x)
        self.velocity_kwargs = velocity_kwargs

        # --- Time derivatives via JVP (precomputed as closures) ---
        def _dt_interpolant(t, x0, x1):
            _, tangent = jax.jvp(
                lambda s: interpolant(s, x0, x1),
                (t,),
                (jnp.ones_like(t),),
            )
            return tangent

        self.dt_interpolant = _dt_interpolant

        if gamma is not None:

            def _dt_gamma(t):
                _, tangent = jax.jvp(gamma, (t,), (jnp.ones_like(t),))
                return tangent

            self.dt_gamma = _dt_gamma
        else:
            self.dt_gamma = None

    @eqx.filter_jit
    def __call__(self, velocity_field, batch, key):
        """Compute the Stochastic Interpolant loss.

        Args:
            velocity_field: The velocity field module (trainable).
            batch: A tuple ``(x0, x1)`` of paired samples. Each element
                is a pytree whose leaves have a leading batch dimension.
            key: PRNG key for sampling ``t`` and (optionally) ``z``.

        Returns:
            Scalar loss value.
        """
        x0, x1 = batch
        batch_size = jax.tree.leaves(x0)[0].shape[0]

        key1, key2 = jax.random.split(key)
        t = jax.random.uniform(key1, (batch_size,))

        # Partition into dynamic components and strict static context
        y0, ctx = eqx.partition(x0, self.dynamic_mask)
        y1, _ = eqx.partition(x1, self.dynamic_mask)

        user_args = self.velocity_kwargs.get("args")

        # Branch resolved at trace time (gamma is static).
        if self.gamma is not None:
            # Generate noise matching the dynamic pytree structure of y0
            y0_leaves, y0_treedef = jax.tree.flatten(y0)
            noise_keys = jax.random.split(key2, len(y0_leaves))
            z = jax.tree.unflatten(
                y0_treedef,
                [jax.random.normal(k, leaf.shape) for k, leaf in zip(noise_keys, y0_leaves)],
            )

            def _sample_loss(ti, y0i, y1i, ctxi, zi):
                interp = self.interpolant(ti, y0i, y1i)
                gamma_t = self.gamma(ti)
                yt = jax.tree.map(lambda i, zp: i + gamma_t * zp, interp, zi)

                dt_interp = self.dt_interpolant(ti, y0i, y1i)
                dt_gamma_t = self.dt_gamma(ti)
                target = jax.tree.map(lambda d, zp: d + dt_gamma_t * zp, dt_interp, zi)

                args_i = (ctxi, user_args) if user_args is not None else ctxi

                pred = velocity_field(ti, yt, args_i)
                sq_res = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), pred, target))
                return sum(sq_res)

            per_sample = jax.vmap(_sample_loss)(t, y0, y1, ctx, z)
        else:

            def _sample_loss(ti, y0i, y1i, ctxi):
                yt = self.interpolant(ti, y0i, y1i)
                target = self.dt_interpolant(ti, y0i, y1i)

                args_i = (ctxi, user_args) if user_args is not None else ctxi

                pred = velocity_field(ti, yt, args_i)
                sq_res = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), pred, target))
                return sum(sq_res)

            per_sample = jax.vmap(_sample_loss)(t, y0, y1, ctx)

        return jnp.mean(per_sample)
