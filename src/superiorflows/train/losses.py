from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from superiorflows.flow import Flow
from superiorflows.partial import PartialUpdater
from superiorflows.partition import state_context_partition


def _vmap_log_prob(flow, batch, key=None):
    """Evaluate flow.log_prob over a batch, splitting keys for Hutchinson."""
    if key is not None:
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        keys = jax.random.split(key, batch_size)
        return jax.vmap(lambda x, k: flow.log_prob(x, key=k))(batch, keys)
    return jax.vmap(flow.log_prob)(batch)


def _vmap_push_forward_and_log_prob(flow, batch, key=None):
    """Evaluate flow.push_forward_and_log_prob over a batch, splitting keys for Hutchinson."""
    if key is not None:
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        keys = jax.random.split(key, batch_size)
        return jax.vmap(lambda x, k: flow.push_forward_and_log_prob(x, key=k))(batch, keys)
    return jax.vmap(flow.push_forward_and_log_prob)(batch)


def _pack_args(ctx, user_args):
    """Combine partition context with user-supplied args."""
    if user_args is not None:
        return (ctx, user_args)
    return ctx


__all__ = [
    "MaximumLikelihoodLoss",
    "EnergyBasedLoss",
    "KullbackLeiblerLoss",
    "StochasticInterpolantLoss",
]


class MaximumLikelihoodLoss(eqx.Module):
    """Negative log-likelihood loss for any normalising flow.

    Computes ``−mean(log p(x))`` where ``p`` is the flow density.  The batch
    ``x`` should contain samples from the target distribution.

    The ``make_bijector`` callable encapsulates all bijector configuration.
    For a CNF::

        make_bijector = lambda vf: ODEBijector(vf, dt0=0.1, ...)

    For a discrete flow where the model IS the bijector::

        make_bijector = lambda m: m

    When ``selection_protocol`` is set, the loss trains a **partial updater**:
    each sample ``x`` is partitioned into state ``s`` and context ``c`` via the
    protocol, and the loss becomes ``−E[log q_θ(s | c)]``.

    Attributes:
        base_distribution: The base (prior) distribution.
        make_bijector: Callable ``model → AbstractBijector``.
        selection_protocol: Optional callable ``(key, x) → dynamic_mask``.
            When ``None`` (default), the loss operates on the full state.
    """

    base_distribution: eqx.Module
    make_bijector: Callable = eqx.field(static=True)
    selection_protocol: Optional[Callable] = eqx.field(default=None, static=True)

    @eqx.filter_jit
    def __call__(self, model, batch, key=None):
        """Compute the NLL loss.

        Args:
            model: The trainable model (e.g., velocity field for a CNF).
            batch: Target samples with shape ``(batch_size, *event_shape)``.
            key: Optional PRNG key (e.g., for Hutchinson estimator).
                **Required** when ``selection_protocol`` is set.

        Returns:
            Tuple ``(loss, aux)`` where ``loss`` is a scalar and
            ``aux`` is a dict of per-component metrics (empty here).
        """
        bijector = self.make_bijector(model)

        if self.selection_protocol is None:
            flow = Flow(bijector, self.base_distribution)
            return -jnp.mean(_vmap_log_prob(flow, batch, key=key)), {}

        updater = PartialUpdater(bijector, self.base_distribution)
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        keys = jax.random.split(key, batch_size)

        def conditional_log_prob(xi, key_i):
            k_sel, k_lp = jax.random.split(key_i)
            mask = self.selection_protocol(k_sel, xi)
            return updater.log_prob(xi, mask, key=k_lp)

        log_probs = jax.vmap(conditional_log_prob)(batch, keys)
        return -jnp.mean(log_probs), {}


class EnergyBasedLoss(eqx.Module):
    """Energy-based (reverse KL) loss for any normalising flow.

    Minimises ``E_q[log q(x) − log p(x)]`` where ``q`` is the pushforward of
    the base distribution through the flow, and ``p`` is the target.

    **Important**: The batch should contain samples from the BASE distribution,
    not the target.

    When ``selection_protocol`` is set, the batch should contain **full states**
    from which the context is extracted.  The dynamic DOFs are sampled from the
    base and pushed forward internally.

    Attributes:
        base_distribution: The base distribution.
        target_distribution: The target distribution with known ``log_prob``.
        make_bijector: Callable ``model → AbstractBijector``.
        selection_protocol: Optional callable ``(key, x) → dynamic_mask``.
    """

    base_distribution: eqx.Module
    target_distribution: eqx.Module
    make_bijector: Callable = eqx.field(static=True)
    selection_protocol: Optional[Callable] = eqx.field(default=None, static=True)

    @eqx.filter_jit
    def __call__(self, model, batch, key=None):
        """Compute the energy-based loss.

        Args:
            model: The trainable model.
            batch: Base distribution samples (or full states when
                ``selection_protocol`` is set).
            key: Optional PRNG key.  **Required** when
                ``selection_protocol`` is set.

        Returns:
            Tuple ``(loss, aux)``.
        """
        bijector = self.make_bijector(model)

        if self.selection_protocol is None:
            flow = Flow(bijector, self.base_distribution)
            x0 = batch
            x1, logq = _vmap_push_forward_and_log_prob(flow, x0, key=key)
            logp = jax.vmap(self.target_distribution.log_prob)(x1)
            return jnp.mean(logq - logp), {}

        updater = PartialUpdater(bijector, self.base_distribution)
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        keys = jax.random.split(key, batch_size)

        def energy_single(xi, key_i):
            k_sel, k_rest = jax.random.split(key_i)
            k_sample, k_lp = jax.random.split(k_rest)
            mask = self.selection_protocol(k_sel, xi)
            x1_i, logq_i = updater.update_and_log_prob(xi, mask, k_sample, key=k_lp)
            logp_i = self.target_distribution.log_prob(x1_i)
            return logq_i - logp_i

        per_sample = jax.vmap(energy_single)(batch, keys)
        return jnp.mean(per_sample), {}


class KullbackLeiblerLoss(eqx.Module):
    """Hybrid forward/reverse KL loss for any normalising flow.

    Combines maximum likelihood (forward KL) and energy-based (reverse KL)
    losses with a blending coefficient ``alpha``:

        ``loss = alpha * NLL + (1 − alpha) * EnergyLoss``

    - ``alpha=1.0``: Pure maximum likelihood (forward KL).
    - ``alpha=0.0``: Pure energy-based (reverse KL).

    **Note**: The batch should contain samples from the TARGET distribution.
    Base samples for the energy term are generated internally.

    When ``selection_protocol`` is set, both sub-losses operate in partial-
    update mode.

    Attributes:
        mle_loss: The MaximumLikelihoodLoss component.
        energy_loss: The EnergyBasedLoss component.
        base_distribution: The base distribution (for internal sampling).
        alpha: Blending coefficient in [0, 1].
    """

    mle_loss: MaximumLikelihoodLoss
    energy_loss: EnergyBasedLoss
    base_distribution: eqx.Module
    alpha: float

    def __init__(self, base_distribution, target_distribution, make_bijector, alpha=0.5, selection_protocol=None):
        """Initialise the hybrid loss.

        Args:
            base_distribution: The base distribution.
            target_distribution: The target distribution.
            make_bijector: Callable ``model → AbstractBijector``.
            alpha: Blending coefficient. 1.0 = pure MLE, 0.0 = pure energy-based.
            selection_protocol: Optional callable ``(key, x) → dynamic_mask``.
        """
        self.mle_loss = MaximumLikelihoodLoss(base_distribution, make_bijector, selection_protocol=selection_protocol)
        self.energy_loss = EnergyBasedLoss(
            base_distribution, target_distribution, make_bijector, selection_protocol=selection_protocol
        )
        self.base_distribution = base_distribution
        self.alpha = alpha

    @eqx.filter_jit
    def __call__(self, model, batch, key):
        """Compute the hybrid KL loss.

        Args:
            model: The trainable model.
            batch: Target distribution samples.
            key: PRNG key (required for internal sampling).

        Returns:
            Tuple ``(loss, aux)``.
        """
        x1 = batch
        batch_size = jax.tree.leaves(batch)[0].shape[0]
        key1, key2, key3 = jax.random.split(key, 3)
        keys = jax.random.split(key1, batch_size)
        x0 = jax.vmap(self.base_distribution.sample)(keys)
        mle_term, _ = self.mle_loss(model, x1, key=key2)
        energy_term, _ = self.energy_loss(model, x0, key=key3)
        return self.alpha * mle_term + (1 - self.alpha) * energy_term, {}


class StochasticInterpolantLoss(eqx.Module):
    """Stochastic Interpolant loss for flow matching training.

    Given coupled samples ``(x0, x1)`` from a coupling ``ν(dx0, dx1)``
    that marginalises onto the base and target, constructs interpolants

        ``xt = I(t, x0, x1) + γ(t) · z``

    and trains the velocity field to match the optimal transport velocity:

        ``L_vel = E_{t, x0, x1, z}[||v(t, xt) - (∂_t I(t, x0, x1) + ∂_t γ(t) · z)||²]``

    Optionally, a **denoiser** ``η(t, x)`` can be trained jointly via:

        ``L_den = E_{t, x0, x1, z}[||η(t, xt) - z||²]``

    The total loss is ``L_vel + denoiser_weight · L_den``. The denoiser
    approximates the score via ``s(t, x) ≈ −η(t, x) / γ(t)`` (Albergo,
    Boffi & Vanden-Eijnden, 2303.08797).

    The interpolation function ``I`` must satisfy ``I(0, x0, x1) = x0`` and
    ``I(1, x0, x1) = x1``. The optional noise schedule ``γ(t)`` must satisfy
    ``γ(0) = γ(1) = 0`` and ``γ(t) > 0`` for ``t ∈ (0, 1)``.

    Time derivatives ``∂_t I`` and ``∂_t γ`` are computed automatically
    via ``jax.jvp`` at initialisation for efficiency.

    If ``gamma`` is ``None``, the deterministic (noiseless) interpolant is used.

    The per-sample loss is defined on single (unbatched) elements and
    ``jax.vmap``-ed over the batch, making it robust to arbitrary pytree
    data structures.

    When ``get_denoiser`` is ``None`` (the default), only the velocity
    loss is computed and the ``__call__`` method traces exactly the same
    code path as the velocity-only case — no overhead.

    When ``selection_protocol`` is set, each sample in the batch is
    partitioned into state and context via the protocol, and the velocity
    field is trained only on the state DOFs.

    Attributes:
        interpolant: Function ``I(t, x0, x1)`` mapping scalar ``t`` and
            single samples ``x0``, ``x1`` to the interpolated point.
        gamma: Optional noise schedule ``γ(t)``, or ``None``.
        dynamic_mask: Function ``mask(x)`` that returns a pytree with the same
            structure as ``x`` but with boolean arrays indicating which
            components are part of the state.
        selection_protocol: Optional callable ``(key, x) → dynamic_mask``.
            When set, overrides ``dynamic_mask`` with a per-sample mask
            generated stochastically for each batch element.
        velocity_kwargs: Keyword arguments passed to the velocity field.
        dt_interpolant: Time derivative ``∂_t I(t, x0, x1)`` (via autodiff).
        dt_gamma: Time derivative ``∂_t γ(t)`` (via autodiff), or ``None``.
        denoiser_weight: Relative weight ``λ`` of the denoiser loss in
            ``L_total = L_vel + λ · L_den``. Only used when ``get_denoiser``
            is not ``None``.
        _get_velocity: Callable that extracts the velocity field from the
            model pytree passed to ``__call__``. Default: identity.
        _get_denoiser: Optional callable that extracts the denoiser from the
            model pytree. ``None`` disables denoiser learning.

    Example (velocity only — current default):
        >>> interpolant = lambda t, x0, x1: (1 - t) * x0 + t * x1
        >>> gamma = lambda t: jnp.sqrt(2 * t * (1 - t))
        >>> loss_fn = StochasticInterpolantLoss(interpolant, gamma=gamma)
        >>> loss, aux = loss_fn(velocity_field, (x0_batch, x1_batch), key=jax.random.key(0))

    Example (velocity + denoiser):
        >>> loss_fn = StochasticInterpolantLoss(
        ...     interpolant, gamma=gamma,
        ...     get_velocity=lambda m: m.velocity_field,
        ...     get_denoiser=lambda m: m.denoiser,
        ... )
        >>> loss, aux = loss_fn(model_pair, (x0_batch, x1_batch), key=jax.random.key(0))
        >>> # aux == {"velocity_loss": ..., "denoiser_loss": ...}
    """

    interpolant: Callable = eqx.field(static=True)
    gamma: Optional[Callable] = eqx.field(static=True)
    velocity_kwargs: dict = eqx.field(static=True)
    dt_interpolant: Callable = eqx.field(static=True)
    dt_gamma: Optional[Callable] = eqx.field(static=True)
    denoiser_weight: float = eqx.field(static=True)
    _get_velocity: Callable = eqx.field(static=True)
    _get_denoiser: Optional[Callable] = eqx.field(static=True)
    dynamic_mask: Callable = eqx.field(
        default=lambda x: jax.tree.map(eqx.is_inexact_array, x),
        static=True,
    )
    selection_protocol: Optional[Callable] = eqx.field(default=None, static=True)

    def __init__(
        self,
        interpolant: Callable,
        gamma: Optional[Callable] = None,
        dynamic_mask: Optional[Callable] = None,
        get_velocity: Optional[Callable] = None,
        get_denoiser: Optional[Callable] = None,
        denoiser_weight: float = 1.0,
        selection_protocol: Optional[Callable] = None,
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
            dynamic_mask: Optional callable ``mask(x)`` that returns a pytree
                of booleans or index arrays. Defaults to
                ``eqx.is_inexact_array``.
            get_velocity: Callable ``model -> velocity_field`` to extract the
                velocity field from the model. Defaults to identity (the model
                *is* the velocity field).
            get_denoiser: Optional callable ``model -> denoiser`` to extract
                the denoiser from the model. ``None`` disables denoiser learning.
                Requires ``gamma`` to be set.
            denoiser_weight: Relative weight of the denoiser loss.
            selection_protocol: Optional callable ``(key, x) → dynamic_mask``.
                When set, overrides ``dynamic_mask`` with a per-sample mask.
            **velocity_kwargs: Extra keyword arguments forwarded to the
                velocity field (e.g., ``args``).
        """
        self.interpolant = interpolant
        self.gamma = gamma
        if dynamic_mask is not None:
            self.dynamic_mask = dynamic_mask
        else:
            self.dynamic_mask = lambda x: jax.tree.map(eqx.is_inexact_array, x)
        self.selection_protocol = selection_protocol
        self.velocity_kwargs = velocity_kwargs
        self.denoiser_weight = denoiser_weight
        self._get_velocity = get_velocity if get_velocity is not None else lambda m: m
        self._get_denoiser = get_denoiser

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

    def __check_init__(self):
        if self._get_denoiser is not None and self.gamma is None:
            raise ValueError(
                "Denoiser learning requires a noise schedule gamma(t). Pass gamma= to StochasticInterpolantLoss."
            )

    def _resolve_mask(self, x, sel_key=None):
        """Return the dynamic mask for a single sample.

        When ``selection_protocol`` is set and ``sel_key`` is provided,
        the mask is generated per-sample.  Otherwise, the global
        ``dynamic_mask`` is used.
        """
        if self.selection_protocol is not None and sel_key is not None:
            return self.selection_protocol(sel_key, x)
        return self.dynamic_mask

    @eqx.filter_jit
    def __call__(self, model, batch, key):
        """Compute the Stochastic Interpolant loss.

        Args:
            model: The trainable model pytree. When ``get_denoiser`` is
                ``None``, this is the velocity field itself. When a denoiser
                is configured, the velocity and denoiser are extracted via
                the accessor callables provided at init.
            batch: A tuple ``(x0, x1)`` of paired samples. Each element
                is a pytree whose leaves have a leading batch dimension.
            key: PRNG key for sampling ``t`` and (optionally) ``z``.

        Returns:
            Tuple ``(loss, aux)`` where ``loss`` is a scalar and ``aux``
            is a dict. When a denoiser is active, ``aux`` contains
            ``{"velocity_loss": ..., "denoiser_loss": ...}``.
        """
        x0, x1 = batch
        batch_size = jax.tree.leaves(x0)[0].shape[0]

        key1, key2, key_sel = jax.random.split(key, 3)
        t = jax.random.uniform(key1, (batch_size,))

        user_args = self.velocity_kwargs.get("args")
        velocity_field = self._get_velocity(model)

        use_per_sample_mask = self.selection_protocol is not None
        sel_keys = jax.random.split(key_sel, batch_size) if use_per_sample_mask else None

        if self.gamma is not None:
            if self._get_denoiser is not None:
                denoiser = self._get_denoiser(model)
                return self._call_gamma_denoiser(t, x0, x1, key2, velocity_field, denoiser, user_args, sel_keys)
            return self._call_gamma(t, x0, x1, key2, velocity_field, user_args, sel_keys)
        return self._call_deterministic(t, x0, x1, velocity_field, user_args, sel_keys)

    def _call_deterministic(self, t, x0, x1, velocity_field, user_args, sel_keys):
        """Deterministic interpolant (no noise)."""

        if sel_keys is None:

            def _sample_loss(ti, x0i, x1i):
                y0i, ctxi, _ = state_context_partition(x0i, self.dynamic_mask)
                y1i, _, _ = state_context_partition(x1i, self.dynamic_mask)

                yt = self.interpolant(ti, y0i, y1i)
                target = self.dt_interpolant(ti, y0i, y1i)

                pred = velocity_field(ti, yt, _pack_args(ctxi, user_args))
                sq_res = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), pred, target))
                return sum(sq_res)

            per_sample = jax.vmap(_sample_loss)(t, x0, x1)
        else:

            def _sample_loss(ti, x0i, x1i, sel_key_i):
                mask_i = self.selection_protocol(sel_key_i, x1i)
                y0i, ctxi, _ = state_context_partition(x0i, mask_i)
                y1i, _, _ = state_context_partition(x1i, mask_i)

                yt = self.interpolant(ti, y0i, y1i)
                target = self.dt_interpolant(ti, y0i, y1i)

                pred = velocity_field(ti, yt, _pack_args(ctxi, user_args))
                sq_res = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), pred, target))
                return sum(sq_res)

            per_sample = jax.vmap(_sample_loss)(t, x0, x1, sel_keys)

        return jnp.mean(per_sample), {}

    def _call_gamma(self, t, x0, x1, key2, velocity_field, user_args, sel_keys):
        """Noisy interpolant without denoiser."""

        if sel_keys is None:
            y0, ctx = state_context_partition(x0, self.dynamic_mask)[:2]
            y1 = state_context_partition(x1, self.dynamic_mask)[0]

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

                pred = velocity_field(ti, yt, _pack_args(ctxi, user_args))
                sq_res = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), pred, target))
                return sum(sq_res)

            per_sample = jax.vmap(_sample_loss)(t, y0, y1, ctx, z)
        else:

            def _sample_loss(ti, x0i, x1i, sel_key_i, noise_key_i):
                mask_i = self.selection_protocol(sel_key_i, x1i)
                y0i, ctxi, _ = state_context_partition(x0i, mask_i)
                y1i, _, _ = state_context_partition(x1i, mask_i)

                y0i_leaves, y0i_treedef = jax.tree.flatten(y0i)
                nk = jax.random.split(noise_key_i, len(y0i_leaves))
                zi = jax.tree.unflatten(
                    y0i_treedef,
                    [jax.random.normal(k, leaf.shape) for k, leaf in zip(nk, y0i_leaves)],
                )

                interp = self.interpolant(ti, y0i, y1i)
                gamma_t = self.gamma(ti)
                yt = jax.tree.map(lambda i, zp: i + gamma_t * zp, interp, zi)

                dt_interp = self.dt_interpolant(ti, y0i, y1i)
                dt_gamma_t = self.dt_gamma(ti)
                target = jax.tree.map(lambda d, zp: d + dt_gamma_t * zp, dt_interp, zi)

                pred = velocity_field(ti, yt, _pack_args(ctxi, user_args))
                sq_res = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), pred, target))
                return sum(sq_res)

            batch_size = jax.tree.leaves(x0)[0].shape[0]
            noise_keys = jax.random.split(key2, batch_size)
            per_sample = jax.vmap(_sample_loss)(t, x0, x1, sel_keys, noise_keys)

        return jnp.mean(per_sample), {}

    def _call_gamma_denoiser(self, t, x0, x1, key2, velocity_field, denoiser, user_args, sel_keys):
        """Noisy interpolant with denoiser."""

        if sel_keys is None:
            y0, ctx = state_context_partition(x0, self.dynamic_mask)[:2]
            y1 = state_context_partition(x1, self.dynamic_mask)[0]

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
                vel_target = jax.tree.map(lambda d, zp: d + dt_gamma_t * zp, dt_interp, zi)

                args_i = _pack_args(ctxi, user_args)

                v_pred = velocity_field(ti, yt, args_i)
                vel_sq = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), v_pred, vel_target))

                eta_pred = denoiser(ti, yt, args_i)
                den_sq = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), eta_pred, zi))

                return sum(vel_sq), sum(den_sq)

            vel_per_sample, den_per_sample = jax.vmap(_sample_loss)(t, y0, y1, ctx, z)
        else:

            def _sample_loss(ti, x0i, x1i, sel_key_i, noise_key_i):
                mask_i = self.selection_protocol(sel_key_i, x1i)
                y0i, ctxi, _ = state_context_partition(x0i, mask_i)
                y1i, _, _ = state_context_partition(x1i, mask_i)

                y0i_leaves, y0i_treedef = jax.tree.flatten(y0i)
                nk = jax.random.split(noise_key_i, len(y0i_leaves))
                zi = jax.tree.unflatten(
                    y0i_treedef,
                    [jax.random.normal(k, leaf.shape) for k, leaf in zip(nk, y0i_leaves)],
                )

                interp = self.interpolant(ti, y0i, y1i)
                gamma_t = self.gamma(ti)
                yt = jax.tree.map(lambda i, zp: i + gamma_t * zp, interp, zi)

                dt_interp = self.dt_interpolant(ti, y0i, y1i)
                dt_gamma_t = self.dt_gamma(ti)
                vel_target = jax.tree.map(lambda d, zp: d + dt_gamma_t * zp, dt_interp, zi)

                args_i = _pack_args(ctxi, user_args)

                v_pred = velocity_field(ti, yt, args_i)
                vel_sq = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), v_pred, vel_target))

                eta_pred = denoiser(ti, yt, args_i)
                den_sq = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), eta_pred, zi))

                return sum(vel_sq), sum(den_sq)

            batch_size = jax.tree.leaves(x0)[0].shape[0]
            noise_keys = jax.random.split(key2, batch_size)
            vel_per_sample, den_per_sample = jax.vmap(_sample_loss)(t, x0, x1, sel_keys, noise_keys)

        vel_loss = jnp.mean(vel_per_sample)
        den_loss = jnp.mean(den_per_sample)
        total = vel_loss + self.denoiser_weight * den_loss
        return total, {"velocity_loss": vel_loss, "denoiser_loss": den_loss}
