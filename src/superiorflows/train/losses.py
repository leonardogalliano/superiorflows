"""Loss functions for training flow-based models.

This module provides differentiable loss modules that wrap the `Flow` class
for maximum likelihood, energy-based, and hybrid KL training objectives.
"""
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp

from superiorflows import Flow

__all__ = ["MaximumLikelihoodLoss", "EnergyBasedLoss", "KullbackLeiblerLoss"]


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
