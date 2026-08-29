"""Normalising flow: bijective transport + base distribution.

A :class:`Flow` extends :class:`NeuralSampler` with inverse operations
and exact ``log_prob`` evaluation, available only when the transport is
bijective.  It also provides backward-compatible aliases
``sample_and_log_prob`` and ``push_forward_and_log_prob`` which delegate
to the unified ``*_log_weight`` methods.

The class operates on arbitrary pytree states; all leaf-level handling
(dynamic/static partitioning, divergence computation) is delegated to the
bijector (transport).
"""

import equinox as eqx

from superiorflows.bijector import AbstractBijector
from superiorflows.sampler import NeuralSampler

__all__ = ["Flow"]


class Flow(NeuralSampler):
    """Normalising flow: pushforward of a base distribution through a bijector.

    Extends :class:`NeuralSampler` with inverse operations and exact
    ``log_prob`` evaluation.  For bijective transport, the log-weight
    IS the exact log-probability density, so ``sample_and_log_prob``
    and ``push_forward_and_log_prob`` are provided as aliases.

    The change-of-variables formula used internally::

        log p(y) = log p_base(f⁻¹(y)) + log|det J_{f⁻¹}(y)|
                 = log p_base(x)       − log|det J_f(x)|

    where ``f`` is the bijector's forward map.

    Attributes:
        transport: The invertible transformation (bijector).
        base_distribution: The base (prior) distribution.  Must provide
            ``sample(key=...)``, ``log_prob(value)``, and ``event_shape``.
    """

    transport: AbstractBijector

    @property
    def bijector(self):
        """Backward-compatible alias for ``transport``."""
        return self.transport

    # --- Backward-compatible aliases (log_weight IS log_prob for flows) ---

    def sample_and_log_prob(self, key, **kwargs):
        """Draw a sample and compute its exact log-probability.

        Alias for :meth:`sample_and_log_weight` — for bijective transport,
        the log-weight is the exact log-density.

        Args:
            key: PRNG key.
            **kwargs: Forwarded to the bijector.

        Returns:
            Tuple ``(y, log_prob_y)``.
        """
        return self.sample_and_log_weight(key, **kwargs)

    def push_forward_and_log_prob(self, x, **kwargs):
        """Push a base sample through the bijector and return its log-prob.

        Alias for :meth:`push_forward_and_log_weight`.

        Args:
            x: A sample from the base distribution.
            **kwargs: Forwarded to the bijector.

        Returns:
            Tuple ``(y, log_prob_y)``.
        """
        return self.push_forward_and_log_weight(x, **kwargs)

    # --- Bijector-specific operations ---

    @eqx.filter_jit
    def log_prob(self, value, **kwargs):
        """Evaluate the log-probability density at ``value``.

        Inverts the bijector to recover the base sample and applies the
        change-of-variables formula.

        Args:
            value: A point in the target space.
            **kwargs: Forwarded to ``bijector.inverse_and_log_det``
                (e.g. ``key=`` for Hutchinson).

        Returns:
            Scalar log-probability density.
        """
        x, inv_logdet = self.transport.inverse_and_log_det(value, **kwargs)
        return self.base_distribution.log_prob(x) + inv_logdet

    @eqx.filter_jit
    def pull_back_and_log_prob(self, y, **kwargs):
        """Pull a target sample back and return the log-prob at that point.

        Symmetric counterpart to :meth:`push_forward_and_log_prob`.

        Args:
            y: A point in the target space.
            **kwargs: Forwarded to ``bijector.inverse_and_log_det``.

        Returns:
            Tuple ``(x, log_prob_y)`` where ``x`` is the recovered base
            sample and ``log_prob_y`` is the log-probability density at ``y``.
        """
        x, inv_logdet = self.transport.inverse_and_log_det(y, **kwargs)
        log_prob_base = self.base_distribution.log_prob(x)
        return x, log_prob_base + inv_logdet
