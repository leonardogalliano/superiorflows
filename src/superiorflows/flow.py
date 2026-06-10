"""Generic normalising flow.

A :class:`Flow` composes an invertible transformation (bijector) with a base
distribution to define a new distribution via the change-of-variables formula.
It is agnostic to the nature of the bijector — ODE-based continuous flows,
discrete invertible architectures, or custom wrapped bijectors all work.

The class operates on arbitrary pytree states; all leaf-level handling
(dynamic/static partitioning, divergence computation) is delegated to the
bijector.
"""

import equinox as eqx
import jax

from superiorflows.bijector import AbstractBijector

__all__ = ["Flow"]


class Flow(eqx.Module):
    """Normalising flow: pushforward of a base distribution through a bijector.

    Composes an invertible transformation with a base distribution to provide
    ``sample``, ``log_prob``, and ``sample_and_log_prob``.

    The change-of-variables formula used internally:

        log p(y) = log p_base(f⁻¹(y)) + log|det J_{f⁻¹}(y)|
                 = log p_base(x)       − log|det J_f(x)|

    where ``f`` is the bijector's forward map.

    Attributes:
        bijector: The invertible transformation defining the flow.
        base_distribution: The base (prior) distribution.  Must provide
            ``sample(key=...)``, ``log_prob(value)``, and ``event_shape``.
    """

    bijector: AbstractBijector
    base_distribution: eqx.Module

    @property
    def event_shape(self):
        return self.base_distribution.event_shape

    @eqx.filter_jit
    def sample(self, key, **kwargs):
        """Draw a sample from the flow.

        Args:
            key: PRNG key for base-distribution sampling.
            **kwargs: Forwarded to ``bijector.forward``.

        Returns:
            A sample from the pushforward distribution.
        """
        x = self.base_distribution.sample(key=key)
        return self.bijector.forward(x, **kwargs)

    @eqx.filter_jit
    def sample_and_log_prob(self, key, **kwargs):
        """Draw a sample and compute its log-probability under the flow.

        More efficient than calling ``sample`` and ``log_prob`` separately
        because the forward pass computes the log-det in tandem.

        Args:
            key: PRNG key.  When the bijector requires a key (e.g. for
                Hutchinson trace estimation), a subkey is split automatically.
            **kwargs: Forwarded to ``bijector.forward_and_log_det``.

        Returns:
            Tuple ``(y, log_prob_y)``.
        """
        key_sample, key_bijector = jax.random.split(key)
        x = self.base_distribution.sample(key=key_sample)
        if "key" not in kwargs:
            kwargs = dict(kwargs, key=key_bijector)
        return self.push_forward_and_log_prob(x, **kwargs)

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
        x, inv_logdet = self.bijector.inverse_and_log_det(value, **kwargs)
        return self.base_distribution.log_prob(x) + inv_logdet

    @eqx.filter_jit
    def push_forward_and_log_prob(self, x, **kwargs):
        """Push a base sample through the bijector and return its log-prob.

        Args:
            x: A sample from the base distribution.
            **kwargs: Forwarded to ``bijector.forward_and_log_det``.

        Returns:
            Tuple ``(y, log_prob_y)``.
        """
        y, fwd_logdet = self.bijector.forward_and_log_det(x, **kwargs)
        log_prob_base = self.base_distribution.log_prob(x)
        return y, log_prob_base - fwd_logdet

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
        x, inv_logdet = self.bijector.inverse_and_log_det(y, **kwargs)
        log_prob_base = self.base_distribution.log_prob(x)
        return x, log_prob_base + inv_logdet
