"""Unified neural sampler composing transport with base distribution.

:class:`NeuralSampler` is the transport-agnostic sampler: it composes
any :class:`AbstractTransport` (bijective or stochastic) with a base
distribution to provide ``sample``, ``sample_and_log_weight``, and
``push_forward_and_log_weight``.

For importance sampling against a target density ρ*::

    y, log_w = sampler.sample_and_log_weight(key)
    log_iw = log_rho_star(y) - log_w
"""

import equinox as eqx
import jax

from superiorflows.transport import AbstractTransport

__all__ = ["NeuralSampler"]


class NeuralSampler(eqx.Module):
    """Unified neural sampler: transport + base distribution.

    Works polymorphically with any :class:`AbstractTransport` (bijective
    or stochastic).  The key method is :meth:`push_forward_and_log_weight`,
    which combines the base log-density with the transport's contribution:

        ``log_weight(y) = log q₀(x₀) + c``

    For importance sampling: ``log w = log ρ*(y) − log_weight(y)``.

    Attributes:
        transport: The transport mechanism (e.g. :class:`ODEBijector`,
            :class:`SDETransport`).
        base_distribution: The base (prior) distribution.  Must provide
            ``sample(key=...)``, ``log_prob(value)``, and ``event_shape``.
    """

    transport: AbstractTransport
    base_distribution: eqx.Module

    @property
    def event_shape(self):
        return self.base_distribution.event_shape

    @eqx.filter_jit
    def sample(self, key, **kwargs):
        """Draw a sample from the pushforward distribution.

        Args:
            key: PRNG key for base-distribution sampling and transport.
            **kwargs: Forwarded to ``transport.forward``.

        Returns:
            A sample from the pushforward distribution.
        """
        key_s, key_t = jax.random.split(key)
        x0 = self.base_distribution.sample(key=key_s)
        kw = dict(kwargs)
        if "key" not in kw:
            kw["key"] = key_t
        return self.transport.forward(x0, **kw)

    @eqx.filter_jit
    def sample_and_log_weight(self, key, **kwargs):
        """Draw a sample and compute its log-weight.

        Args:
            key: PRNG key.  A subkey is split for the transport if
                ``key`` is not already in ``kwargs``.
            **kwargs: Forwarded to :meth:`push_forward_and_log_weight`.

        Returns:
            Tuple ``(y, log_weight)``.
        """
        key_s, key_t = jax.random.split(key)
        x0 = self.base_distribution.sample(key=key_s)
        if "key" not in kwargs:
            kwargs = dict(kwargs, key=key_t)
        return self.push_forward_and_log_weight(x0, **kwargs)

    @eqx.filter_jit
    def push_forward_and_log_weight(self, x, **kwargs):
        """Push a base sample through the transport and return log-weight.

        This is the 'pure' function: takes an already-sampled ``x₀`` and
        returns ``(y, log_weight)`` where
        ``log_weight = log q₀(x₀) + c`` and ``c`` is the transport's
        contribution.

        Args:
            x: A sample from the base distribution.
            **kwargs: Forwarded to ``transport.forward_and_log_weight``.

        Returns:
            Tuple ``(y, log_weight)``.
        """
        y, c = self.transport.forward_and_log_weight(x, **kwargs)
        log_q0 = self.base_distribution.log_prob(x)
        return y, log_q0 + c
