"""Partial base distribution and updater for conditional sampling.

Provides the distribution and inference layer for partial updates: given
a full configuration ``x = (s, c)``, resample only the state ``s``
whilst holding the context ``c`` fixed.

The :class:`PartialBase` wraps an unconditional base distribution and
a context template, producing full-dimensional samples where the dynamic
DOFs are drawn from the base and the static DOFs are copied from the
context.  :class:`PartialUpdater` composes a bijector with a base
distribution to provide ``update``, ``log_prob``, and
``update_and_log_prob`` for MCMC inner loops and training losses.

The :class:`Flow` class requires no modifications — all partial-update
logic is encapsulated here.
"""

import equinox as eqx

from superiorflows.flow import Flow
from superiorflows.partition import merge_state, state_context_partition

__all__ = ["PartialBase", "PartialUpdater"]


class PartialBase(eqx.Module):
    """Base distribution for partial sampling on a subset of DOFs.

    Wraps an unconditional base distribution that produces samples
    matching the dynamic-DOF shape.  :meth:`sample` merges the base
    sample with the static context to produce a full-dimensional state.
    :meth:`log_prob` extracts the dynamic DOFs and evaluates only over
    those.

    For genuinely conditional bases :math:`q_0(s \\mid c)`, subclass
    and override :meth:`sample` / :meth:`log_prob` to forward the
    static context to the base distribution.

    Attributes:
        base_distribution: Distribution over the dynamic DOFs.
            Must provide ``sample(key=...) → s`` and
            ``log_prob(s) → scalar`` where ``s`` has the dynamic-DOF
            shape (e.g. ``(n, d)`` for ``n`` selected particles).
        context: Full state serving as template.  Static DOFs are
            copied from here into sampled states.
        dynamic_mask: Mask selecting which DOFs are dynamic.  Same
            format as accepted by :func:`state_context_partition`:
            a PyTree of scalar booleans and/or 1-D integer index
            arrays.
    """

    base_distribution: eqx.Module
    context: eqx.Module
    dynamic_mask: object

    def sample(self, key):
        """Draw ``s ~ q₀(·)`` and merge with context → full state."""
        s = self.base_distribution.sample(key=key)
        _, _, spec = state_context_partition(self.context, self.dynamic_mask)
        return merge_state(s, self.context, spec)

    def log_prob(self, x):
        """Evaluate ``log q₀(s)`` where ``s`` is the dynamic part of ``x``."""
        s, _, _ = state_context_partition(x, self.dynamic_mask)
        return self.base_distribution.log_prob(s)

    def sample_and_log_prob(self, key):
        """Draw ``s`` and return ``(full_state, log_prob)``."""
        x = self.sample(key)
        return x, self.log_prob(x)


class PartialUpdater(eqx.Module):
    """Partial updater: update state ``s`` whilst conditioning on context ``c``.

    Composes a bijector with a base distribution.  Each method receives
    a ``dynamic_mask`` that selects which DOFs are state vs context,
    enabling the same updater to be called with different masks (e.g.
    different random particle selections per sample in a batch).

    Used directly for inference (MCMC inner loops) and internally by
    the training losses when a selection protocol is active.

    Attributes:
        bijector: The invertible transformation (e.g.
            :class:`ODEBijector`).
        base_distribution: Distribution over the dynamic DOFs.
    """

    bijector: eqx.Module
    base_distribution: eqx.Module

    def log_prob(self, x, dynamic_mask, **kwargs):
        """Evaluate ``log q_θ(s | c)`` for a given full state and mask.

        Args:
            x: Full state.
            dynamic_mask: Mask selecting the dynamic DOFs.
            **kwargs: Forwarded to the bijector (e.g. ``key`` for
                Hutchinson trace estimation).

        Returns:
            Scalar log-probability of the dynamic DOFs under the flow.
        """
        partial_base = PartialBase(self.base_distribution, x, dynamic_mask)
        flow = Flow(self.bijector, partial_base)
        return flow.log_prob(x, dynamic_mask=dynamic_mask, **kwargs)

    def update(self, x, dynamic_mask, key, **kwargs):
        """Update the dynamic DOFs of ``x``, keeping context fixed.

        Draws from the partial base, pushes through the bijector, and
        merges the result with the static context.

        Args:
            x: Full state (context is extracted from this).
            dynamic_mask: Mask selecting the dynamic DOFs.
            key: PRNG key for base-distribution sampling.
            **kwargs: Forwarded to the bijector.

        Returns:
            Updated full state with new dynamic DOFs.
        """
        partial_base = PartialBase(self.base_distribution, x, dynamic_mask)
        flow = Flow(self.bijector, partial_base)
        return flow.sample(key=key, dynamic_mask=dynamic_mask, **kwargs)

    def update_and_log_prob(self, x, dynamic_mask, key, **kwargs):
        """Update dynamic DOFs and return the log-probability.

        Args:
            x: Full state.
            dynamic_mask: Mask selecting the dynamic DOFs.
            key: PRNG key.
            **kwargs: Forwarded to the bijector.

        Returns:
            Tuple ``(x_updated, log_prob)``.
        """
        partial_base = PartialBase(self.base_distribution, x, dynamic_mask)
        flow = Flow(self.bijector, partial_base)
        return flow.sample_and_log_prob(key=key, dynamic_mask=dynamic_mask, **kwargs)

    def push_forward_and_log_prob(self, x, dynamic_mask, **kwargs):
        """Push a partial base sample through the bijector.

        Assumes ``x`` already contains a base sample in the dynamic
        DOFs (i.e. the dynamic part of ``x`` was drawn from the base
        distribution).

        Args:
            x: Full state with base sample in dynamic DOFs.
            dynamic_mask: Mask selecting the dynamic DOFs.
            **kwargs: Forwarded to the bijector.

        Returns:
            Tuple ``(x_pushed, log_prob)``.
        """
        partial_base = PartialBase(self.base_distribution, x, dynamic_mask)
        flow = Flow(self.bijector, partial_base)
        return flow.push_forward_and_log_prob(x, dynamic_mask=dynamic_mask, **kwargs)
