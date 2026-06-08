"""Abstract bijector interface for invertible transformations.

Provides a base class for bijectors (invertible, differentiable maps) that
operate on arbitrary pytree states.  The partition/merge logic for dynamic
masking lives here — concrete subclasses need only implement the ``_``
prefixed methods (``_forward``, ``_inverse``, ``_forward_and_log_det``,
``_inverse_and_log_det``) which receive an already-partitioned dynamic
state and a context ``args``.

All public methods accept ``**kwargs`` so that concrete implementations
may receive additional arguments without modifying the abstract interface —
for example, PRNG keys for stochastic log-det estimators, or solver
overrides for ODE-based flows.

The ``dynamic_mask`` can be **overridden per call** by passing
``dynamic_mask=...`` as a keyword argument to any public method.
"""

from typing import Callable

import equinox as eqx
import jax

from superiorflows.partition import merge_state, state_context_partition

__all__ = ["AbstractBijector", "DistreqxBijectorWrapper"]


def _pack_args(ctx, user_args):
    """Combine partition context with user-supplied args."""
    if user_args is not None:
        return (ctx, user_args)
    return ctx


class AbstractBijector(eqx.Module):
    """Abstract invertible transformation on arbitrary pytree states.

    Handles dynamic-mask partitioning at this level so that concrete
    subclasses only ever see the dynamic degrees of freedom.

    Subclasses must implement :meth:`_forward_and_log_det` and
    :meth:`_inverse_and_log_det`.  The remaining ``_forward`` and
    ``_inverse`` are provided automatically but may be overridden for
    efficiency.

    Sign convention: ``forward_and_log_det`` returns
    ``(y, log|det J_f(x)|)`` and ``inverse_and_log_det`` returns
    ``(x, log|det J_{f^{-1}}(y)|)``.  These satisfy
    ``log|det J_f(x)| = -log|det J_{f^{-1}}(f(x))|``.

    Attributes:
        dynamic_mask: Default mask selecting which leaves (or elements
            within a leaf) are treated as dynamic degrees of freedom.
            Can be a callable ``leaf → bool``, a PyTree of booleans, or
            a PyTree with integer index arrays for element-level
            selection.  See :func:`state_context_partition`.
    """

    dynamic_mask: Callable = eqx.field(
        default=lambda x: jax.tree.map(eqx.is_inexact_array, x),
        kw_only=True,
    )

    # --- Public (final) interface: partition → delegate → merge ----------

    @eqx.filter_jit
    def forward(self, x, **kwargs):
        """Compute y = f(x).

        Partitions ``x`` via the dynamic mask, delegates to
        :meth:`_forward`, and merges the result back into the full
        state.  Pass ``dynamic_mask=...`` to override the default mask.
        """
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        x_dyn, ctx, spec = state_context_partition(x, mask)
        args = _pack_args(ctx, kw.pop("args", None))
        y_dyn = self._forward(x_dyn, args=args, **kw)
        return merge_state(y_dyn, x, spec)

    @eqx.filter_jit
    def inverse(self, y, **kwargs):
        """Compute x = f⁻¹(y)."""
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        y_dyn, ctx, spec = state_context_partition(y, mask)
        args = _pack_args(ctx, kw.pop("args", None))
        x_dyn = self._inverse(y_dyn, args=args, **kw)
        return merge_state(x_dyn, y, spec)

    @eqx.filter_jit
    def forward_and_log_det(self, x, **kwargs):
        """Compute (y, log|det J_f(x)|)."""
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        x_dyn, ctx, spec = state_context_partition(x, mask)
        args = _pack_args(ctx, kw.pop("args", None))
        y_dyn, logdet = self._forward_and_log_det(x_dyn, args=args, **kw)
        return merge_state(y_dyn, x, spec), logdet

    @eqx.filter_jit
    def inverse_and_log_det(self, y, **kwargs):
        """Compute (x, log|det J_{f⁻¹}(y)|)."""
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        y_dyn, ctx, spec = state_context_partition(y, mask)
        args = _pack_args(ctx, kw.pop("args", None))
        x_dyn, logdet = self._inverse_and_log_det(y_dyn, args=args, **kw)
        return merge_state(x_dyn, y, spec), logdet

    def forward_log_det_jacobian(self, x, **kwargs):
        """Compute log|det J_f(x)| without returning y."""
        _, logdet = self.forward_and_log_det(x, **kwargs)
        return logdet

    def inverse_log_det_jacobian(self, y, **kwargs):
        """Compute log|det J_{f⁻¹}(y)| without returning x."""
        _, logdet = self.inverse_and_log_det(y, **kwargs)
        return logdet

    # --- Abstract extension points (implement in subclasses) -------------

    def _forward(self, x, *, args=None, **kwargs):
        """Compute y = f(x) on already-partitioned dynamic state.

        Args:
            x: Dynamic state (static leaves are ``None``).
            args: Context from :func:`state_context_partition`, possibly
                packed with user-supplied args as ``(ctx, user_args)``.
            **kwargs: Subclass-specific parameters.
        """
        y, _ = self._forward_and_log_det(x, args=args, **kwargs)
        return y

    def _inverse(self, y, *, args=None, **kwargs):
        """Compute x = f⁻¹(y) on already-partitioned dynamic state."""
        x, _ = self._inverse_and_log_det(y, args=args, **kwargs)
        return x

    def _forward_and_log_det(self, x, *, args=None, **kwargs):
        """Compute (y, log|det J_f(x)|) on the dynamic state.

        **Must be implemented by subclasses.**
        """
        raise NotImplementedError

    def _inverse_and_log_det(self, y, *, args=None, **kwargs):
        """Compute (x, log|det J_{f⁻¹}(y)|) on the dynamic state.

        **Must be implemented by subclasses.**
        """
        raise NotImplementedError


class DistreqxBijectorWrapper(AbstractBijector):
    """Adapt a distreqx bijector to the superiorflows interface.

    Wraps any object that exposes the distreqx bijector protocol
    (``forward``, ``inverse``, ``forward_and_log_det``,
    ``inverse_and_log_det``) so that it can be used wherever an
    :class:`AbstractBijector` is expected.
    """

    _bijector: eqx.Module

    def _forward(self, x, *, args=None, **kwargs):
        return self._bijector.forward(x)

    def _inverse(self, y, *, args=None, **kwargs):
        return self._bijector.inverse(y)

    def _forward_and_log_det(self, x, *, args=None, **kwargs):
        return self._bijector.forward_and_log_det(x)

    def _inverse_and_log_det(self, y, *, args=None, **kwargs):
        return self._bijector.inverse_and_log_det(y)
