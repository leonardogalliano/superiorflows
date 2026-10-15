"""Abstract bijector interface for invertible transformations.

Provides a base class for bijectors (invertible, differentiable maps) that
operate on arbitrary pytree states.  Inherits partitioning logic from
:class:`AbstractTransport` — concrete subclasses need only implement the
``_`` prefixed methods (``_forward_and_log_det``, ``_inverse_and_log_det``)
which receive an already-partitioned dynamic state and a context ``args``.

All public methods accept ``**kwargs`` so that concrete implementations
may receive additional arguments without modifying the abstract interface —
for example, PRNG keys for stochastic log-det estimators, or solver
overrides for ODE-based flows.

The ``dynamic_mask`` can be **overridden per call** by passing
``dynamic_mask=...`` as a keyword argument to any public method.
"""

import equinox as eqx

from superiorflows.transport import AbstractTransport, _pack_args, _partition_and_apply

__all__ = ["AbstractBijector", "_pack_args"]


class AbstractBijector(AbstractTransport):
    """Abstract invertible transformation on arbitrary pytree states.

    Extends :class:`AbstractTransport` with inverse operations and the
    bijector-specific ``forward_and_log_det`` / ``inverse_and_log_det``
    methods.

    The unified :meth:`_forward_and_log_weight` bridges to
    :meth:`_forward_and_log_det` via ``c = −log_det``, matching the
    convention ``log_weight = log q₀ + c = log q₀ − log|det J|``.
    Symmetrically, :meth:`_backward_and_log_weight` returns
    ``c = −log|det J_{f⁻¹}(y)|``, the contribution of the map applied
    backward.

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

    # --- Unified interface bridge ---

    def _forward_and_log_weight(self, x, *, args=None, **kwargs):
        """Bridge to the unified transport interface.

        Returns ``(y, c)`` where ``c = −log|det J|``, so that
        ``log_weight = log q₀ + c = log q₀ − log|det J|``.
        """
        y, log_det = self._forward_and_log_det(x, args=args, **kwargs)
        return y, -log_det

    def _backward(self, y, *, args=None, **kwargs):
        """Backward transport delegates to inverse for bijectors."""
        return self._inverse(y, args=args, **kwargs)

    def _backward_and_log_weight(self, y, *, args=None, **kwargs):
        """Bridge backward transport to :meth:`_inverse_and_log_det`.

        Returns ``(x, c)`` with ``c = −log|det J_{f⁻¹}(y)| = log|det J_f(x)|``:
        the map applied backward is ``f⁻¹``, so ``c`` follows the same rule
        as the forward bridge.  It is the deterministic limit of the SDE
        backward path action ``S⁻``, so ``log q₀(x) − c = log q_f(y)`` is the
        model density at ``y`` and ``log π(y) + c`` is the log-density at
        ``x`` of ``π`` pulled back through ``f⁻¹``.
        """
        x, inv_logdet = self._inverse_and_log_det(y, args=args, **kwargs)
        return x, -inv_logdet

    # --- Bijector-specific public interface ---

    @eqx.filter_jit
    def inverse(self, y, **kwargs):
        """Compute ``x = f⁻¹(y)``."""
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        user_args = kw.pop("args", None)
        return _partition_and_apply(self._inverse, y, mask, user_args, **kw)

    @eqx.filter_jit
    def forward_and_log_det(self, x, **kwargs):
        """Compute ``(y, log|det J_f(x)|)``."""
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        user_args = kw.pop("args", None)
        return _partition_and_apply(self._forward_and_log_det, x, mask, user_args, **kw)

    @eqx.filter_jit
    def inverse_and_log_det(self, y, **kwargs):
        """Compute ``(x, log|det J_{f⁻¹}(y)|)``."""
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        user_args = kw.pop("args", None)
        return _partition_and_apply(self._inverse_and_log_det, y, mask, user_args, **kw)

    def forward_log_det_jacobian(self, x, **kwargs):
        """Compute ``log|det J_f(x)|`` without returning ``y``."""
        _, logdet = self.forward_and_log_det(x, **kwargs)
        return logdet

    def inverse_log_det_jacobian(self, y, **kwargs):
        """Compute ``log|det J_{f⁻¹}(y)|`` without returning ``x``."""
        _, logdet = self.inverse_and_log_det(y, **kwargs)
        return logdet

    # --- Abstract extension points (implement in subclasses) ---

    def _forward(self, x, *, args=None, **kwargs):
        """Compute ``y = f(x)`` on already-partitioned dynamic state.

        Args:
            x: Dynamic state (static leaves are ``None``).
            args: Context from :func:`state_context_partition`, possibly
                packed with user-supplied args as ``(ctx, user_args)``.
            **kwargs: Subclass-specific parameters.
        """
        y, _ = self._forward_and_log_det(x, args=args, **kwargs)
        return y

    def _inverse(self, y, *, args=None, **kwargs):
        """Compute ``x = f⁻¹(y)`` on already-partitioned dynamic state."""
        x, _ = self._inverse_and_log_det(y, args=args, **kwargs)
        return x

    def _forward_and_log_det(self, x, *, args=None, **kwargs):
        """Compute ``(y, log|det J_f(x)|)`` on the dynamic state.

        **Must be implemented by subclasses.**
        """
        raise NotImplementedError

    def _inverse_and_log_det(self, y, *, args=None, **kwargs):
        """Compute ``(x, log|det J_{f⁻¹}(y)|)`` on the dynamic state.

        **Must be implemented by subclasses.**
        """
        raise NotImplementedError
