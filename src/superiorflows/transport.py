"""Abstract base class for state transports and partitioning helpers.

Provides :class:`AbstractTransport`, the shared base class for all
transport mechanisms (bijective flows, SDE evolvers, etc.).  The
partition → delegate → merge pipeline is implemented once here and
inherited by all concrete transports.

The helper :func:`_partition_and_apply` eliminates boilerplate in
public methods: it partitions the state via the dynamic mask, packs
the context into ``args``, delegates to a private implementation
method, and merges the result back.
"""

from typing import Callable

import equinox as eqx
import jax

from superiorflows.partial import PartitionSpec, merge_state, state_context_partition

__all__ = ["AbstractTransport"]


def _pack_args(ctx, index_meta, user_args=None):
    """Combine partition context, index mapping metadata, and user-supplied args."""
    if index_meta is None:
        return (ctx, user_args) if user_args is not None else ctx
    return (ctx, (index_meta, user_args)) if user_args is not None else (ctx, index_meta)


def _partition_and_apply(apply_fn, x, dynamic_mask, user_args, **kwargs):
    """Partition state, apply transform on dynamic DOFs, and merge back.

    Handles the full pipeline:

    1. Partition ``x`` into dynamic and static parts via ``dynamic_mask``.
    2. Pack context (static part + index metadata + user args) into ``args``.
    3. Call ``apply_fn(x_dyn, args=args, **kwargs)``.
    4. If the result is a tuple ``(y_dyn, *aux)``, merge ``y_dyn`` back
       into the full state and return ``(y_full, *aux)``.
       Otherwise, merge and return the full state directly.

    Args:
        apply_fn: Private implementation method with signature
            ``(x_dyn, *, args=None, **kwargs) → y_dyn`` or
            ``(x_dyn, *, args=None, **kwargs) → (y_dyn, *aux)``.
        x: Full state PyTree.
        dynamic_mask: Mask selecting dynamic degrees of freedom.
        user_args: Optional user-supplied args to forward.
        **kwargs: Additional keyword arguments forwarded to ``apply_fn``.

    Returns:
        Merged result with the same structure as ``x``, plus any
        auxiliary outputs from ``apply_fn``.
    """
    x_dyn, ctx, spec = state_context_partition(x, dynamic_mask)
    index_meta = spec.index_meta if isinstance(spec, PartitionSpec) else None
    args = _pack_args(ctx, index_meta, user_args)
    res = apply_fn(x_dyn, args=args, **kwargs)
    if isinstance(res, tuple):
        y_dyn, *aux = res
        return (merge_state(y_dyn, x, spec), *aux)
    return merge_state(res, x, spec)


class AbstractTransport(eqx.Module):
    """Abstract base class for all transport mechanisms.

    Handles dynamic-mask partitioning so that concrete subclasses
    only ever see the dynamic degrees of freedom.  The key unified
    method is :meth:`forward_and_log_weight`, returning ``(y, c)``
    where ``c`` is the transport's additive contribution to the
    log-weight:

        ``log_weight(y) = log q₀(x) + c``

    For bijectors: ``c = −log|det J|`` (exact change of variables).
    For SDEs: ``c = Σ(log p⁺ − log p⁻)`` (Girsanov path action).

    Attributes:
        dynamic_mask: Default mask selecting which leaves (or elements)
            are dynamic.  Can be a callable ``leaf → bool``, a PyTree
            of booleans, or a PyTree with integer index arrays.
            Overridable per call via ``dynamic_mask=...`` kwarg.
    """

    dynamic_mask: Callable = eqx.field(
        default=lambda x: jax.tree.map(eqx.is_inexact_array, x),
        kw_only=True,
    )

    # --- Public interface: partition → delegate → merge ---

    @eqx.filter_jit
    def forward(self, x, **kwargs):
        """Compute ``y = f(x)``.

        Partitions ``x`` via the dynamic mask, delegates to
        :meth:`_forward`, and merges the result.  Pass
        ``dynamic_mask=...`` to override the default mask.
        """
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        user_args = kw.pop("args", None)
        return _partition_and_apply(self._forward, x, mask, user_args, **kw)

    @eqx.filter_jit
    def forward_and_log_weight(self, x, **kwargs):
        """Forward transport with log-weight contribution.

        Returns ``(y, c)`` where ``c`` is the transport's additive
        contribution to the log-weight:
        ``log_weight(y) = log q₀(x) + c``.
        """
        kw = dict(kwargs)
        mask = kw.pop("dynamic_mask", self.dynamic_mask)
        user_args = kw.pop("args", None)
        return _partition_and_apply(self._forward_and_log_weight, x, mask, user_args, **kw)

    # --- Abstract extension points ---

    def _forward(self, x, *, args=None, **kwargs):
        """Compute ``y = f(x)`` on pre-partitioned dynamic state.

        Default implementation discards the log-weight from
        :meth:`_forward_and_log_weight`.  Override for efficiency
        when the log-weight is not needed.
        """
        y, _ = self._forward_and_log_weight(x, args=args, **kwargs)
        return y

    def _forward_and_log_weight(self, x, *, args=None, **kwargs):
        """Compute ``(y, c)`` on pre-partitioned dynamic state.

        Must be implemented by subclasses.

        Args:
            x: Dynamic state (static leaves are ``None``).
            args: Context from :func:`state_context_partition`, possibly
                packed with user-supplied args.
            **kwargs: Subclass-specific parameters (e.g. ``key``).

        Returns:
            Tuple ``(y, c)`` where ``y`` is the transformed dynamic
            state and ``c`` is the scalar log-weight contribution.
        """
        raise NotImplementedError
