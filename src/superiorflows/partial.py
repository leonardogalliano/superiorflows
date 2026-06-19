"""Partial base distribution, state partitioning, and selection protocols.

Provides the implementation of state-context partitioning, selection protocols,
and the partial base distribution/updater for conditional sampling.
"""

from typing import Any, List, NamedTuple, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = [
    "PartitionSpec",
    "state_context_partition",
    "merge_state",
    "reconstruct_state",
    "uniform_index_selection",
    "fixed_selection",
    "PartialBase",
    "PartialFlowUpdater",
]


class PartitionSpec(NamedTuple):
    """Specification of a state-context partition.

    Used by :func:`merge_state` and :func:`reconstruct_state`.
    """

    treedef: Any
    leaf_info: List[Tuple[str, Any]]
    index_meta: Any


def _compute_complement(indices, total_size):
    """Compute the complement of ``indices`` within ``range(total_size)``.

    JIT-compatible: the output shape ``(total_size - len(indices),)`` is
    static because both ``total_size`` and ``len(indices)`` are known at
    trace time.
    """
    is_selected = jnp.zeros(total_size, dtype=bool).at[indices].set(True)
    marked = jnp.where(is_selected, total_size, jnp.arange(total_size))
    return jnp.sort(marked)[: total_size - indices.shape[0]]


def _is_scalar_bool(m):
    """Check whether ``m`` is a scalar boolean (Python bool or 0-d array)."""
    if isinstance(m, bool):
        return True
    if hasattr(m, "shape") and m.shape == ():
        return True
    return False


def state_context_partition(x, mask):
    """Partition a state PyTree into dynamic and static (context) parts.

    This is a drop-in generalisation of ``eqx.partition`` that additionally
    supports **index-array masks** for element-level selection within
    array leaves.

    Args:
        x: State PyTree.
        mask: Either a **PyTree** matching the structure of ``x`` with
            leaves that are scalar booleans or 1-D integer index arrays,
            or a **callable** ``leaf → bool`` applied to each leaf (for
            backwards compatibility with the default ``dynamic_mask``).

    Returns:
        ``(dynamic, static, spec)`` where:

        - ``dynamic``: PyTree with same structure as ``x``. Leaves
          selected by the mask contain the dynamic data; others are
          ``None``.
        - ``static``: complementary PyTree. For scalar-bool masks this
          mirrors ``eqx.partition``. For index masks, the static leaf
          contains the complement elements ``leaf[complement_indices]``.
        - ``spec``: opaque partition specification used by
          :func:`merge_state` to reassemble the full state.
    """
    x_flat, x_treedef = jax.tree.flatten(x)

    if callable(mask) and not hasattr(mask, "__jax_tree_flatten__"):
        m_flat = [mask(leaf) for leaf in x_flat]
    else:
        m_flat, m_treedef = jax.tree.flatten(mask)
        if m_treedef != x_treedef:
            raise ValueError(f"Mask tree structure does not match state. State has {x_treedef}, mask has {m_treedef}.")

    dyn_leaves = []
    ctx_leaves = []
    leaf_info = []
    meta_leaves = []

    for leaf, m in zip(x_flat, m_flat):
        if _is_scalar_bool(m):
            if bool(m):
                dyn_leaves.append(leaf)
                ctx_leaves.append(None)
                leaf_info.append(("full_dynamic", None))
                meta_leaves.append(None)
            else:
                dyn_leaves.append(None)
                ctx_leaves.append(leaf)
                leaf_info.append(("full_static", None))
                meta_leaves.append(None)
        else:
            indices = jnp.asarray(m)
            complement = _compute_complement(indices, leaf.shape[0])
            dyn_leaves.append(leaf[indices])
            ctx_leaves.append(leaf[complement])
            leaf_info.append(("indexed", (indices, complement)))
            meta_leaves.append((indices, complement))

    dynamic = x_treedef.unflatten(dyn_leaves)
    static = x_treedef.unflatten(ctx_leaves)
    if all(meta is None for meta in meta_leaves):
        index_meta = None
    else:
        index_meta = x_treedef.unflatten(meta_leaves)

    return dynamic, static, PartitionSpec(x_treedef, leaf_info, index_meta)


def merge_state(new_dynamic, original_x, spec):
    """Scatter updated dynamic leaves back into the original state.

    For scalar-bool partitions this is equivalent to
    ``eqx.combine(new_dynamic, original_static)``.  For index-mask
    partitions, the updated elements are scattered at their original
    positions within the full array.

    Args:
        new_dynamic: PyTree with the same structure as ``original_x``.
            Dynamic leaves contain updated values; static leaves are
            ``None``.
        original_x: The original (un-partitioned) state.
        spec: Partition specification returned by
            :func:`state_context_partition`.

    Returns:
        Merged state PyTree with updated dynamic values.
    """
    if isinstance(spec, PartitionSpec):
        treedef = spec.treedef
        leaf_info = spec.leaf_info
    else:
        treedef, leaf_info = spec
    new_flat = jax.tree.flatten(new_dynamic, is_leaf=lambda n: n is None)[0]
    orig_flat = jax.tree.flatten(original_x)[0]

    merged = []
    for new_leaf, orig_leaf, (kind, data) in zip(new_flat, orig_flat, leaf_info):
        if kind == "full_dynamic":
            merged.append(new_leaf)
        elif kind == "full_static":
            merged.append(orig_leaf)
        elif kind == "indexed":
            indices, _complement = data
            merged.append(orig_leaf.at[indices].set(new_leaf))

    return treedef.unflatten(merged)


def reconstruct_state(dynamic, static, index_meta):
    """Reconstruct the full state from partitioned dynamic and static parts.

    Uses ``index_meta`` to scatter dynamic and static elements back into their
    original positions.

    Args:
        dynamic: Dynamic PyTree.
        static: Static PyTree.
        index_meta: PyTree matching the state structure with ``None`` or
            ``(indices, complement)`` leaves.

    Returns:
        Reconstructed state PyTree.
    """
    if index_meta is None:

        def combine_leaf(d, c):
            return d if d is not None else c

        return jax.tree.map(combine_leaf, dynamic, static, is_leaf=lambda x: x is None)

    def reconstruct_leaf(d, c, meta):
        if meta is None:
            return d if d is not None else c
        indices, complement = meta
        total_size = d.shape[0] + c.shape[0]
        out_shape = (total_size,) + d.shape[1:]
        out = jnp.zeros(out_shape, dtype=d.dtype)
        return out.at[indices].set(d).at[complement].set(c)

    def is_leaf_meta(x):
        if x is None:
            return True
        if isinstance(x, tuple) and len(x) == 2 and not isinstance(x[0], tuple):
            return True
        return False

    return jax.tree.map(reconstruct_leaf, dynamic, static, index_meta, is_leaf=is_leaf_meta)


def uniform_index_selection(n: int):
    """Select ``n`` elements uniformly at random along axis 0.

    Returns a callable ``(key, x) → indices`` where ``indices`` is a
    sorted 1-D integer array of shape ``(n,)``.

    The caller is responsible for wrapping the raw index array into the
    appropriate PyTree mask if the state is a structured PyTree (e.g.
    ``ParticleSystem(positions=indices, species=False, box=False)``).

    Args:
        n: Number of elements to select (static, determines output shape).

    Returns:
        A selection protocol callable.

    Example:
        >>> protocol = uniform_index_selection(4)
        >>> indices = protocol(jax.random.key(0), jnp.arange(10))
        >>> indices.shape
        (4,)
    """

    def protocol(key, x):
        first_leaf = jax.tree.leaves(x)[0]
        total = first_leaf.shape[0]
        indices = jax.random.choice(key, total, shape=(n,), replace=False)
        return jnp.sort(indices)

    return protocol


def fixed_selection(mask):
    """Deterministic selection — always use the same mask.

    Args:
        mask: A fixed dynamic mask (PyTree of booleans / index arrays,
            or a plain index array).

    Returns:
        A selection protocol callable that ignores the key and state.
    """
    return lambda key, x: mask


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


class PartialFlowUpdater(eqx.Module):
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
        from superiorflows.flow import Flow

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
        from superiorflows.flow import Flow

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
        from superiorflows.flow import Flow

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
        from superiorflows.flow import Flow

        partial_base = PartialBase(self.base_distribution, x, dynamic_mask)
        flow = Flow(self.bijector, partial_base)
        return flow.push_forward_and_log_prob(x, dynamic_mask=dynamic_mask, **kwargs)
