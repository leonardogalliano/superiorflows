"""State-context partition for conditional normalising flows.

Generalises equinox's ``eqx.partition`` to support element-level index
masks, enabling conditional updates on subsets of degrees of freedom
(patches).

Three mask types at each leaf of the mask PyTree:

- ``True`` (bool scalar): entire leaf is dynamic — identical to
  ``eqx.partition``.
- ``False`` (bool scalar): entire leaf is static — identical to
  ``eqx.partition``.
- ``jnp.ndarray`` (1-D integer array): element-level selection along the
  first axis of the corresponding leaf. Dynamic gets ``leaf[indices]``
  (shape ``(n, ...)``), static gets ``leaf[complement]`` (shape
  ``(N-n, ...)``).

When all mask leaves are scalar booleans, the output is exactly equivalent
to ``eqx.partition`` — zero overhead, zero behavioural change.
"""

from typing import Any, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp

__all__ = ["state_context_partition", "merge_state", "reconstruct_state", "PartitionSpec"]


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
