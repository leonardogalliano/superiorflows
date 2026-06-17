"""Selection protocols for partial updates.

A selection protocol is a callable ``(key, x) → dynamic_mask`` that
decides which degrees of freedom are treated as state (to be resampled)
versus context (to be held fixed) for a given configuration ``x``.

The returned ``dynamic_mask`` must be compatible with
:func:`superiorflows.partition.state_context_partition`: either a PyTree
of scalar booleans, a PyTree with 1-D integer index arrays, or a plain
1-D index array for flat-array states.

The patch size (number of selected DOFs) must be **static** — known at
trace time — because it determines array shapes under JIT.  Only the
*identity* of which DOFs are selected may vary per call.
"""

import jax
import jax.numpy as jnp

__all__ = ["uniform_index_selection", "fixed_selection"]


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
