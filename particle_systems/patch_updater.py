"""Patch-based local updater for particle systems.

Orchestrates the gather → partition → flow → scatter pipeline for
patch-based MCMC on periodic flat tori. Connects the selection protocol
(which identifies patch and context particles) with the core library's
:class:`~superiorflows.partial.PartialFlowUpdater` (which handles the
ODE-based conditional sampling).

The two main entry points are:

- :meth:`PatchUpdater.update_and_log_prob`: propose a new patch configuration
  and compute the forward log-density log q(x'_p | x_c).
- :meth:`PatchUpdater.log_prob`: evaluate the log-density of an existing
  patch configuration given its selection.

These methods are deliberately general and do not compute the full MH
acceptance ratio. A downstream MCMC sampler will compose them with
energy differences and selection probability ratios.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from particle_systems.particle_system import ParticleSystem
from superiorflows.partial import PartialFlowUpdater

__all__ = ["UniformPatchBase", "PatchUpdater"]


class UniformPatchBase(eqx.Module):
    """Uniform base distribution for patch sampling.

    Samples N_patch positions uniformly in [0, L)^d. Species and box are
    not sampled — they are inherited from the context via
    :class:`~superiorflows.partial.PartialBase`.

    The log-probability is the uniform density on the flat torus:
    −N_patch · d · log(L).

    Attributes:
        N_patch: Number of patch particles.
        d: Spatial dimension.
        L: Box side length (assumes cubic box).
    """

    N_patch: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    L: float = eqx.field(static=True)

    def sample(self, key):
        """Draw N_patch uniform positions in [0, L)^d.

        Returns:
            ParticleSystem with ``positions`` of shape ``(N_patch, d)`` and
            ``species=None``, ``box=None`` (matching the dynamic partition).
        """
        pos = jax.random.uniform(key, shape=(self.N_patch, self.d), maxval=self.L)
        return ParticleSystem(positions=pos, species=None, box=None)

    def log_prob(self, value):
        """Log-density of uniform distribution on [0, L)^{N_patch × d}.

        Args:
            value: ParticleSystem with positions of shape ``(N_patch, d)``.

        Returns:
            Scalar log-probability −N_patch · d · log(L).
        """
        return -self.N_patch * self.d * jnp.log(self.L)


class PatchUpdater(eqx.Module):
    """Orchestrate patch-based local updates for particle systems.

    Workflow::

        full_x → [selection_protocol] → x_active (N_patch + N_context)
               → [PartialFlowUpdater] → x_active' (inner partition: patch vs context)
               → [scatter] → full_x'

    The inner :class:`~superiorflows.partial.PartialFlowUpdater` handles
    the two-way split of the active sub-system into patch (dynamic) and
    context (static) via the standard ``state_context_partition`` machinery.

    Attributes:
        partial_updater: Wraps the bijector and base distribution.
        selection_fn: Callable ``(key, x) → SelectionResult``.
        N_patch: Static patch allocation size.
        N_context: Static context allocation size.
    """

    partial_updater: PartialFlowUpdater
    selection_fn: object = eqx.field(static=True)
    N_patch: int = eqx.field(static=True)
    N_context: int = eqx.field(static=True)

    def __init__(self, bijector, base_distribution, selection_fn, N_patch, N_context):
        self.partial_updater = PartialFlowUpdater(bijector, base_distribution)
        object.__setattr__(self, "selection_fn", selection_fn)
        self.N_patch = N_patch
        self.N_context = N_context

    def update_and_log_prob(self, x, key, **kwargs):
        """Propose a new patch and compute its log-density.

        Draws a random patch/context partition via the selection protocol,
        samples from the base distribution, pushes through the flow, and
        returns the updated full configuration with the forward log-density.

        Args:
            x: Full particle system configuration.
            key: PRNG key.
            **kwargs: Forwarded to the bijector (e.g. ``key`` for Hutchinson).

        Returns:
            ``(x_new, log_q, selection_result)``: updated full config,
            forward log-density log q(x'_p | x_c), and the selection
            result for backward density evaluation.
        """
        k_sel, k_flow = jax.random.split(key)
        sel = self.selection_fn(k_sel, x)
        x_active = self._build_subsystem(x, sel)
        mask = self._inner_mask()

        x_active_new, log_q = self.partial_updater.update_and_log_prob(
            x_active, mask, k_flow, args=(sel.n_patch, sel.n_context), **kwargs
        )
        x_new = self._scatter_patch(x, x_active_new, sel)

        return x_new, log_q, sel

    def log_prob(self, x, sel, **kwargs):
        """Compute the flow log-density for a given configuration and selection.

        Evaluates log q(x_p | x_c) by extracting the sub-system specified
        by ``sel``, applying the inner partition, and computing the flow
        log-probability via the inverse ODE + change of variables.

        Args:
            x: Full particle system configuration.
            sel: :class:`~particle_systems.selection.SelectionResult`
                specifying which particles form the patch and context.
            **kwargs: Forwarded to the bijector.

        Returns:
            Scalar log-probability of the patch under the flow.
        """
        x_active = self._build_subsystem(x, sel)
        mask = self._inner_mask()
        return self.partial_updater.log_prob(x_active, mask, args=(sel.n_patch, sel.n_context), **kwargs)

    def _inner_mask(self):
        """ParticleSystem-shaped mask selecting patch positions as dynamic DOFs."""
        return ParticleSystem(
            positions=jnp.arange(self.N_patch),
            species=False,
            box=False,
        )

    def _build_subsystem(self, x, sel):
        """Extract the (N_patch + N_context)-particle sub-system.

        Concatenates patch and context indices and gathers positions, species,
        and box. The layout is [patch_positions; context_positions] so that
        the inner mask ``arange(N_patch)`` correctly selects the patch.

        Args:
            x: Full particle system.
            sel: Selection result with ``patch_idx`` and ``context_idx``.

        Returns:
            ParticleSystem with ``M = N_patch + N_context`` particles.
        """
        active_idx = jnp.concatenate([sel.patch_idx, sel.context_idx])
        return ParticleSystem(
            positions=x.positions[active_idx],
            species=x.species[active_idx],
            box=x.box,
        )

    def _scatter_patch(self, x, x_active_new, sel):
        """Scatter updated patch positions back into the full configuration.

        Handles the padding convention correctly: when patch_idx contains
        duplicated entries (padded slots), we ensure all writes to the
        same index produce the correct updated value. Since JAX's
        ``at[].set()`` uses last-write-wins for duplicate indices and
        patch_idx is sorted (padded entries come after valid ones), we
        replace padded slot values with the last valid particle's updated
        position. This way, the last write to a duplicated index is always
        the correct value.

        Args:
            x: Original full particle system.
            x_active_new: Updated active sub-system from the flow.
            sel: Selection result with ``patch_idx`` and ``n_patch``.

        Returns:
            Full ParticleSystem with only patch positions updated.
        """
        new_patch_pos = x_active_new.positions[: self.N_patch]

        last_idx = jnp.maximum(sel.n_patch - 1, 0)
        last_valid_pos = new_patch_pos[last_idx]
        valid_mask = jnp.arange(self.N_patch) < sel.n_patch
        safe_pos = jnp.where(valid_mask[:, None], new_patch_pos, last_valid_pos[None, :])

        new_positions = x.positions.at[sel.patch_idx].set(safe_pos)
        return ParticleSystem(positions=new_positions, species=x.species, box=x.box)
