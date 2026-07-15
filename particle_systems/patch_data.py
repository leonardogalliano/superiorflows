"""Data pipeline transforms for patch-based training.

This module provides grain transforms to extract patches from full particle
system configurations, ensuring the processed sub-systems correctly follow
the conditional target distribution and match the flow's base distribution.
"""

import jax
import jax.numpy as jnp
import numpy as np

from particle_systems.particle_system import ParticleSystem, _solve_ot_single
from particle_systems.patch_updater import GaussianPatchBase
from particle_systems.selection import random_point_select


class ExtractAndShiftSubsystem:
    """Grain transform: Extract a sub-system and center it at L/2.

    For each full configuration in the batch:
    1. Selects an anchor and identifies patch/context particles.
    2. Extracts the sub-system (patch + context).
    3. Shifts coordinates so the anchor is exactly at L/2, wrapping modulo L.

    This ensures that the target sub-systems are distributed around L/2,
    allowing the use of a fixed base distribution (Gaussian centered at L/2)
    in the downstream losses, avoiding the need for dynamic per-sample bases.
    """

    def __init__(self, N_patch, N_context, r_patch, r_context, sigma_r=0.0, seed=0):
        self.N_patch = N_patch
        self.N_context = N_context
        self.r_patch = r_patch
        self.r_context = r_context
        self.sigma_r = sigma_r
        self.seed = seed
        # JIT-compile the selection for fast execution on CPU threads
        self._select_fn = jax.jit(random_point_select(N_patch, N_context, r_patch, r_context, sigma_r))

    def __call__(self, index, x):
        with jax.default_device(jax.devices("cpu")[0]):
            key = jax.random.fold_in(jax.random.key(self.seed), index)
            sel = self._select_fn(key, x)

            active_idx = jnp.concatenate([sel.patch_idx, sel.context_idx])
            sub_pos = x.positions[active_idx]
            sub_spec = x.species[active_idx]

            L = float(x.box[0])
            center = L / 2.0
            shift = center - sel.anchor
            shifted_pos = (sub_pos + shift) % L

            return (ParticleSystem(positions=shifted_pos, species=sub_spec, box=x.box), sel.n_patch, sel.n_context)


class CouplePatchBase:
    """Grain transform: Generate base samples for Stochastic Interpolant training.

    Applied **after** batching via ``map_with_index``.  Receives a tuple
    ``(x1_sub, n_patch, n_context)`` and produces ``(x0, x1, n_patch, n_context)``
    where ``x0`` has base-sampled patch DOFs and shared context.
    """

    def __init__(self, N_patch, d, L, sigma, seed=0):
        self.N_patch = N_patch
        self.d = d
        self.L = L
        self.base_dist = GaussianPatchBase(
            center=jnp.full((d,), L / 2.0),
            sigma=sigma,
            N_patch=N_patch,
            d=d,
            L=L,
        )
        self.seed = seed

    def __call__(self, index, batch):
        x1_sub, n_patch, n_context = batch
        batch_size = x1_sub.positions.shape[0]

        with jax.default_device(jax.devices("cpu")[0]):
            keys = jax.random.split(jax.random.fold_in(jax.random.key(self.seed), index), batch_size)
            base_patches = jax.vmap(self.base_dist.sample)(keys).positions

            context_pos = x1_sub.positions[:, self.N_patch :]
            x0_pos = jnp.concatenate([base_patches, context_pos], axis=1)

            x0_sub = ParticleSystem(positions=x0_pos, species=x1_sub.species, box=x1_sub.box)

        return (x0_sub, x1_sub, n_patch, n_context)


class PatchEquivariantOptimalTransport:
    """Equivariant Optimal Transport restricted to the patch DOFs.

    The context particles are static and must not be permuted or transported.
    This applies OT only to the first `N_patch` particles.
    """

    def __init__(self, N_patch):
        self.N_patch = N_patch

    def __call__(self, batch):
        x0, x1, n_patch, n_context = batch

        pos_0 = np.asarray(x0.positions)
        pos_1 = np.asarray(x1.positions)
        species = np.asarray(x1.species)
        box = np.asarray(x1.box)

        B = pos_0.shape[0]
        aligned_positions = np.copy(pos_0)

        for i in range(B):
            # Only apply OT to the first N_patch particles
            p0_patch = pos_0[i, : self.N_patch]
            p1_patch = pos_1[i, : self.N_patch]
            spec_patch = species[i, : self.N_patch]

            aligned_patch, _ = _solve_ot_single(p0_patch, p1_patch, spec_patch, spec_patch, box[i])
            aligned_positions[i, : self.N_patch] = aligned_patch

        x0_aligned = ParticleSystem(positions=jnp.asarray(aligned_positions), species=x0.species, box=x0.box)
        return (x0_aligned, x1, n_patch, n_context)
