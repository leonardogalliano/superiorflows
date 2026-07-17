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
import numpy as np

from particle_systems.particle_system import ParticleSystem, _solve_ot_single, log_map
from particle_systems.selection import random_point_select
from superiorflows.partial import PartialFlowUpdater

__all__ = [
    "GaussianPatchBase",
    "UniformPatchBase",
    "PatchUpdater",
    "ExtractAndShiftSubsystem",
    "CouplePatchBase",
    "PatchEquivariantOptimalTransport",
]


class GaussianPatchBase(eqx.Module):
    """Gaussian base distribution for patch sampling, centred on a given point.

    Samples N_patch positions from a multivariate Gaussian centred on
    ``center`` (typically the selection anchor). Positions are drawn
    on R^d **without** wrapping to the torus, avoiding the ill-defined
    heat-kernel density on the flat torus. The ODE flow handles PBC
    internally via ``log_map``/``exp_map``.

    The log-probability uses the minimum-image displacement (``log_map``)
    for the torus distance, which is exact up to exponentially small
    corrections when σ ≪ L.

    Attributes:
        center: Gaussian centre, shape (d,).
        sigma: Standard deviation of the isotropic Gaussian.
        N_patch: Number of patch particles.
        d: Spatial dimension.
        L: Box side length (for torus distance in log_prob).
    """

    center: jnp.ndarray
    sigma: float = eqx.field(static=True)
    N_patch: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    L: float = eqx.field(static=True)

    def sample(self, key, **kwargs):
        """Draw N_patch positions from N(center, σ²I).

        Returns:
            ParticleSystem with ``positions`` of shape ``(N_patch, d)`` and
            ``species=None``, ``box=None`` (matching the dynamic partition).
        """
        noise = jax.random.normal(key, shape=(self.N_patch, self.d))
        positions = self.center + self.sigma * noise
        return ParticleSystem(positions=positions, species=None, box=None)

    def log_prob(self, value, **kwargs):
        """Log-density using minimum-image displacement for torus consistency.

        For σ ≪ L, this is the standard Gaussian log-density evaluated at the
        shortest torus displacement from each particle to the centre:

            log q₀ = −(1/2σ²) Σᵢ ‖log_map(xᵢ, center, L)‖² − (Nd/2) log(2πσ²)

        Args:
            value: ParticleSystem with positions of shape ``(N_patch, d)``.

        Returns:
            Scalar log-probability.
        """
        center_broadcast = jnp.broadcast_to(self.center, value.positions.shape)
        disp = log_map(value.positions, center_broadcast, self.L)
        sq_dist = jnp.sum(disp**2)
        return -0.5 * sq_dist / self.sigma**2 - 0.5 * self.N_patch * self.d * jnp.log(2 * jnp.pi * self.sigma**2)


class UniformPatchBase(eqx.Module):
    """Uniform base distribution for patch sampling.

    Samples N_patch positions uniformly in [0, L)^d. Kept for backward
    compatibility and as a reference baseline; prefer
    :class:`GaussianPatchBase` for better flow initialisation.

    Attributes:
        N_patch: Number of patch particles.
        d: Spatial dimension.
        L: Box side length (assumes cubic box).
    """

    N_patch: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    L: float = eqx.field(static=True)

    def sample(self, key, **kwargs):
        """Draw N_patch uniform positions in [0, L)^d."""
        pos = jax.random.uniform(key, shape=(self.N_patch, self.d), maxval=self.L)
        return ParticleSystem(positions=pos, species=None, box=None)

    def log_prob(self, value, **kwargs):
        """Log-density of uniform distribution on [0, L)^{N_patch × d}."""
        return -self.N_patch * self.d * jnp.log(self.L)


class PatchUpdater(eqx.Module):
    """Orchestrate patch-based local updates for particle systems.

    Workflow::

        full_x → [selection_protocol] → x_active (N_patch + N_context)
               → [PartialFlowUpdater] → x_active' (inner partition: patch vs context)
               → [scatter] → full_x'

    The base distribution is centred on the selection anchor for each
    update, so the flow starts from positions near the patch
    neighbourhood rather than uniformly in the box.

    The inner :class:`~superiorflows.partial.PartialFlowUpdater` handles
    the two-way split of the active sub-system into patch (dynamic) and
    context (static) via the standard ``state_context_partition`` machinery.

    Attributes:
        bijector: The ODE bijector (stored separately so the base can be
            reconstructed per call with a fresh centre).
        selection_fn: Callable ``(key, x) → SelectionResult``.
        N_patch: Static patch allocation size.
        N_context: Static context allocation size.
        sigma: Standard deviation of the Gaussian base distribution.
        L: Box side length.
        d: Spatial dimension.
    """

    bijector: eqx.Module
    selection_fn: object = eqx.field(static=True)
    N_patch: int = eqx.field(static=True)
    N_context: int = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    L: float = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, bijector, selection_fn, N_patch, N_context, *, sigma, L, d):
        self.bijector = bijector
        object.__setattr__(self, "selection_fn", selection_fn)
        self.N_patch = N_patch
        self.N_context = N_context
        self.sigma = sigma
        self.L = L
        self.d = d

    def _make_base(self, center):
        """Construct a Gaussian base distribution centred on ``center``."""
        return GaussianPatchBase(center=center, sigma=self.sigma, N_patch=self.N_patch, d=self.d, L=self.L)

    def _make_updater(self, center):
        """Build a fresh PartialFlowUpdater with a base centred on ``center``."""
        return PartialFlowUpdater(self.bijector, self._make_base(center))

    def update(self, x, key, **kwargs):
        """Propose a new patch.

        Draws a random patch/context partition via the selection protocol,
        constructs a Gaussian base centred on the selection anchor, samples
        from the base, pushes through the flow, and returns the updated full
        configuration.

        Args:
            x: Full particle system configuration.
            key: PRNG key.
            **kwargs: Forwarded to the bijector (e.g. ``key`` for Hutchinson).

        Returns:
            ``(x_new, selection_result)``: updated full config and the selection
            result.
        """
        k_sel, k_flow = jax.random.split(key)
        sel = self.selection_fn(k_sel, x)
        x_active = self._build_subsystem(x, sel)
        mask = self._inner_mask()
        updater = self._make_updater(sel.anchor)

        x_active_new = updater.update(x_active, mask, k_flow, args=(sel.n_patch, sel.n_context), **kwargs)
        x_new = self._scatter_patch(x, x_active_new, sel)

        return x_new, sel

    def update_and_log_prob(self, x, key, **kwargs):
        """Propose a new patch and compute its log-density.

        Draws a random patch/context partition via the selection protocol,
        constructs a Gaussian base centred on the selection anchor, samples
        from the base, pushes through the flow, and returns the updated full
        configuration with the forward log-density.

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
        updater = self._make_updater(sel.anchor)

        x_active_new, log_q = updater.update_and_log_prob(
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
        updater = self._make_updater(sel.anchor)
        return updater.log_prob(x_active, mask, args=(sel.n_patch, sel.n_context), **kwargs)

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
