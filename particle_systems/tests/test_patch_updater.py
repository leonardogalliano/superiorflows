"""End-to-end tests for patch-based sampling infrastructure.

Tests the full pipeline: selection → sub-system extraction → conditional EGNN
→ PartialFlowUpdater → scatter, verifying correctness of shapes, preservation
invariants, masking, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particle_systems.particle_system import ParticleSystem
from particle_systems.patch_updater import (
    CouplePatchBase,
    ExtractAndShiftSubsystem,
    GaussianPatchBase,
    PatchEquivariantOptimalTransport,
    PatchUpdater,
    UniformPatchBase,
)
from particle_systems.selection import random_point_select
from particle_systems.velocities.conditional_egnn import PatchEGNNVelocity
from superiorflows.ode import ODEBijector

# ── Fixtures ─────────────────────────────────────────────────────────────────

N = 44
D = 2
L = 6.075
N_PATCH = 10
N_CONTEXT = 20
R_PATCH = 1.5
R_CONTEXT = 3.0
N_SPECIES = 3
HIDDEN_NF = 16
N_LAYERS = 2


@pytest.fixture
def key():
    return jax.random.key(42)


@pytest.fixture
def config(key):
    """JBB particle system configuration loaded from data."""
    import os

    import jax.numpy as jnp

    from particle_systems.particle_system import ParticleSystem, TrajectoryDataSource

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "jbb_T1.0.xyz")
    source = TrajectoryDataSource(data_path)

    # Pick a specific frame to be deterministic, e.g. index 0
    frame = source[0]
    return ParticleSystem(
        positions=jnp.array(frame.positions), species=jnp.array(frame.species), box=jnp.array(frame.box)
    )


@pytest.fixture
def velocity_field(key):
    """PatchEGNNVelocity with random parameters."""
    return PatchEGNNVelocity(
        N_patch=N_PATCH,
        N_context=N_CONTEXT,
        d=D,
        n_species=N_SPECIES,
        hidden_nf=HIDDEN_NF,
        n_layers=N_LAYERS,
        key=key,
    )


@pytest.fixture
def patch_updater(velocity_field):
    """PatchUpdater wired to a random EGNN with Gaussian base."""
    import diffrax as dfx

    bijector = ODEBijector(
        velocity_field=velocity_field,
        solver=dfx.Euler(),
        augmented_solver=dfx.Euler(),
        dt0=0.1,
        stepsize_controller=dfx.ConstantStepSize(),
        augmented_stepsize_controller=dfx.ConstantStepSize(),
    )
    selection_fn = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.2)

    return PatchUpdater(
        bijector,
        selection_fn,
        N_PATCH,
        N_CONTEXT,
        sigma=R_PATCH / 2,
        L=L,
        d=D,
    )


# ── Selection protocol tests ────────────────────────────────────────────────


class TestSelectionProtocol:
    def test_output_shapes(self, key, config):
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT)
        sel = select(key, config)

        assert sel.anchor.shape == (D,)
        assert sel.radius.shape == ()
        assert sel.patch_idx.shape == (N_PATCH,)
        assert sel.context_idx.shape == (N_CONTEXT,)
        assert sel.n_patch.shape == ()
        assert sel.n_context.shape == ()

    def test_valid_indices_are_sorted(self, key, config):
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT)
        sel = select(key, config)

        n_p = int(sel.n_patch)
        if n_p > 1:
            valid = sel.patch_idx[:n_p]
            assert jnp.all(valid[:-1] <= valid[1:]), "Valid patch indices must be sorted"

    def test_padding_duplicates_last_valid(self, key, config):
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT)
        sel = select(key, config)

        n_p = int(sel.n_patch)
        if n_p > 0 and n_p < N_PATCH:
            last_valid = sel.patch_idx[n_p - 1]
            padded = sel.patch_idx[n_p:]
            assert jnp.all(padded == last_valid), "Padded slots must duplicate the last valid index"

    def test_determinism(self, config):
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT)
        sel1 = select(jax.random.key(99), config)
        sel2 = select(jax.random.key(99), config)
        assert jnp.array_equal(sel1.patch_idx, sel2.patch_idx)
        assert jnp.array_equal(sel1.context_idx, sel2.context_idx)

    def test_vmap_compatibility(self, config):
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT)
        keys = jax.random.split(jax.random.key(0), 16)
        vmapped = jax.vmap(select, in_axes=(0, None))
        results = vmapped(keys, config)
        assert results.patch_idx.shape == (16, N_PATCH)
        assert results.n_patch.shape == (16,)

    def test_soft_log_pi_ratio_full_patch(self):
        from particle_systems.selection import soft_log_pi_ratio

        # Test full patch (capacity reached)
        patch_idx = jnp.array([0, 1, 2])
        context_idx = jnp.array([3, 4, 5, 6, 7])
        n_patch = jnp.array(3)
        n_context = jnp.array(5)

        # Distances before: patch is [0.1, 0.2, 0.3], context is [0.4, 0.5, 0.6, 0.7, 0.8]
        dists_before = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # Distances after: patch moved to [0.15, 0.25, 0.35], which is still < context
        dists_after_valid = jnp.array([0.15, 0.25, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # Distances after: patch moved to [0.5, 0.6, 0.7], which overlaps with context!
        dists_after_invalid = jnp.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        r_mean = 0.35
        sigma_r = 0.1

        log_pi_valid = soft_log_pi_ratio(
            dists_before, dists_after_valid, patch_idx, context_idx, n_patch, n_context, r_mean, sigma_r
        )
        # Should be finite and > -inf
        assert jnp.isfinite(log_pi_valid)

        log_pi_invalid = soft_log_pi_ratio(
            dists_before, dists_after_invalid, patch_idx, context_idx, n_patch, n_context, r_mean, sigma_r
        )
        # Should be exactly -inf because order is violated
        assert log_pi_invalid == -jnp.inf


# ── PatchEGNNVelocity tests ─────────────────────────────────────────────────


class TestPatchEGNNVelocity:
    def test_masking_isolates_padded_particles(self, velocity_field, config, key):
        """Padded particles must not affect valid particle velocities.

        Runs the same valid particles with different padded positions and
        verifies that valid velocities are identical.
        """
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT)
        sel = select(key, config)

        active_idx = jnp.concatenate([sel.patch_idx, sel.context_idx])
        M = N_PATCH + N_CONTEXT

        pos1 = config.positions[active_idx]
        species = config.species[active_idx]
        box = config.box

        (k2,) = jax.random.split(key, 1)
        pos2 = pos1.at[int(sel.n_patch + sel.n_context) :].set(
            jax.random.uniform(k2, (M - int(sel.n_patch + sel.n_context), D), maxval=L)
        )

        x_active1 = ParticleSystem(positions=pos1, species=species, box=box)
        x_active2 = ParticleSystem(positions=pos2, species=species, box=box)

        inner_mask = ParticleSystem(positions=jnp.arange(N_PATCH), species=False, box=False)

        from superiorflows.bijector import _pack_args
        from superiorflows.partial import state_context_partition

        x_dyn1, ctx1, spec1 = state_context_partition(x_active1, inner_mask)
        args1 = _pack_args(ctx1, spec1.index_meta, (sel.n_patch, sel.n_context))
        v1 = velocity_field(0.5, x_dyn1, args1)

        x_dyn2, ctx2, spec2 = state_context_partition(x_active2, inner_mask)
        args2 = _pack_args(ctx2, spec2.index_meta, (sel.n_patch, sel.n_context))
        v2 = velocity_field(0.5, x_dyn2, args2)

        n_p = int(sel.n_patch)
        if n_p > 0:
            assert jnp.allclose(v1.positions[:n_p], v2.positions[:n_p], atol=1e-5), (
                "Valid patch velocities must be identical regardless of padded positions"
            )


# ── PatchUpdater end-to-end tests ────────────────────────────────────────────


class TestPatchUpdater:
    def test_update_preserves_shapes(self, patch_updater, config, key):
        x_new, log_q, sel = patch_updater.update_and_log_prob(config, key)

        assert x_new.positions.shape == config.positions.shape
        assert x_new.species.shape == config.species.shape
        assert x_new.box.shape == config.box.shape

    def test_species_preserved(self, patch_updater, config, key):
        x_new, _, _ = patch_updater.update_and_log_prob(config, key)
        assert jnp.array_equal(x_new.species, config.species)

    def test_box_preserved(self, patch_updater, config, key):
        x_new, _, _ = patch_updater.update_and_log_prob(config, key)
        assert jnp.array_equal(x_new.box, config.box)

    def test_exterior_positions_unchanged(self, patch_updater, config, key):
        """Positions outside the patch+context selection must not change."""
        x_new, _, sel = patch_updater.update_and_log_prob(config, key)

        n_p = int(sel.n_patch)
        n_c = int(sel.n_context)
        is_patch = jnp.zeros(N, dtype=bool).at[sel.patch_idx[:n_p]].set(True)
        is_context = jnp.zeros(N, dtype=bool).at[sel.context_idx[:n_c]].set(True)
        is_exterior = ~(is_patch | is_context)

        exterior_orig = config.positions[is_exterior]
        exterior_new = x_new.positions[is_exterior]
        assert jnp.allclose(exterior_orig, exterior_new, atol=1e-6), "Exterior positions must be unchanged"

    def test_context_positions_unchanged(self, patch_updater, config, key):
        """Context particle positions must not change."""
        x_new, _, sel = patch_updater.update_and_log_prob(config, key)

        n_c = int(sel.n_context)
        if n_c > 0:
            valid_ctx = sel.context_idx[:n_c]
            ctx_orig = config.positions[valid_ctx]
            ctx_new = x_new.positions[valid_ctx]
            assert jnp.allclose(ctx_orig, ctx_new, atol=1e-6), "Context positions must be unchanged"

    def test_log_density_finite(self, patch_updater, config, key):
        _, log_q_fwd, sel = patch_updater.update_and_log_prob(config, key)
        assert jnp.isfinite(log_q_fwd), f"Forward log-density must be finite, got {log_q_fwd}"

        log_q_bwd = patch_updater.log_prob(config, sel)
        assert jnp.isfinite(log_q_bwd), f"Backward log-density must be finite, got {log_q_bwd}"

    def test_jit_compatibility(self, patch_updater, config, key):
        """The full propose method must compile and run under eqx.filter_jit."""
        import equinox as eqx

        @eqx.filter_jit
        def step(updater, x, k):
            return updater.update_and_log_prob(x, k)

        x_new, log_q, sel = step(patch_updater, config, key)
        assert x_new.positions.shape == config.positions.shape
        assert jnp.isfinite(log_q)

    def test_log_prob_jit_compatibility(self, patch_updater, config, key):
        """log_prob must also compile under eqx.filter_jit."""
        import equinox as eqx

        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.2)
        sel = select(key, config)

        @eqx.filter_jit
        def eval_lp(updater, x, s):
            return updater.log_prob(x, s)

        log_q = eval_lp(patch_updater, config, sel)
        assert jnp.isfinite(log_q)


# ── UniformPatchBase tests ───────────────────────────────────────────────────


class TestGaussianPatchBase:
    def test_sample_shape(self, key):
        center = jnp.array([3.0, 3.0])
        base = GaussianPatchBase(center=center, sigma=R_PATCH / 2, N_patch=N_PATCH, d=D, L=L)
        s = base.sample(key=key)
        assert s.positions.shape == (N_PATCH, D)
        assert s.species is None
        assert s.box is None

    def test_samples_near_center(self, key):
        """Samples should cluster around the centre, not be uniformly spread."""
        center = jnp.array([3.0, 3.0])
        sigma = 0.5
        base = GaussianPatchBase(center=center, sigma=sigma, N_patch=N_PATCH, d=D, L=L)
        s = base.sample(key=key)
        dists = jnp.sqrt(jnp.sum((s.positions - center) ** 2, axis=-1))
        assert jnp.mean(dists) < 3 * sigma, "Samples should be close to the centre"

    def test_log_prob_value(self):
        center = jnp.array([3.0, 3.0])
        sigma = 0.5
        base = GaussianPatchBase(center=center, sigma=sigma, N_patch=N_PATCH, d=D, L=L)
        dummy = ParticleSystem(positions=jnp.broadcast_to(center, (N_PATCH, D)), species=None, box=None)
        expected = -0.5 * N_PATCH * D * jnp.log(2 * jnp.pi * sigma**2)
        assert jnp.allclose(base.log_prob(dummy), expected, atol=1e-5)

    def test_log_prob_uses_torus_distance(self):
        """log_prob should use minimum-image displacement, not Euclidean."""
        center = jnp.array([0.1, 0.1])
        sigma = 0.5
        base = GaussianPatchBase(center=center, sigma=sigma, N_patch=1, d=D, L=L)
        near = ParticleSystem(positions=jnp.array([[L - 0.1, 0.1]]), species=None, box=None)
        far = ParticleSystem(positions=jnp.array([[L / 2, L / 2]]), species=None, box=None)
        assert base.log_prob(near) > base.log_prob(far), "Position near the centre via PBC should have higher log-prob"


class TestUniformPatchBase:
    def test_sample_shape(self, key):
        base = UniformPatchBase(N_patch=N_PATCH, d=D, L=L)
        s = base.sample(key=key)
        assert s.positions.shape == (N_PATCH, D)
        assert s.species is None
        assert s.box is None

    def test_log_prob_value(self):
        base = UniformPatchBase(N_patch=N_PATCH, d=D, L=L)
        dummy = ParticleSystem(positions=jnp.zeros((N_PATCH, D)), species=None, box=None)
        expected = -N_PATCH * D * jnp.log(L)
        assert jnp.allclose(base.log_prob(dummy), expected)

    def test_positions_in_box(self, key):
        base = UniformPatchBase(N_patch=N_PATCH, d=D, L=L)
        s = base.sample(key=key)
        assert jnp.all(s.positions >= 0)
        assert jnp.all(s.positions < L)


# ── Scatter correctness tests ────────────────────────────────────────────────


class TestScatterCorrectness:
    def test_padded_indices_do_not_overwrite_valid_updates(self, key, config):
        """Regression test: padded duplicate indices must not overwrite the
        last valid particle's updated position with its original value.
        """
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT)

        found = False
        for i in range(100):
            k = jax.random.fold_in(key, i)
            sel = select(k, config)
            n_p = int(sel.n_patch)
            if 0 < n_p < N_PATCH:
                found = True
                break

        if not found:
            pytest.skip("Could not find a selection with padding in 100 trials")

        last_valid_original_idx = sel.patch_idx[n_p - 1]
        orig_pos = config.positions[last_valid_original_idx]

        perturbed_pos = config.positions.at[last_valid_original_idx].set(orig_pos + 0.1)
        config_perturbed = ParticleSystem(positions=perturbed_pos, species=config.species, box=config.box)

        from particle_systems.patch_updater import PatchUpdater

        new_patch_pos = config_perturbed.positions[sel.patch_idx]

        M = N_PATCH + N_CONTEXT
        active_pos = jnp.zeros((M, D))
        active_pos = active_pos.at[:N_PATCH].set(new_patch_pos)

        x_active_new = ParticleSystem(
            positions=active_pos,
            species=config.species[jnp.concatenate([sel.patch_idx, sel.context_idx])],
            box=config.box,
        )

        updater_dummy = object.__new__(PatchUpdater)
        object.__setattr__(updater_dummy, "N_patch", N_PATCH)
        result = updater_dummy._scatter_patch(config, x_active_new, sel)

        scattered = result.positions[last_valid_original_idx]
        expected = config_perturbed.positions[last_valid_original_idx]
        assert jnp.allclose(scattered, expected, atol=1e-6), (
            f"Last valid particle must get its updated position, not the original. Got {scattered}, expected {expected}"
        )


# ── Distance check tests ────────────────────────────────────────────────────


class TestDistanceCheck:
    def test_check_patch_remainder_separation(self, config, key):
        from particle_systems.selection import check_patch_remainder_separation, random_point_select

        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT)
        sel = select(key, config)

        n_p = int(sel.n_patch)
        n_c = int(sel.n_context)

        # We need a selection that actually has some remainder particles
        assert n_p + n_c < N, "Test requires some remainder particles"

        # Identify remainder particles
        valid_patch_mask = jnp.arange(N_PATCH) < n_p
        valid_ctx_mask = jnp.arange(N_CONTEXT) < n_c

        is_patch = jnp.any((jnp.arange(N)[:, None] == sel.patch_idx[None, :]) & valid_patch_mask[None, :], axis=1)
        is_context = jnp.any((jnp.arange(N)[:, None] == sel.context_idx[None, :]) & valid_ctx_mask[None, :], axis=1)
        is_remainder = ~(is_patch | is_context)

        r_cut = 2.5
        L_half = config.box[0] / 2.0
        assert L_half * 1.414 > r_cut, "Box too small for test"

        # Override all positions to create a controlled environment
        # Place all particles at the origin initially
        new_positions = jnp.zeros_like(config.positions)

        # Place all remainder particles at L/2, L/2
        safe_remainder_pos = jnp.array([L_half, L_half])
        new_positions = jnp.where(is_remainder[:, None], safe_remainder_pos, new_positions)

        # Place patch particles at origin (distance to remainder is L_half*sqrt(2) > r_cut)
        import equinox as eqx

        config = eqx.tree_at(lambda c: c.positions, config, new_positions)

        # Case 1: Valid move. The new patch stays at origin.
        pos_patch_new_safe = jnp.zeros((N_PATCH, D))

        is_safe = check_patch_remainder_separation(
            config.positions, pos_patch_new_safe, sel.patch_idx, sel.context_idx, n_p, n_c, config.box, r_cut
        )
        assert bool(is_safe) is True

        # Case 2: Invalid move. One patch particle moves to be exactly r_cut - 0.1 away from the remainder cluster.
        # Remainder is at [L_half, L_half]. Move patch[0] to [L_half - (r_cut - 0.1), L_half].
        invalid_pos_new = safe_remainder_pos - jnp.array([r_cut - 0.1, 0.0])
        pos_patch_new_invalid = pos_patch_new_safe.at[0].set(invalid_pos_new)

        is_safe_invalid = check_patch_remainder_separation(
            config.positions, pos_patch_new_invalid, sel.patch_idx, sel.context_idx, n_p, n_c, config.box, r_cut
        )
        assert bool(is_safe_invalid) is False


class TestPatchDataTransforms:
    @pytest.fixture
    def patch_data_config(self):
        key = jax.random.key(42)
        k1, k2 = jax.random.split(key)
        species = jnp.array([0] * 20 + [1] * 12 + [2] * 12)
        return ParticleSystem(
            positions=jax.random.uniform(k1, (44, 2), maxval=6.075),
            species=species,
            box=jnp.full((2,), 6.075),
        )

    def test_extract_and_shift(self, patch_data_config):
        transform = ExtractAndShiftSubsystem(10, 20, 1.5, 3.0, sigma_r=0.2, seed=42)

        result = transform(0, patch_data_config)
        sub_sys, n_patch, n_context = result

        assert isinstance(sub_sys, ParticleSystem)
        assert sub_sys.positions.shape == (30, 2)
        assert sub_sys.species.shape == (30,)
        assert np.allclose(sub_sys.box, patch_data_config.box)
        assert int(n_patch) + int(n_context) <= 30

        # Check that all shifted positions are within the box [0, L)^D
        assert jnp.all(sub_sys.positions >= 0.0)
        assert jnp.all(sub_sys.positions < 6.075)

    def test_couple_patch_base(self, patch_data_config):
        transform = ExtractAndShiftSubsystem(10, 20, 1.5, 3.0, sigma_r=0.2, seed=42)
        sub_sys, n_p, n_c = transform(0, patch_data_config)

        # CouplePatchBase expects a batched input because it is applied after batching.
        batched_sub_sys = ParticleSystem(
            positions=sub_sys.positions[None, ...],
            species=sub_sys.species[None, ...],
            box=sub_sys.box[None, ...],
        )
        batched_n_p = jnp.array([n_p])
        batched_n_c = jnp.array([n_c])

        coupler = CouplePatchBase(10, d=2, L=6.075, sigma=0.5, seed=42)
        x0, x1, out_np, out_nc = coupler(0, (batched_sub_sys, batched_n_p, batched_n_c))

        # Shapes should preserve batch dimension
        assert x0.positions.shape == (1, 30, 2)
        assert x1.positions.shape == (1, 30, 2)

        # Context particles (from N_PATCH onwards) must be identical in base and target
        assert np.allclose(x0.positions[0, 10:], x1.positions[0, 10:])

        # Species and box must be identical
        assert np.allclose(x0.species, x1.species)
        assert np.allclose(x0.box, x1.box)

        # Counts must be passed through
        assert np.allclose(out_np, batched_n_p)
        assert np.allclose(out_nc, batched_n_c)

    def test_patch_equivariant_ot(self, patch_data_config):
        transform = ExtractAndShiftSubsystem(10, 20, 1.5, 3.0, sigma_r=0.2, seed=42)
        sub_sys, n_p, n_c = transform(0, patch_data_config)

        # Create manual batch of size 2
        batched_sub_sys = ParticleSystem(
            positions=jnp.stack([sub_sys.positions, sub_sys.positions]),
            species=jnp.stack([sub_sys.species, sub_sys.species]),
            box=jnp.stack([sub_sys.box, sub_sys.box]),
        )
        batched_n_p = jnp.array([n_p, n_p])
        batched_n_c = jnp.array([n_c, n_c])

        coupler = CouplePatchBase(10, d=2, L=6.075, sigma=0.5, seed=42)
        x0, x1, np_out, nc_out = coupler(0, (batched_sub_sys, batched_n_p, batched_n_c))

        ot = PatchEquivariantOptimalTransport(10)
        x0_aligned, x1_out, np_ot, nc_ot = ot((x0, x1, np_out, nc_out))

        # Check that context particles (indices N_PATCH:) are completely untouched
        assert np.allclose(x0_aligned.positions[:, 10:], x0.positions[:, 10:])

        # Check that the patch positions have same shape
        assert x0_aligned.positions.shape == x0.positions.shape

        # Alignment should not alter species or box
        assert np.allclose(x0_aligned.species, x0.species)
        assert np.allclose(x0_aligned.box, x0.box)

        # Counts must be passed through
        assert np.allclose(np_ot, np_out)
        assert np.allclose(nc_ot, nc_out)

    def test_base_permutation_invariance(self):
        center = jnp.full((2,), 6.075 / 2.0)
        base_dist = GaussianPatchBase(
            center=center,
            sigma=0.5,
            N_patch=10,
            d=2,
            L=6.075,
        )
        x = base_dist.sample(jax.random.key(123))
        log_p = base_dist.log_prob(x)

        # Permute the particles
        import numpy as np

        perm = np.random.permutation(10)
        x_perm = ParticleSystem(positions=x.positions[perm], species=None, box=None)
        log_p_perm = base_dist.log_prob(x_perm)
        assert np.isclose(log_p, log_p_perm)


class TestPatchCallbacks:
    def test_patch_acceptance_callback(self, patch_updater, config):
        from particle_systems.callbacks_particles import PatchAcceptanceCallback

        def dummy_energy(pos, spec, n_valid):
            return jnp.sum(pos**2)

        batched_config = ParticleSystem(
            positions=config.positions[None, ...],
            species=config.species[None, ...],
            box=config.box[None, ...],
        )

        cb = PatchAcceptanceCallback(
            val_data=[batched_config],
            make_patch_updater=lambda m: patch_updater,
            masked_energy_fn=dummy_energy,
            temperature=1.0,
            r_mean=R_PATCH,
            sigma_r=0.2,
            r_cut=2.5,
            r_context=R_CONTEXT,
            eval_freq=1,
        )

        class MockTrainer:
            model = None
            key = jax.random.key(123)

        trainer = MockTrainer()
        cb.on_train_start(trainer)

        logs = {}
        cb.on_step_end(trainer, step=1, logs=logs)

        assert "acc_mean" in logs
        assert "log_pi_ratio_mean" in logs
        assert "log_q_ratio_mean" in logs
        assert "dU_T_mean" in logs
        assert np.isfinite(logs["acc_mean"])

    def test_patch_local_energy_callback(self, patch_updater, config):
        from particle_systems.callbacks_particles import PatchLocalEnergyCallback

        def dummy_local_energy(pos, spec, n_patch, n_valid):
            mask = jnp.arange(pos.shape[0]) < n_patch
            masked_pos = jnp.where(mask[:, None], pos, 0.0)
            return jnp.sum(masked_pos**2)

        batched_config = ParticleSystem(
            positions=config.positions[None, ...],
            species=config.species[None, ...],
            box=config.box[None, ...],
        )

        cb = PatchLocalEnergyCallback(
            val_data=[batched_config],
            make_patch_updater=lambda m: patch_updater,
            masked_local_energy_fn=dummy_local_energy,
            eval_freq=1,
        )

        class MockTrainer:
            model = None
            key = jax.random.key(123)

        trainer = MockTrainer()
        cb.on_train_start(trainer)

        assert cb._target_u_loc is not None
        assert np.isfinite(cb._target_u_loc_mean)

        logs = {}
        cb.on_step_end(trainer, step=1, logs=logs)

        assert "u_loc_mean" in logs
        assert "u_loc_median" in logs
        assert np.isfinite(logs["u_loc_mean"])
