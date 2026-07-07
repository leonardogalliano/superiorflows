"""End-to-end tests for patch-based sampling infrastructure.

Tests the full pipeline: selection → sub-system extraction → conditional EGNN
→ PartialFlowUpdater → scatter, verifying correctness of shapes, preservation
invariants, masking, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import pytest

from particle_systems.particle_system import ParticleSystem
from particle_systems.patch_updater import GaussianPatchBase, PatchUpdater, UniformPatchBase
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
    """Random JBB-like particle system configuration."""
    k1, k2 = jax.random.split(key)
    species = jnp.array([0] * 20 + [1] * 12 + [2] * 12)
    return ParticleSystem(
        positions=jax.random.uniform(k1, (N, D), maxval=L),
        species=species,
        box=jnp.full((D,), L),
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
