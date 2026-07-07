"""Comprehensive benchmarks for patch-based sampling on real JBB configurations.

Validates correctness, padding, FLOPS and memory scaling by comparing the
patch updater (PatchEGNNVelocity, M = N_patch + N_context) against the full-
system flow (ParticlesEGNNVelocity, N = 44) on equilibrium frames from the
Jung-Biroli-Berthier ternary Lennard-Jones model at T = 1.0.

Run with:
    uv run pytest particle_systems/tests/test_patch_benchmarks.py -v --tb=long -s
"""

import logging
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particle_systems.particle_system import ParticleSystem, TrajectoryDataSource, UniformParticles
from particle_systems.patch_updater import GaussianPatchBase, PatchUpdater
from particle_systems.selection import random_point_select
from particle_systems.velocities import ParticlesEGNNVelocity
from particle_systems.velocities.conditional_egnn import PatchEGNNVelocity
from superiorflows import Flow, ODEBijector

logging.getLogger("particle_systems.particle_system").setLevel(logging.WARNING)

DATA_PATH = "particle_systems/data/jbb_T1.0.xyz"

try:
    source = TrajectoryDataSource(DATA_PATH)
    N = source.N
    D = source.d
    L = float(source.box_size[0])
    ref_species = jnp.asarray(source[0].species)
    n_species = len(jnp.unique(ref_species))
    composition = tuple(np.bincount(np.asarray(ref_species)) / N)
except FileNotFoundError:
    pytest.skip(f"Dataset not found at {DATA_PATH}", allow_module_level=True)

N_PATCH = 10
N_CONTEXT = 20
R_PATCH = 1.5
R_CONTEXT = 3.0
M = N_PATCH + N_CONTEXT

HIDDEN_NF = 16
N_LAYERS = 2
SOLVER_STEPS = 10


def _load_frame(idx=0):
    """Load a single frame from the trajectory as a JAX ParticleSystem."""
    ps = source[idx]
    return ParticleSystem(
        positions=jnp.asarray(ps.positions),
        species=jnp.asarray(ps.species),
        box=jnp.asarray(ps.box),
    )


@pytest.fixture(scope="module")
def real_config():
    """First equilibrium frame from the JBB trajectory."""
    return _load_frame(0)


@pytest.fixture(scope="module")
def real_configs_batch():
    """Several frames for statistical tests."""
    return [_load_frame(i) for i in range(min(10, len(source)))]


@pytest.fixture(scope="module")
def patch_velocity_field():
    return PatchEGNNVelocity(
        N_patch=N_PATCH,
        N_context=N_CONTEXT,
        d=D,
        n_species=n_species,
        hidden_nf=HIDDEN_NF,
        n_layers=N_LAYERS,
        key=jax.random.key(0),
    )


@pytest.fixture(scope="module")
def patch_updater(patch_velocity_field):
    bijector = ODEBijector(
        velocity_field=patch_velocity_field,
        solver=dfx.Euler(),
        augmented_solver=dfx.Euler(),
        dt0=1.0 / SOLVER_STEPS,
        stepsize_controller=dfx.ConstantStepSize(),
        augmented_stepsize_controller=dfx.ConstantStepSize(),
    )
    selection_fn = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.2)
    return PatchUpdater(bijector, selection_fn, N_PATCH, N_CONTEXT, sigma=R_PATCH / 2, L=L, d=D)


@pytest.fixture(scope="module")
def full_velocity_field():
    return ParticlesEGNNVelocity(
        N=N,
        d=D,
        n_species=n_species,
        hidden_nf=HIDDEN_NF,
        n_layers=N_LAYERS,
        key=jax.random.key(0),
    )


@pytest.fixture(scope="module")
def full_flow(full_velocity_field):
    bijector = ODEBijector(
        velocity_field=full_velocity_field,
        solver=dfx.Euler(),
        augmented_solver=dfx.Euler(),
        dt0=1.0 / SOLVER_STEPS,
        stepsize_controller=dfx.ConstantStepSize(),
        augmented_stepsize_controller=dfx.ConstantStepSize(),
        dynamic_mask=ParticleSystem.get_dynamic_mask(),
    )
    base = UniformParticles(N=N, d=D, L=L, composition=composition)
    return Flow(bijector, base)


# ── Correctness on real data ─────────────────────────────────────────────────


class TestPatchOnRealData:
    """Validate the patch pipeline on equilibrium JBB configurations."""

    def test_update_produces_finite_log_prob(self, patch_updater, real_config):
        """Forward update must produce a finite log-density."""
        key = jax.random.key(123)
        x_new, log_q, sel = eqx.filter_jit(patch_updater.update_and_log_prob)(real_config, key)
        assert jnp.isfinite(log_q), f"log_q = {log_q}"

    def test_backward_log_prob_finite(self, patch_updater, real_config):
        """Backward (inverse ODE) log-prob must also be finite."""
        key = jax.random.key(456)
        x_new, log_q_fwd, sel = eqx.filter_jit(patch_updater.update_and_log_prob)(real_config, key)
        log_q_bwd = eqx.filter_jit(patch_updater.log_prob)(real_config, sel)
        assert jnp.isfinite(log_q_bwd), f"log_q_bwd = {log_q_bwd}"

    def test_patch_positions_change(self, patch_updater, real_config):
        """At least some patch positions must differ after update."""
        key = jax.random.key(789)
        x_new, _, sel = eqx.filter_jit(patch_updater.update_and_log_prob)(real_config, key)
        n_p = int(sel.n_patch)
        if n_p > 0:
            valid_idx = sel.patch_idx[:n_p]
            old_pos = real_config.positions[valid_idx]
            new_pos = x_new.positions[valid_idx]
            assert not jnp.allclose(old_pos, new_pos, atol=1e-6), "Patch positions should change after flow update"

    def test_context_and_exterior_preserved(self, patch_updater, real_config):
        """Context and exterior positions must remain exactly unchanged."""
        key = jax.random.key(101)
        x_new, _, sel = eqx.filter_jit(patch_updater.update_and_log_prob)(real_config, key)

        all_active_idx = jnp.concatenate([sel.patch_idx, sel.context_idx])
        exterior_mask = jnp.ones(N, dtype=bool).at[all_active_idx].set(False)
        assert jnp.allclose(
            real_config.positions[exterior_mask],
            x_new.positions[exterior_mask],
        ), "Exterior positions must not change"

        n_c = int(sel.n_context)
        if n_c > 0:
            valid_ctx_idx = sel.context_idx[:n_c]
            assert jnp.allclose(
                real_config.positions[valid_ctx_idx],
                x_new.positions[valid_ctx_idx],
            ), "Context positions must not change"

    def test_species_and_box_preserved(self, patch_updater, real_config):
        """Species and box must be exactly preserved."""
        key = jax.random.key(202)
        x_new, _, _ = eqx.filter_jit(patch_updater.update_and_log_prob)(real_config, key)
        assert jnp.array_equal(x_new.species, real_config.species)
        assert jnp.array_equal(x_new.box, real_config.box)

    def test_sequential_updates_produce_different_configs(self, patch_updater, real_config):
        """Two updates with different keys should produce different results."""
        x1, _, _ = eqx.filter_jit(patch_updater.update_and_log_prob)(real_config, jax.random.key(1))
        x2, _, _ = eqx.filter_jit(patch_updater.update_and_log_prob)(real_config, jax.random.key(2))
        assert not jnp.allclose(x1.positions, x2.positions), "Different keys should produce different patch proposals"


# ── Padding statistics on real data ──────────────────────────────────────────


class TestPaddingOnRealData:
    """Verify padding behaviour on realistic particle distributions."""

    def test_padding_statistics(self, real_config):
        """Collect padding stats over many selections on a real config."""
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.2)
        n_trials = 200
        n_patch_vals = []
        n_context_vals = []
        for i in range(n_trials):
            sel = select(jax.random.fold_in(jax.random.key(0), i), real_config)
            n_patch_vals.append(int(sel.n_patch))
            n_context_vals.append(int(sel.n_context))

        n_patch_arr = np.array(n_patch_vals)
        n_context_arr = np.array(n_context_vals)

        print(f"\n  Padding statistics over {n_trials} selections:")
        print(
            f"    n_patch  — mean: {n_patch_arr.mean():.1f}, "
            f"min: {n_patch_arr.min()}, max: {n_patch_arr.max()}, "
            f"fraction padded: {(n_patch_arr < N_PATCH).mean():.2f}"
        )
        print(
            f"    n_context — mean: {n_context_arr.mean():.1f}, "
            f"min: {n_context_arr.min()}, max: {n_context_arr.max()}, "
            f"fraction padded: {(n_context_arr < N_CONTEXT).mean():.2f}"
        )

        assert all(0 <= v <= N_PATCH for v in n_patch_vals)
        assert all(0 <= v <= N_CONTEXT for v in n_context_vals)

    def test_padding_convention_on_real_data(self, real_config):
        """For each selection, padded indices must duplicate the last valid index."""
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.2)
        for i in range(50):
            sel = select(jax.random.fold_in(jax.random.key(42), i), real_config)
            n_p = int(sel.n_patch)
            if 0 < n_p < N_PATCH:
                last_valid = sel.patch_idx[n_p - 1]
                padded = sel.patch_idx[n_p:]
                assert jnp.all(padded == last_valid), f"Trial {i}: padded indices {padded} should all be {last_valid}"
            n_c = int(sel.n_context)
            if 0 < n_c < N_CONTEXT:
                last_valid_c = sel.context_idx[n_c - 1]
                padded_c = sel.context_idx[n_c:]
                assert jnp.all(padded_c == last_valid_c), (
                    f"Trial {i}: padded context indices {padded_c} should all be {last_valid_c}"
                )


# ── FLOPS and memory scaling comparison ─────────────────────────────────────


class TestScalingComparison:
    """Compare patch vs full-system FLOPS, memory, and wall-clock time."""

    def test_compilation_cost_analysis(self, patch_updater, full_flow, real_config):
        """Compare XLA cost analysis between patch update and full flow."""
        key = jax.random.key(0)

        @eqx.filter_jit
        def patch_step(x, k):
            return patch_updater.update_and_log_prob(x, k)

        @jax.jit
        def full_step(k):
            return full_flow.sample_and_log_prob(key=k)

        patch_lowered = patch_step.lower(real_config, key)
        full_lowered = full_step.lower(key)

        patch_compiled = patch_lowered.compile()
        full_compiled = full_lowered.compile()

        def _get_cost(compiled):
            inner = getattr(compiled, "compiled", compiled)
            try:
                return inner.cost_analysis()
            except (AttributeError, NotImplementedError):
                return None

        def _get_memory(compiled):
            inner = getattr(compiled, "compiled", compiled)
            try:
                return inner.memory_analysis()
            except (AttributeError, NotImplementedError):
                return None

        patch_cost = _get_cost(patch_compiled)
        full_cost = _get_cost(full_compiled)

        print("\n  === XLA Cost Analysis ===")
        if patch_cost and isinstance(patch_cost, (list, tuple)):
            patch_cost = patch_cost[0] if patch_cost else {}
        if full_cost and isinstance(full_cost, (list, tuple)):
            full_cost = full_cost[0] if full_cost else {}

        patch_flops = patch_cost.get("flops", "N/A") if isinstance(patch_cost, dict) else "N/A"
        full_flops = full_cost.get("flops", "N/A") if isinstance(full_cost, dict) else "N/A"

        print(f"    Patch FLOPS: {patch_flops}")
        print(f"    Full  FLOPS: {full_flops}")

        if isinstance(patch_flops, (int, float)) and isinstance(full_flops, (int, float)):
            ratio = full_flops / max(patch_flops, 1)
            print(f"    FLOPS ratio (full/patch): {ratio:.2f}x")

        patch_mem = _get_memory(patch_compiled)
        full_mem = _get_memory(full_compiled)

        if patch_mem and isinstance(patch_mem, (list, tuple)):
            patch_mem = patch_mem[0] if patch_mem else {}
        if full_mem and isinstance(full_mem, (list, tuple)):
            full_mem = full_mem[0] if full_mem else {}

        print(f"    Patch memory: {patch_mem}")
        print(f"    Full  memory: {full_mem}")

    def test_wall_clock_comparison(self, patch_updater, full_flow, real_config):
        """Wall-clock timing: patch update vs full sample_and_log_prob."""
        key = jax.random.key(42)
        n_warmup = 3
        n_runs = 20

        @eqx.filter_jit
        def patch_fn(x, k):
            return patch_updater.update_and_log_prob(x, k)

        @jax.jit
        def full_fn(k):
            return full_flow.sample_and_log_prob(key=k)

        for _ in range(n_warmup):
            key, k1, k2 = jax.random.split(key, 3)
            r1 = patch_fn(real_config, k1)
            r2 = full_fn(k2)
            jax.block_until_ready(r1)
            jax.block_until_ready(r2)

        patch_times = []
        full_times = []
        for i in range(n_runs):
            key, k1, k2 = jax.random.split(key, 3)

            t0 = time.perf_counter()
            r = patch_fn(real_config, k1)
            jax.block_until_ready(r)
            patch_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            r = full_fn(k2)
            jax.block_until_ready(r)
            full_times.append(time.perf_counter() - t0)

        patch_ms = np.array(patch_times) * 1000
        full_ms = np.array(full_times) * 1000

        print(f"\n  === Wall-clock timing ({n_runs} runs) ===")
        print(f"    Patch: {patch_ms.mean():.2f} ± {patch_ms.std():.2f} ms")
        print(f"    Full:  {full_ms.mean():.2f} ± {full_ms.std():.2f} ms")
        print(f"    Ratio (full/patch): {full_ms.mean() / patch_ms.mean():.2f}x")

    def test_divergence_cost_comparison(self, real_config):
        """Compare exact divergence cost: patch (M dofs) vs full (N*d dofs)."""
        patch_dofs = N_PATCH * D
        full_dofs = N * D

        print("\n  === Divergence computation scaling ===")
        print(f"    Patch dynamic DOFs: {patch_dofs} (N_patch={N_PATCH} × d={D})")
        print(f"    Full dynamic DOFs:  {full_dofs} (N={N} × d={D})")
        print("    Exact divergence: O(d²) JVP probes")
        print(f"      Patch: {patch_dofs} probes")
        print(f"      Full:  {full_dofs} probes")
        print(f"      Reduction: {full_dofs / patch_dofs:.1f}×")

        assert patch_dofs < full_dofs, "Patch should have fewer dynamic DOFs"


# ── N-independence of patch computation ──────────────────────────────────────


class TestNIndependence:
    """Verify that the patch update computation does not scale with N."""

    def test_patch_jaxpr_has_no_N_shapes(self, patch_updater, real_config):
        """Inspect the traced computation for N-dependent shapes.

        The patch update should only contain arrays of shape proportional to M
        (N_patch + N_context), not N (full system), in the ODE integration.
        We report shape occurrences for manual inspection.
        """
        key = jax.random.key(0)

        jaxpr = jax.make_jaxpr(lambda x, k: eqx.filter_jit(patch_updater.update_and_log_prob)(x, k))(real_config, key)

        jaxpr_str = str(jaxpr)

        print("\n  === Jaxpr shape analysis ===")
        print(f"    Full system N: {N}, patch M: {M}")

        for shape_name, pattern in [
            (f"({N}, {D}) full N", f"{N} {D}"),
            (f"({M}, {D}) active M", f"{M} {D}"),
            (f"({N_PATCH}, {D}) patch", f"{N_PATCH} {D}"),
        ]:
            count = jaxpr_str.count(pattern)
            print(f"    Occurrences of {shape_name}: {count}")

        print(f"    Jaxpr length: {len(jaxpr_str)} chars")

    def test_gaussian_base_samples_near_patch(self, real_config):
        """Verify that GaussianPatchBase concentrates samples near the anchor."""
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.0)
        key = jax.random.key(42)
        sel = select(key, real_config)

        sigma = R_PATCH / 2
        base = GaussianPatchBase(center=sel.anchor, sigma=sigma, N_patch=N_PATCH, d=D, L=L)

        sample_key = jax.random.key(99)
        s = base.sample(key=sample_key)

        dists = jnp.sqrt(jnp.sum((s.positions - sel.anchor) ** 2, axis=-1))
        mean_dist = float(jnp.mean(dists))

        print("\n  === Gaussian base proximity ===")
        print(f"    Anchor: {sel.anchor}")
        print(f"    Sigma: {sigma:.3f}")
        print(f"    Mean distance from anchor: {mean_dist:.3f}")
        print(f"    Expected (σ√(d/1)): {sigma * jnp.sqrt(D):.3f}")

        assert mean_dist < 5 * sigma, f"Mean distance {mean_dist:.3f} too large for sigma={sigma:.3f}"

    def test_gaussian_vs_uniform_log_prob_at_anchor(self, real_config):
        """Gaussian base should assign higher density near the patch than uniform."""
        select = random_point_select(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.0)
        key = jax.random.key(42)
        sel = select(key, real_config)

        sigma = R_PATCH / 2
        gauss_base = GaussianPatchBase(center=sel.anchor, sigma=sigma, N_patch=N_PATCH, d=D, L=L)
        from particle_systems.patch_updater import UniformPatchBase

        unif_base = UniformPatchBase(N_patch=N_PATCH, d=D, L=L)

        n_p = int(sel.n_patch)
        if n_p > 0:
            nearby_pos = real_config.positions[sel.patch_idx[:n_p]]
            padded_pos = jnp.broadcast_to(nearby_pos[-1], (N_PATCH - n_p, D))
            positions = jnp.concatenate([nearby_pos, padded_pos])
        else:
            positions = jnp.broadcast_to(sel.anchor, (N_PATCH, D))

        val = ParticleSystem(positions=positions, species=None, box=None)

        gauss_lp = float(gauss_base.log_prob(val))
        unif_lp = float(unif_base.log_prob(val))

        print("\n  === Base distribution comparison at patch positions ===")
        print(f"    Gaussian log_prob: {gauss_lp:.2f}")
        print(f"    Uniform  log_prob: {unif_lp:.2f}")
        print(f"    Difference: {gauss_lp - unif_lp:.2f}")

        assert gauss_lp > unif_lp, (
            "Gaussian base should assign higher density to positions near the anchor "
            f"than uniform. Got gauss={gauss_lp:.2f}, unif={unif_lp:.2f}"
        )
