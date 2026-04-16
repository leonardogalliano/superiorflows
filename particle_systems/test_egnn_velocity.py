"""Tests for ParticlesEGNNVelocity and its integration with the Flow.

Verifies that:
  1. The EGNN velocity field has the correct output signature.
  2. It is filter_jit-compilable.
  3. A Flow built with the EGNN can run sample_and_log_prob end-to-end.

To run:
    cd /Users/Leonardo/Documents/Postdoc/Projects/superiorflows
    uv run pytest particle_systems/test_egnn_velocity.py -v
"""

import logging

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from superiorflows import Flow

from particle_systems.particle_system import (
    ParticleSystem,
    TrajectoryDataSource,
    UniformParticles,
)
from particle_systems.velocities import ParticlesEGNNVelocity

logging.getLogger("particle_systems.particle_system").setLevel(logging.WARNING)

# ── Dataset ───────────────────────────────────────────────────────────────────

TRJ_PATH = "particle_systems/data/ss14_T0.1.xyz"
POTENTIAL_FILE = "particle_systems/models/soft_spheres_14.json"

try:
    target_source = TrajectoryDataSource(TRJ_PATH)
    N = target_source.N
    d = target_source.d
    L = float(target_source.metadata["cell"][0])
    temperature = float(target_source.metadata["T"])
    ref_species = np.asarray(target_source[0].species)
    composition = tuple(np.bincount(ref_species) / len(ref_species))
    n_species = len(np.unique(ref_species))
except FileNotFoundError:
    pytest.skip(
        f"Dataset not found at {TRJ_PATH}, skipping EGNN tests.",
        allow_module_level=True,
    )

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def base_dist():
    return UniformParticles(N=N, d=d, L=L, composition=composition)


@pytest.fixture(scope="module")
def egnn_velocity():
    """Small EGNN (2 layers, 32 hidden) for fast functional tests."""
    key = jax.random.key(0)
    return ParticlesEGNNVelocity(
        N=N,
        d=d,
        n_species=n_species,
        hidden_nf=32,
        n_layers=2,
        key=key,
    )


@pytest.fixture(scope="module")
def single_frame():
    """A single ParticleSystem frame from the dataset."""
    frame = target_source[0]
    return ParticleSystem(
        positions=jnp.array(frame.positions),
        species=jnp.array(frame.species),
        box=jnp.array(frame.box),
    )


@pytest.fixture(scope="module")
def flow(egnn_velocity, base_dist):
    return Flow(
        velocity_field=egnn_velocity,
        base_distribution=base_dist,
        dynamic_mask=ParticleSystem.get_dynamic_mask(),
        solver=dfx.Tsit5(),
        augmented_solver=dfx.Tsit5(),
        stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3),
        augmented_stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3),
    )


# ── Velocity tests ────────────────────────────────────────────────────────────


def test_egnn_velocity_output_shape(egnn_velocity, single_frame):
    """Single-frame call returns ParticleSystem with positions (N, d)."""
    x_dyn, ctx = eqx.partition(single_frame, ParticleSystem.get_dynamic_mask())
    v = egnn_velocity(0.5, x_dyn, ctx)
    assert v.positions.shape == (N, d), f"Expected ({N}, {d}), got {v.positions.shape}"
    assert jnp.all(jnp.isfinite(v.positions)), "Velocity contains non-finite values"


def test_egnn_velocity_filter_jit(egnn_velocity, single_frame):
    """Velocity field must be filter_jit-compilable and consistent."""
    x_dyn, ctx = eqx.partition(single_frame, ParticleSystem.get_dynamic_mask())
    v1 = egnn_velocity(0.5, x_dyn, ctx)
    v2 = eqx.filter_jit(egnn_velocity)(0.5, x_dyn, ctx)
    assert jnp.allclose(v1.positions, v2.positions, atol=1e-5), "JIT and eager velocity outputs differ"


def test_egnn_velocity_t0_t1(egnn_velocity, single_frame):
    """Velocity at t=0 and t=1 should both be finite."""
    x_dyn, ctx = eqx.partition(single_frame, ParticleSystem.get_dynamic_mask())
    for t in [0.0, 1.0]:
        v = egnn_velocity(t, x_dyn, ctx)
        assert jnp.all(jnp.isfinite(v.positions)), f"Non-finite velocity at t={t}"


# ── Flow integration tests ────────────────────────────────────────────────────


def test_flow_sample(flow, base_dist):
    """Flow.sample should return a ParticleSystem with the right shape and finite values."""
    key = jax.random.key(42)
    samples = flow.sample(seed=key, sample_shape=(4,))
    assert samples.positions.shape == (4, N, d)
    assert jnp.all(jnp.isfinite(samples.positions))


def test_flow_sample_and_log_prob(flow, base_dist):
    """Flow.sample_and_log_prob should return finite samples and finite log-probs."""
    key = jax.random.key(7)
    samples, lps = flow.sample_and_log_prob(seed=key, sample_shape=(4,))
    assert samples.positions.shape == (4, N, d)
    assert lps.shape == (4,)
    assert jnp.all(jnp.isfinite(samples.positions)), "Non-finite sample positions"
    assert jnp.all(jnp.isfinite(lps)), "Non-finite log-probs"


def test_flow_log_prob_single(flow, single_frame):
    """log_prob of a single frame should be a finite scalar."""
    lp = flow.log_prob(single_frame)
    assert lp.shape == (), f"Expected scalar, got shape {lp.shape}"
    assert jnp.isfinite(lp), f"log_prob is not finite: {lp}"


def test_flow_log_prob_batched(flow):
    """log_prob of a batch should return a finite vector."""
    frames = [target_source[i] for i in range(4)]
    batch = ParticleSystem(
        positions=jnp.stack([jnp.array(f.positions) for f in frames]),
        species=jnp.stack([jnp.array(f.species) for f in frames]),
        box=jnp.stack([jnp.array(f.box) for f in frames]),
    )
    lps = flow.log_prob(batch)
    assert lps.shape == (4,)
    assert jnp.all(jnp.isfinite(lps)), f"Non-finite log-probs: {lps}"


# ── Parameter count ───────────────────────────────────────────────────────────


def test_egnn_parameter_count(egnn_velocity):
    """Sanity-check that the model has a non-trivial number of parameters."""
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(egnn_velocity, eqx.is_inexact_array)))
    print(f"\nEGNN parameter count (n_layers=2, hidden_nf=32): {num_params:,}")
    assert num_params > 0
