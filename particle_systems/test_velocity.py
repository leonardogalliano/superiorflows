"""Tests for ParticlesMLPVelocity and Flow integration with particle systems."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from superiorflows import Flow

from particle_systems.particle_system import ParticleSystem, TrajectoryDataSource, UniformParticles
from particle_systems.velocities import ParticlesMLPVelocity

DATA_PATH = "/Users/Leonardo/Documents/PhD/Projects/ParticlesMC/data/datasets" "/SS142D/T0.1/N10/M100/steps10000000"

N, d = 10, 2


@pytest.fixture(scope="module")
def velocity_field():
    key = jax.random.key(0)
    return ParticlesMLPVelocity(N=N, d=d, n_species=2, width=32, depth=2, key=key)


@pytest.fixture(scope="module")
def trajectory_batch():
    source = TrajectoryDataSource(DATA_PATH)
    frames = [source[i] for i in range(8)]
    return ParticleSystem(
        positions=jnp.stack([jnp.array(f.positions) for f in frames]),
        species=jnp.stack([jnp.array(f.species) for f in frames]),
        box=jnp.stack([jnp.array(f.box) for f in frames]),
    )


@pytest.fixture(scope="module")
def base_distribution(trajectory_batch):
    L = float(trajectory_batch.box[0, 0])
    return UniformParticles(N=N, d=d, L=L, composition=(0.5, 0.5))


# ── Velocity field ────────────────────────────────────────────────────────────


def test_velocity_field_output_shape(velocity_field, trajectory_batch):
    """Single-frame call should return ParticleSystem with positions of shape (N, d)."""
    frame = ParticleSystem(
        positions=trajectory_batch.positions[0],
        species=trajectory_batch.species[0],
        box=trajectory_batch.box[0],
    )
    x_dyn, ctx_stat = eqx.partition(frame, ParticleSystem.get_dynamic_mask())
    v = velocity_field(0.5, x_dyn, ctx_stat)
    assert v.positions.shape == (N, d)
    assert jnp.all(jnp.isfinite(v.positions))


def test_velocity_field_filter_jit(velocity_field, trajectory_batch):
    """Velocity field should be filter_jit-compilable."""
    frame = ParticleSystem(
        positions=trajectory_batch.positions[0],
        species=trajectory_batch.species[0],
        box=trajectory_batch.box[0],
    )
    x_dyn, ctx_stat = eqx.partition(frame, ParticleSystem.get_dynamic_mask())

    v1 = velocity_field(0.5, x_dyn, ctx_stat)
    v2 = eqx.filter_jit(velocity_field)(0.5, x_dyn, ctx_stat)
    assert jnp.allclose(v1.positions, v2.positions, atol=1e-5)


# ── Flow integration ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def flow(velocity_field, base_distribution):
    import diffrax as dfx

    return Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        dynamic_mask=ParticleSystem.get_dynamic_mask(),
        stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3),
        augmented_stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3),
    )


def test_flow_log_prob_single(flow, trajectory_batch):
    """log_prob on a single frame should return a finite scalar."""
    frame = ParticleSystem(
        positions=trajectory_batch.positions[0],
        species=trajectory_batch.species[0],
        box=trajectory_batch.box[0],
    )
    lp = flow.log_prob(frame)
    assert lp.shape == ()
    assert jnp.isfinite(lp)


def test_flow_log_prob_batched(flow, trajectory_batch):
    """log_prob on a batch of frames should return finite values for each."""
    lp = flow.log_prob(trajectory_batch)
    assert lp.shape == (8,)
    assert jnp.all(jnp.isfinite(lp))
