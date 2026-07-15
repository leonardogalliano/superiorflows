"""Verification of the patch data pipeline transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from particle_systems.particle_system import ParticleSystem
from particle_systems.patch_data import (
    CouplePatchBase,
    ExtractAndShiftSubsystem,
    PatchEquivariantOptimalTransport,
)

N = 44
D = 2
L = 6.075
N_PATCH = 10
N_CONTEXT = 20
R_PATCH = 1.5
R_CONTEXT = 3.0


@pytest.fixture
def config():
    key = jax.random.key(42)
    k1, k2 = jax.random.split(key)
    species = jnp.array([0] * 20 + [1] * 12 + [2] * 12)
    return ParticleSystem(
        positions=jax.random.uniform(k1, (N, D), maxval=L),
        species=species,
        box=jnp.full((D,), L),
    )


def test_extract_and_shift(config):
    transform = ExtractAndShiftSubsystem(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.2, seed=42)

    result = transform(0, config)
    sub_sys, n_patch, n_context = result

    assert isinstance(sub_sys, ParticleSystem)
    assert sub_sys.positions.shape == (N_PATCH + N_CONTEXT, D)
    assert sub_sys.species.shape == (N_PATCH + N_CONTEXT,)
    assert np.allclose(sub_sys.box, config.box)
    assert int(n_patch) + int(n_context) <= N_PATCH + N_CONTEXT

    # Check that all shifted positions are within the box [0, L)^D
    assert jnp.all(sub_sys.positions >= 0.0)
    assert jnp.all(sub_sys.positions < L)


def test_couple_patch_base(config):
    transform = ExtractAndShiftSubsystem(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.2, seed=42)
    sub_sys, n_p, n_c = transform(0, config)

    # CouplePatchBase expects a batched input because it is applied after batching.
    batched_sub_sys = ParticleSystem(
        positions=sub_sys.positions[None, ...],
        species=sub_sys.species[None, ...],
        box=sub_sys.box[None, ...],
    )
    batched_n_p = jnp.array([n_p])
    batched_n_c = jnp.array([n_c])

    coupler = CouplePatchBase(N_PATCH, d=D, L=L, sigma=0.5, seed=42)
    x0, x1, out_np, out_nc = coupler(0, (batched_sub_sys, batched_n_p, batched_n_c))

    # Shapes should preserve batch dimension
    assert x0.positions.shape == (1, N_PATCH + N_CONTEXT, D)
    assert x1.positions.shape == (1, N_PATCH + N_CONTEXT, D)

    # Context particles (from N_PATCH onwards) must be identical in base and target
    assert np.allclose(x0.positions[0, N_PATCH:], x1.positions[0, N_PATCH:])

    # Species and box must be identical
    assert np.allclose(x0.species, x1.species)
    assert np.allclose(x0.box, x1.box)

    # Counts must be passed through
    assert np.allclose(out_np, batched_n_p)
    assert np.allclose(out_nc, batched_n_c)


def test_patch_equivariant_ot(config):
    transform = ExtractAndShiftSubsystem(N_PATCH, N_CONTEXT, R_PATCH, R_CONTEXT, sigma_r=0.2, seed=42)
    sub_sys, n_p, n_c = transform(0, config)

    # Create manual batch of size 2
    batched_sub_sys = ParticleSystem(
        positions=jnp.stack([sub_sys.positions, sub_sys.positions]),
        species=jnp.stack([sub_sys.species, sub_sys.species]),
        box=jnp.stack([sub_sys.box, sub_sys.box]),
    )
    batched_n_p = jnp.array([n_p, n_p])
    batched_n_c = jnp.array([n_c, n_c])

    coupler = CouplePatchBase(N_PATCH, d=D, L=L, sigma=0.5, seed=42)
    x0, x1, np_out, nc_out = coupler(0, (batched_sub_sys, batched_n_p, batched_n_c))

    ot = PatchEquivariantOptimalTransport(N_PATCH)
    x0_aligned, x1_out, np_ot, nc_ot = ot((x0, x1, np_out, nc_out))

    # Check that context particles (indices N_PATCH:) are completely untouched
    assert np.allclose(x0_aligned.positions[:, N_PATCH:], x0.positions[:, N_PATCH:])

    # Check that the patch positions have same shape
    assert x0_aligned.positions.shape == x0.positions.shape

    # Alignment should not alter species or box
    assert np.allclose(x0_aligned.species, x0.species)
    assert np.allclose(x0_aligned.box, x0.box)

    # Counts must be passed through
    assert np.allclose(np_ot, np_out)
    assert np.allclose(nc_ot, nc_out)


def test_base_permutation_invariance():
    from particle_systems.patch_updater import GaussianPatchBase

    key = jax.random.key(123)
    center = jnp.full((D,), L / 2.0)
    base_dist = GaussianPatchBase(
        center=center,
        sigma=0.5,
        N_patch=N_PATCH,
        d=D,
        L=L,
    )
    x = base_dist.sample(key)
    log_p = base_dist.log_prob(x)

    # Permute the particles
    perm = np.random.permutation(N_PATCH)
    x_perm = ParticleSystem(positions=x.positions[perm], species=None, box=None)
    log_p_perm = base_dist.log_prob(x_perm)
    assert np.isclose(log_p, log_p_perm)
