from typing import Optional

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from superiorflows import Flow


class System(eqx.Module):
    positions: jnp.ndarray
    species: jnp.ndarray
    box: jnp.ndarray
    temperature: Optional[float] = eqx.field(static=True, default=None)

    @property
    def N(self):
        return self.positions.shape[-2]

    @property
    def d(self):
        return self.positions.shape[-1]


class ParticlesVelocityField(eqx.Module):
    params: jax.Array

    def __call__(self, t, state: System, args):
        # ctx = args
        vr = -t * state.positions @ self.params.T
        vs = jnp.zeros_like(state.species)
        return System(
            positions=vr,
            species=vs,
            box=None,
            temperature=state.temperature,
        )


class UniformSystem(eqx.Module, dsx.Distribution):
    box: jnp.ndarray
    ref_species: jnp.ndarray
    temperature: Optional[float] = eqx.field(static=True, default=None)

    def __init__(self, box: jax.Array, ref_species: jax.Array, temperature: float = None):
        self.box = box
        self.ref_species = ref_species
        self.temperature = temperature

    @property
    def event_shape(self):
        return System(
            positions=(self.ref_species.shape[0], self.box.shape[0]),
            species=(self.ref_species.shape[0],),
            box=(self.box.shape[0],),
            temperature=None,
        )

    def _sample_n(self, key, n):
        N = self.ref_species.shape[0]
        d = self.box.shape[0]

        k1, k2 = jax.random.split(key)

        pos = jax.random.uniform(k1, shape=(n, N, d), minval=0.0, maxval=self.box)

        keys_perm = jax.random.split(k2, n)

        def _permute(k):
            return jax.random.permutation(k, self.ref_species)

        species = jax.vmap(_permute)(keys_perm)

        batched_box = jnp.broadcast_to(self.box, (n, d))

        return System(positions=pos, species=species, box=batched_box, temperature=self.temperature)

    def log_prob(self, value: System):
        N = self.ref_species.shape[0]

        vol_log = jnp.sum(jnp.log(self.box))
        base_log_prob = -N * vol_log

        in_box = jnp.all((value.positions >= 0.0) & (value.positions <= self.box), axis=(-1, -2))

        sorted_val_species = jnp.sort(value.species, axis=-1)
        sorted_ref_species = jnp.sort(self.ref_species, axis=-1)

        valid_composition = jnp.all(jnp.isclose(sorted_val_species, sorted_ref_species), axis=-1)

        is_valid = in_box & valid_composition
        return jnp.where(is_valid, base_log_prob, -jnp.inf)


@pytest.fixture
def uniform_box_distribution_setup():
    N = 4
    d = 2
    L = 5.0
    box = jnp.ones(d) * L
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    ref_species = jax.random.uniform(subkey, shape=(N,), minval=0.5, maxval=2.0)
    return UniformSystem(box=box, ref_species=ref_species, temperature=1.0)


def test_uniform_box_distribution(uniform_box_distribution_setup):
    dist = uniform_box_distribution_setup
    N, d = dist.event_shape.positions
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    M = 10
    X = dist.sample(seed=subkey, sample_shape=(M,))
    log_probs = dist.log_prob(X)
    assert X.positions.shape == (M, N, d)
    assert X.species.shape == (M, N)
    assert X.box.shape == (M, d)
    assert log_probs.shape == (M,)


@pytest.fixture
def particles_velocity_field_setup(uniform_box_distribution_setup):
    dist = uniform_box_distribution_setup
    N, d = dist.event_shape.positions
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    params = jax.random.normal(subkey, (d, d))
    velocity_field = ParticlesVelocityField(params=params)
    return velocity_field


def test_particles_velocity_field(uniform_box_distribution_setup, particles_velocity_field_setup):
    velocity_field = particles_velocity_field_setup
    t = 1.0
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = uniform_box_distribution_setup.sample(seed=subkey)

    dynamic_mask = eqx.tree_at(lambda x: (x.positions, x.species, x.box), x, replace=(True, True, False))
    y, ctx = eqx.partition(x, dynamic_mask)

    v = velocity_field(t, y, ctx)
    assert v.positions.shape == x.positions.shape
    assert v.species.shape == x.species.shape
    assert v.box is None
    M = 10
    key, subkey = jax.random.split(key)
    X = uniform_box_distribution_setup.sample(seed=subkey, sample_shape=(M,))

    Y, Ctx = eqx.partition(X, eqx.tree_at(lambda x: (x.positions, x.species, x.box), X, replace=(True, True, False)))

    V = jax.vmap(velocity_field, in_axes=(None, 0, 0))(t, Y, Ctx)
    assert V.positions.shape == X.positions.shape
    assert V.species.shape == X.species.shape
    assert V.box is None


@pytest.fixture
def particles_flow_setup(uniform_box_distribution_setup, particles_velocity_field_setup):
    dist = uniform_box_distribution_setup
    velocity_field = particles_velocity_field_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    dynamic_mask = eqx.tree_at(
        lambda x: (x.positions, x.species, x.box), dist.sample(seed=subkey), replace=(True, True, False)
    )
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=dist,
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
        dynamic_mask=dynamic_mask,
    )
    return flow


def test_particles_flow(particles_flow_setup):
    flow = particles_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x0 = particles_flow_setup.base_distribution.sample(seed=subkey)
    x1 = flow.apply_map(x0)
    assert jax.tree.map(
        lambda x, y: jnp.allclose(x, y, atol=1e-5, rtol=1e-5),
        flow.apply_inverse_map(flow.apply_map(x0=x0)),
        x0,
    )
    x1, logq1 = flow.apply_map_and_log_prob(x0)
    flow.log_prob(x1)
    assert jax.tree.map(
        lambda x, y: jnp.allclose(x, y, atol=1e-5, rtol=1e-5),
        flow.apply_inverse_map(x1),
        x0,
    )


def test_particles_flow_batched(particles_flow_setup):
    flow = particles_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    M = 10
    X0 = flow.base_distribution.sample(seed=subkey, sample_shape=(M,))
    X1 = jax.vmap(flow.apply_map)(X0)
    assert X1.positions.shape == (M,) + X0.positions.shape[1:]
    assert jax.tree.map(
        lambda x, y: jnp.allclose(x, y, atol=1e-5, rtol=1e-5),
        jax.vmap(flow.apply_inverse_map)(X1),
        X0,
    )
    X1, logq1 = jax.vmap(flow.apply_map_and_log_prob)(X0)
    assert X1.positions.shape == (M,) + X0.positions.shape[1:]
    assert logq1.shape == (M,)
    assert flow.log_prob(X1).shape == (M,)
    assert jax.tree.map(
        lambda x, y: jnp.allclose(x, y, atol=1e-5, rtol=1e-5),
        jax.vmap(flow.apply_inverse_map)(X1),
        X0,
    )


def test_array_of_systems(particles_flow_setup):
    flow = particles_flow_setup
    N, d = flow.base_distribution.event_shape.positions
    box = flow.base_distribution.box
    ref_species = flow.base_distribution.ref_species
    key = jax.random.PRNGKey(0)
    key, *subkeys = jax.random.split(key, num=2 * 3 + 1)
    x1 = System(
        positions=jax.random.uniform(subkeys[0], shape=(N, d)),
        species=jax.random.permutation(subkeys[1], ref_species),
        box=box,
        temperature=1.0,
    )
    x2 = System(
        positions=jax.random.uniform(subkeys[2], shape=(N, d)),
        species=jax.random.permutation(subkeys[3], ref_species),
        box=box,
        temperature=1.0,
    )
    x3 = System(
        positions=jax.random.uniform(subkeys[4], shape=(N, d)),
        species=jax.random.permutation(subkeys[5], ref_species),
        box=box,
        temperature=1.0,
    )
    list_of_systems = [x1, x2, x3]
    batched_systems = jax.tree.map(lambda *vals: jnp.array(vals), *list_of_systems)
    assert batched_systems.positions.shape == (3, N, d)
    x1, logq1 = jax.vmap(flow.apply_map_and_log_prob)(batched_systems)
    assert x1.positions.shape == (3, N, d)
    assert logq1.shape == (3,)
    assert jax.tree.map(
        lambda x, y: jnp.allclose(x, y, atol=1e-5, rtol=1e-5),
        jax.vmap(flow.apply_inverse_map)(x1),
        batched_systems,
    )


def test_particle_flow_performance(benchmark, particles_flow_setup):
    flow = particles_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x0 = flow.base_distribution.sample(seed=subkey)

    def run_apply_map_and_log_prob():
        x1, logq1 = flow.apply_map_and_log_prob(x0)
        jax.tree.map(lambda x: x.block_until_ready(), x1)
        logq1.block_until_ready()
        return x1, logq1

    benchmark(run_apply_map_and_log_prob)


def test_particle_flow_batched_performance(benchmark, particles_flow_setup):
    flow = particles_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    M = 128
    X0 = flow.base_distribution.sample(seed=subkey, sample_shape=(M,))

    def run_apply_map_and_log_prob():
        X1, logq1 = jax.vmap(flow.apply_map_and_log_prob)(X0)
        jax.tree.map(lambda x: x.block_until_ready(), X1)
        logq1.block_until_ready()
        return X1, logq1

    benchmark(run_apply_map_and_log_prob)


@eqx.filter_jit
def foo_loss(flow, X):
    logq = flow.log_prob(X)
    return jnp.mean(logq)


def test_particle_ad_performance(benchmark, particles_flow_setup):
    flow = particles_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    M = 128
    X0 = flow.base_distribution.sample(seed=subkey, sample_shape=(M,))
    X = jax.vmap(flow.apply_map)(X0)

    def run_ad():
        jax.grad(foo_loss)(flow, X).velocity_field.params.block_until_ready()

    benchmark(run_ad)


# ============================================================================
# Hutchinson Trace Estimator Tests for Particles
# ============================================================================


@pytest.fixture
def hutchinson_particles_flow_setup(uniform_box_distribution_setup, particles_velocity_field_setup):
    """Particle flow with Hutchinson estimator enabled."""
    dist = uniform_box_distribution_setup
    velocity_field = particles_velocity_field_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    dynamic_mask = eqx.tree_at(
        lambda x: (x.positions, x.species, x.box), dist.sample(seed=subkey), replace=(True, True, False)
    )
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=dist,
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
        dynamic_mask=dynamic_mask,
        hutchinson_samples=5,  # Use 5 random vectors
    )
    return flow


def test_hutchinson_particles_flow(hutchinson_particles_flow_setup, particles_flow_setup):
    """Test Hutchinson estimator on particle systems."""
    hutchinson_flow = hutchinson_particles_flow_setup
    exact_flow = particles_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    x0 = exact_flow.base_distribution.sample(seed=subkey1)

    # Exact result
    x1_exact, logq1_exact = exact_flow.apply_map_and_log_prob(x0)

    # Hutchinson result
    x1_hutch, logq1_hutch = hutchinson_flow.apply_map_and_log_prob(x0, key=subkey2)

    # Positions should match (only divergence differs)
    assert jnp.allclose(x1_exact.positions, x1_hutch.positions, atol=1e-5)
    assert jnp.allclose(x1_exact.species, x1_hutch.species, atol=1e-5)
    # Log prob should be reasonably close
    assert jnp.abs(logq1_exact - logq1_hutch) < 2.0


def test_hutchinson_particles_flow_batched(hutchinson_particles_flow_setup, particles_flow_setup):
    """Test batched Hutchinson estimator on particle systems."""
    hutchinson_flow = hutchinson_particles_flow_setup
    exact_flow = particles_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    M = 10

    X0 = exact_flow.base_distribution.sample(seed=subkey1, sample_shape=(M,))

    # Exact results
    X1_exact, logq1_exact = jax.vmap(exact_flow.apply_map_and_log_prob)(X0)

    # Hutchinson results
    keys = jax.random.split(subkey2, M)
    X1_hutch, logq1_hutch = jax.vmap(lambda x, k: hutchinson_flow.apply_map_and_log_prob(x, key=k))(X0, keys)

    assert X1_hutch.positions.shape == X1_exact.positions.shape
    assert logq1_hutch.shape == (M,)
    assert jnp.allclose(X1_exact.positions, X1_hutch.positions, atol=1e-5)


def test_hutchinson_particles_performance(benchmark, hutchinson_particles_flow_setup):
    """Benchmark Hutchinson estimator on particle systems."""
    flow = hutchinson_particles_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    x0 = flow.base_distribution.sample(seed=subkey1)

    def run_apply_map_and_log_prob():
        x1, logq1 = flow.apply_map_and_log_prob(x0, key=subkey2)
        jax.tree.map(lambda x: x.block_until_ready(), x1)
        logq1.block_until_ready()
        return x1, logq1

    benchmark(run_apply_map_and_log_prob)
