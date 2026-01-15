import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from superiorflows import Flow


@pytest.fixture
def uniform_distribution_setup():
    d = (4, 2)
    low = -jnp.ones(d)
    high = jnp.ones(d)
    uniform_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=len(d))
    return uniform_dist


def test_uniform_base(uniform_distribution_setup):
    uniform_dist = uniform_distribution_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    d = uniform_dist.distribution.low.shape
    M = 3
    X0 = uniform_dist.sample(seed=subkey, sample_shape=(M,))
    log_probs = jax.vmap(uniform_dist.log_prob)(X0)
    assert X0.shape == (M,) + d
    assert log_probs.shape == (M,)


class VelocityField(eqx.Module):
    params: jax.Array

    def __call__(self, t, x, args):
        return -t * x @ self.params.T


@pytest.fixture
def velocity_field_setup(uniform_distribution_setup):
    key = jax.random.PRNGKey(0)
    d = uniform_distribution_setup.distribution.low.shape
    key, subkey = jax.random.split(key)
    params = jax.random.normal(subkey, (d[1], d[1]))
    velocity_field = VelocityField(params=params)
    return velocity_field


def test_velocity_field(uniform_distribution_setup, velocity_field_setup):
    uniform_dist = uniform_distribution_setup
    velocity_field = velocity_field_setup
    t = 1.0
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = uniform_dist.sample(seed=subkey)
    v = velocity_field(t, x, None)
    assert v.shape == x.shape


def test_flow_extra_args(uniform_distribution_setup, velocity_field_setup):
    uniform_dist = uniform_distribution_setup
    velocity_field = velocity_field_setup
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=uniform_dist,
        dt0=0.1,
        extra_args={"max_steps": 1000},
    )
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = uniform_dist.sample(seed=subkey)
    x1 = flow.integrate(x, dt0=0.01).ys[-1]
    x2 = flow.apply_map(x)
    x3 = flow.apply_inverse_map(x2)
    assert jnp.allclose(x1, x2, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(x3, x, atol=1e-5, rtol=1e-5)


@pytest.fixture
def flow_setup(uniform_distribution_setup, velocity_field_setup):
    uniform_dist = uniform_distribution_setup
    velocity_field = velocity_field_setup
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=uniform_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
    )
    return flow


def test_flow(flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x0 = flow.base_distribution.sample(seed=subkey)
    x1 = flow.apply_map(x0)
    assert jnp.allclose(flow.apply_inverse_map(x1), x0, atol=1e-5, rtol=1e-5)
    x1, logq1 = flow.apply_map_and_log_prob(x0)
    flow.log_prob(x1)
    assert jnp.allclose(flow.apply_inverse_map(x1), x0, atol=1e-5, rtol=1e-5)


def test_flow_batched(flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    M = 3
    X0 = flow.base_distribution.sample(seed=subkey, sample_shape=(M,))
    X1 = jax.vmap(flow.apply_map)(X0)
    assert X1.shape == (M,) + X0.shape[1:]
    assert jnp.allclose(flow.apply_inverse_map(X1), X0, atol=1e-5, rtol=1e-5)
    X1, logq1 = jax.vmap(flow.apply_map_and_log_prob)(X0)
    assert X1.shape == (M,) + X0.shape[1:]
    assert logq1.shape == (M,)
    assert flow.log_prob(X1).shape == (M,)
    assert jnp.allclose(flow.apply_inverse_map(X1), X0, atol=1e-4, rtol=1e-4)


def test_flow_performance(benchmark, flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x0 = flow.base_distribution.sample(seed=subkey)

    def run_apply_map_and_log_prob():
        x1, logq1 = flow.apply_map_and_log_prob(x0)
        x1.block_until_ready()
        logq1.block_until_ready()
        return x1, logq1

    benchmark(run_apply_map_and_log_prob)


def test_batched_performance(benchmark, flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    M = 128
    X0 = flow.base_distribution.sample(seed=subkey, sample_shape=(M,))

    def run_apply_map_and_log_prob():
        X1, logq1 = jax.vmap(flow.apply_map_and_log_prob)(X0)
        X1.block_until_ready()
        logq1.block_until_ready()
        return X1, logq1

    benchmark(run_apply_map_and_log_prob)


@eqx.filter_jit
def foo_loss(flow, X):
    logq = jax.vmap(flow.log_prob)(X)
    return jnp.mean(logq)


def test_ad_performance(benchmark, flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    M = 128
    X0 = flow.base_distribution.sample(seed=subkey, sample_shape=(M,))
    X = jax.vmap(flow.apply_map)(X0)

    def run_ad():
        jax.grad(foo_loss)(flow, X).velocity_field.params.block_until_ready()

    benchmark(run_ad)
