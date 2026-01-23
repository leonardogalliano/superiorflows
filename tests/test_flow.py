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
    logq = flow.log_prob(X)
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


def test_flow_distrax_interface(flow_setup):
    flow = flow_setup
    assert isinstance(flow, dsx.Distribution)
    assert flow.event_shape == flow.base_distribution.event_shape


def test_flow_sample_n(flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)
    samples = flow.sample(seed=key, sample_shape=(10,))
    assert samples.shape == (10,) + flow.event_shape

    samples_and_log_prob, log_prob = flow.sample_and_log_prob(seed=key, sample_shape=(10,))
    assert samples_and_log_prob.shape == (10,) + flow.event_shape
    assert log_prob.shape == (10,)
    assert jnp.allclose(samples, samples_and_log_prob, atol=1e-4)


def test_flow_log_prob_batched(flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)
    samples = flow.sample(seed=key, sample_shape=(10,))
    log_probs = flow.log_prob(samples)
    assert log_probs.shape == (10,)

    def single_log_prob(x):
        return flow.log_prob(x)

    log_probs_vmap = jax.vmap(single_log_prob)(samples)
    assert jnp.allclose(log_probs, log_probs_vmap, atol=1e-5)


def test_sample_n_performance(benchmark, flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)

    def run_sample_n():
        samples = flow.sample(seed=key, sample_shape=(128,))
        samples.block_until_ready()
        return samples

    benchmark(run_sample_n)


def test_sample_n_and_log_prob_performance(benchmark, flow_setup):
    flow = flow_setup
    key = jax.random.PRNGKey(0)

    def run_sample_n_and_log_prob():
        samples, log_prob = flow.sample_and_log_prob(seed=key, sample_shape=(128,))
        samples.block_until_ready()
        log_prob.block_until_ready()
        return samples, log_prob

    benchmark(run_sample_n_and_log_prob)


# ============================================================================
# Hutchinson Trace Estimator Tests
# ============================================================================


@pytest.fixture
def hutchinson_flow_setup(uniform_distribution_setup, velocity_field_setup):
    """Flow with Hutchinson estimator enabled."""
    uniform_dist = uniform_distribution_setup
    velocity_field = velocity_field_setup
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=uniform_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
        hutchinson_samples=10,  # Use 10 random vectors
    )
    return flow


def test_hutchinson_flow(hutchinson_flow_setup, flow_setup):
    """Test that Hutchinson estimator gives close results to exact computation."""
    hutchinson_flow = hutchinson_flow_setup
    exact_flow = flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    x0 = exact_flow.base_distribution.sample(seed=subkey1)

    # Exact result
    x1_exact, logq1_exact = exact_flow.apply_map_and_log_prob(x0)

    # Hutchinson result
    x1_hutch, logq1_hutch = hutchinson_flow.apply_map_and_log_prob(x0, key=subkey2)

    # Trajectories should be identical (only divergence differs)
    assert jnp.allclose(x1_exact, x1_hutch, atol=1e-5)
    # Log prob should be close (stochastic, so larger tolerance)
    assert jnp.abs(logq1_exact - logq1_hutch) < 1.0  # Reasonable for 10 samples


def test_hutchinson_flow_batched(hutchinson_flow_setup, flow_setup):
    """Test Hutchinson estimator with batched inputs."""
    hutchinson_flow = hutchinson_flow_setup
    exact_flow = flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    M = 10

    X0 = exact_flow.base_distribution.sample(seed=subkey1, sample_shape=(M,))

    # Exact results
    X1_exact, logq1_exact = jax.vmap(exact_flow.apply_map_and_log_prob)(X0)

    # Hutchinson results (need to split keys for each sample)
    keys = jax.random.split(subkey2, M)
    X1_hutch, logq1_hutch = jax.vmap(lambda x, k: hutchinson_flow.apply_map_and_log_prob(x, key=k))(X0, keys)

    assert X1_hutch.shape == X1_exact.shape
    assert logq1_hutch.shape == (M,)
    assert jnp.allclose(X1_exact, X1_hutch, atol=1e-5)


def test_hutchinson_log_prob(hutchinson_flow_setup, flow_setup):
    """Test Hutchinson log_prob method."""
    hutchinson_flow = hutchinson_flow_setup
    exact_flow = flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    x0 = exact_flow.base_distribution.sample(seed=subkey1)
    x1 = exact_flow.apply_map(x0)

    # Exact log prob
    log_prob_exact = exact_flow.log_prob(x1)

    # Hutchinson log prob
    log_prob_hutch = hutchinson_flow.log_prob(x1, key=subkey2)

    # Should be reasonably close
    assert jnp.abs(log_prob_exact - log_prob_hutch) < 1.0


def test_hutchinson_distrax_interface(hutchinson_flow_setup):
    """Test that Hutchinson flow works with distrax sampling interface."""
    flow = hutchinson_flow_setup
    key = jax.random.PRNGKey(0)

    # sample_and_log_prob should work with Hutchinson
    samples, log_probs = flow.sample_and_log_prob(seed=key, sample_shape=(10,))
    assert samples.shape == (10,) + flow.event_shape
    assert log_probs.shape == (10,)


def test_hutchinson_performance(benchmark, hutchinson_flow_setup):
    """Benchmark Hutchinson estimator."""
    flow = hutchinson_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    x0 = flow.base_distribution.sample(seed=subkey1)

    def run_apply_map_and_log_prob():
        x1, logq1 = flow.apply_map_and_log_prob(x0, key=subkey2)
        x1.block_until_ready()
        logq1.block_until_ready()
        return x1, logq1

    benchmark(run_apply_map_and_log_prob)
