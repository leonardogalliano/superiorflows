import diffrax as dfx
import distrax as dsx
import jax
import jax.numpy as jnp
import pytest
from superiorflows import Flow


@pytest.fixture
def uniform_distribution_setup():
    d = (1,)
    low = -jnp.ones(d)
    high = jnp.ones(d)
    uniform_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=len(d))
    return uniform_dist


def velocity_field(t, x, args):
    return t * x**2


def true_solution(t, x0):
    return x0 / (1 - 0.5 * t**2 * x0)


def true_density_solution(t, f0, x0):
    return f0 + 2 * jnp.log(jnp.abs(1 - 0.5 * t**2 * x0))


@pytest.fixture
def flow_setup(uniform_distribution_setup):
    uniform_dist = uniform_distribution_setup
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
    M = 10
    X0 = flow.base_distribution.sample(seed=subkey, sample_shape=(M,))
    flow_result = jax.vmap(flow.apply_map)(X0)
    true_result = jax.vmap(true_solution, in_axes=(None, 0))(1.0, X0)
    assert jnp.allclose(flow_result, true_result)
    inverse_flow_result = jax.vmap(flow.apply_inverse_map)(flow_result)
    assert jnp.allclose(inverse_flow_result, X0)
    augmented_flow_result, logq = jax.vmap(flow.apply_map_and_log_prob)(X0)
    assert jnp.allclose(augmented_flow_result, true_result)
    log_prob_flow_result = jax.vmap(flow.log_prob)(flow_result)
    log_prob_augmented_result = jax.vmap(flow.log_prob)(augmented_flow_result)
    assert jnp.allclose(logq, log_prob_flow_result, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(logq, log_prob_augmented_result, atol=1e-5, rtol=1e-5)
    true_log_prob = jax.vmap(true_density_solution, in_axes=(None, 0, 0))(1.0, flow.base_distribution.log_prob(X0), X0)
    assert jnp.allclose(logq, true_log_prob.reshape(-1), atol=1e-5, rtol=1e-5)
    assert jnp.allclose(log_prob_flow_result, true_log_prob.reshape(-1), atol=1e-5, rtol=1e-5)
    assert jnp.allclose(log_prob_augmented_result, true_log_prob.reshape(-1), atol=1e-4, rtol=1e-4)


def analytical_divergence_fn(velocity_field_fn, t, x, args):
    """Analytical divergence for v(t, x) = t * x².

    dv/dx = 2 * t * x, summed over all dimensions.
    """
    v = velocity_field_fn(t, x, args)
    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    div_v = jnp.sum(2 * t * x_flat)
    return v, div_v


@pytest.fixture
def analytical_flow_setup(uniform_distribution_setup):
    uniform_dist = uniform_distribution_setup
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=uniform_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
        divergence_fn=analytical_divergence_fn,
    )
    return flow


def test_flow_analytical(analytical_flow_setup):
    """Test that analytical divergence matches the true closed-form density."""
    flow = analytical_flow_setup
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    M = 10
    X0 = flow.base_distribution.sample(seed=subkey, sample_shape=(M,))

    flow_result = jax.vmap(flow.apply_map)(X0)
    true_result = jax.vmap(true_solution, in_axes=(None, 0))(1.0, X0)
    assert jnp.allclose(flow_result, true_result)

    augmented_flow_result, logq = jax.vmap(flow.apply_map_and_log_prob)(X0)
    assert jnp.allclose(augmented_flow_result, true_result)

    true_log_prob = jax.vmap(true_density_solution, in_axes=(None, 0, 0))(1.0, flow.base_distribution.log_prob(X0), X0)
    assert jnp.allclose(logq, true_log_prob.reshape(-1), atol=1e-5, rtol=1e-5)

    log_prob_flow_result = jax.vmap(flow.log_prob)(augmented_flow_result)
    assert jnp.allclose(log_prob_flow_result, true_log_prob.reshape(-1), atol=1e-4, rtol=1e-4)
