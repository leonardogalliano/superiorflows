"""Tests for NeuralSampler, Flow subclass, and PartialUpdater integration."""

import distreqx.distributions as dsx
import jax
import jax.numpy as jnp

from superiorflows import Flow, NeuralSampler, ODEBijector, PartialUpdater, SDETransport


def test_neural_sampler_with_ode():
    """Verify NeuralSampler composing ODEBijector with Gaussian base."""
    d = 4
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))
    bijector = ODEBijector(
        velocity_field=lambda t, x, a: -0.5 * x,
        dt0=0.1,
    )

    sampler = NeuralSampler(bijector, base_dist)
    assert sampler.event_shape == (d,)

    key = jax.random.key(123)
    y = sampler.sample(key)
    assert y.shape == (d,)

    y, log_w = sampler.sample_and_log_weight(key)
    assert y.shape == (d,)
    assert jnp.isfinite(log_w)

    # For deterministic ODE flow, Flow.log_prob(y) should match log_w
    flow = Flow(bijector, base_dist)
    log_p = flow.log_prob(y)
    assert jnp.allclose(log_w, log_p, atol=1e-4)


def test_neural_sampler_with_sde():
    """Verify NeuralSampler composing SDETransport with Gaussian base."""
    d = 4
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))
    transport = SDETransport(
        velocity_field=lambda t, x, a: -0.5 * x,
        score_field=lambda t, x, a: -x,
        noise_schedule=lambda t: 0.1,
        num_steps=50,
    )

    sampler = NeuralSampler(transport, base_dist)
    key = jax.random.key(456)

    y = sampler.sample(key)
    assert y.shape == (d,)

    y, log_w = sampler.sample_and_log_weight(key)
    assert y.shape == (d,)
    assert jnp.isfinite(log_w)


def test_flow_subclass_aliases():
    """Verify Flow provides sample_and_log_prob and push_forward_and_log_prob aliases."""
    d = 2
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))
    bijector = ODEBijector(
        velocity_field=lambda t, x, a: -x,
        dt0=0.1,
    )
    flow = Flow(bijector, base_dist)

    key = jax.random.key(789)
    y, log_p = flow.sample_and_log_prob(key)
    assert y.shape == (d,)
    assert jnp.isfinite(log_p)

    x0 = jnp.array([1.0, -1.0])
    y_p, log_p_push = flow.push_forward_and_log_prob(x0)
    assert y_p.shape == (d,)
    assert jnp.isfinite(log_p_push)

    # pull_back_and_log_prob
    x_rec, log_p_pull = flow.pull_back_and_log_prob(y_p)
    assert jnp.allclose(x_rec, x0, atol=1e-4)
    assert jnp.allclose(log_p_pull, log_p_push, atol=1e-4)


def test_partial_updater_with_sde():
    """Verify PartialUpdater works with SDETransport for partial/Gibbs sampling."""
    d_dyn = 3
    dyn_indices = jnp.array([0, 2, 4])
    complement = jnp.array([1, 3, 5])

    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d_dyn), jnp.ones(d_dyn))
    transport = SDETransport(
        velocity_field=lambda t, x, a: -0.5 * x,
        score_field=lambda t, x, a: -x,
        noise_schedule=lambda t: 0.1,
        num_steps=30,
    )

    updater = PartialUpdater(transport, base_dist)
    x_old = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    key = jax.random.key(999)

    x_new, log_w = updater.update_and_log_weight(x_old, dyn_indices, key)

    # Context unchanged
    assert jnp.allclose(x_new[complement], x_old[complement])
    # Dynamic DOFs updated
    assert not jnp.allclose(x_new[dyn_indices], x_old[dyn_indices])
    assert jnp.isfinite(log_w)
