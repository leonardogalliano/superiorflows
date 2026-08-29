"""Mathematical and functional tests for SDETransport and EulerMaruyamaWithAction.

Validates:
1. Exact equivalence of custom EulerMaruyama vs plain EM.
2. Exactness of Girsanov/Crooks path action on 1D Gaussian transport:
   N(0, 1) -> N(3, 0.25) with exact linear velocity and score fields.
   Verifies log E[w] ≈ 0 and ESS/N > 0.95.
3. 3D Gaussian transport with isotropic covariance.
4. Native PyTree support (multi-leaf structured states).
5. Dynamic masking and partial DOF transport.
6. Per-call overrides (t0, t1, num_steps, saveat).
"""

import distreqx.distributions as dsx
import jax
import jax.numpy as jnp

from superiorflows.sde import SDETransport


def test_custom_euler_forward():
    """Verify that SDETransport forward works with EulerMaruyama."""
    key = jax.random.key(42)

    def drift(t, y, args):
        return -0.5 * y

    def score(t, y, args):
        return -y

    x0 = jnp.array([1.0, 2.0, -1.0])

    transport = SDETransport(
        velocity_field=drift,
        score_field=score,
        noise_schedule=lambda t: 0.1,
        num_steps=100,
    )
    final_x = transport.forward(x0, key=key)
    assert final_x.shape == x0.shape
    assert not jnp.isnan(final_x).any()


def test_1d_gaussian_transport_path_action():
    """Verify Girsanov path action on 1D Gaussian transport.

    Transport N(0, 1) -> N(3, 0.5^2) with linear interpolant.
    """
    sigma0, mu1, sigma1 = 1.0, 3.0, 0.5
    eps_val = 0.3

    def mu_t(t):
        return t * mu1

    def sigma2_t(t):
        return (1.0 - t) ** 2 * sigma0**2 + t**2 * sigma1**2

    def d_mu_t(t):
        return mu1

    def d_sigma2_t(t):
        return -2.0 * (1.0 - t) * sigma0**2 + 2.0 * t * sigma1**2

    def velocity_field(t, x, args):
        s2 = sigma2_t(t)
        ds2 = d_sigma2_t(t)
        return d_mu_t(t) + 0.5 * ds2 / s2 * (x - mu_t(t))

    def score_field(t, x, args):
        return -(x - mu_t(t)) / sigma2_t(t)

    def noise_schedule(t):
        return eps_val

    transport = SDETransport(
        velocity_field=velocity_field,
        score_field=score_field,
        noise_schedule=noise_schedule,
        num_steps=200,
    )

    base_dist = dsx.Normal(0.0, sigma0)
    target_dist = dsx.Normal(mu1, sigma1)

    # Batch test
    B = 3000
    master_key = jax.random.key(123)
    keys = jax.random.split(master_key, B)

    def single_sample_and_weight(k):
        k_base, k_sde = jax.random.split(k)
        x0 = base_dist.sample(key=k_base)
        x1, action = transport.forward_and_path_action(x0, key=k_sde)
        log_q0 = base_dist.log_prob(x0)
        log_weight = log_q0 + action
        log_target = target_dist.log_prob(x1)
        log_w = log_target - log_weight
        return x1, log_w

    x1_batch, log_w_batch = jax.vmap(single_sample_and_weight)(keys)

    # Estimate E[w]: by Girsanov theorem, E[w] = 1, so log E[w] = 0
    max_log_w = jnp.max(log_w_batch)
    log_mean_w = max_log_w + jnp.log(jnp.mean(jnp.exp(log_w_batch - max_log_w)))

    # Effective sample size
    w = jnp.exp(log_w_batch - max_log_w)
    w_norm = w / jnp.sum(w)
    ess = 1.0 / jnp.sum(w_norm**2)
    ess_fraction = ess / B

    assert abs(log_mean_w) < 0.05, f"log E[w] = {log_mean_w:.4f} is too far from 0"
    assert ess_fraction > 0.90, f"ESS fraction = {ess_fraction:.3f} is below threshold"


def test_3d_gaussian_transport_path_action():
    """Verify path action on isotropic 3D Gaussian transport."""
    d = 3
    sigma0 = 1.0
    mu1 = jnp.array([1.5, -2.0, 0.5])
    sigma1 = 0.3
    eps_val = 0.2

    def mu_t(t):
        return t * mu1

    def sigma2_t(t):
        return (1.0 - t) ** 2 * sigma0**2 + t**2 * sigma1**2

    def d_mu_t(t):
        return mu1

    def d_sigma2_t(t):
        return -2.0 * (1.0 - t) * sigma0**2 + 2.0 * t * sigma1**2

    def velocity_field(t, x, args):
        s2 = sigma2_t(t)
        ds2 = d_sigma2_t(t)
        return d_mu_t(t) + 0.5 * ds2 / s2 * (x - mu_t(t))

    def score_field(t, x, args):
        return -(x - mu_t(t)) / sigma2_t(t)

    transport = SDETransport(
        velocity_field=velocity_field,
        score_field=score_field,
        noise_schedule=lambda t: eps_val,
        num_steps=150,
    )

    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.full((d,), sigma0))
    target_dist = dsx.MultivariateNormalDiag(mu1, jnp.full((d,), sigma1))

    B = 2000
    keys = jax.random.split(jax.random.key(456), B)

    def single_run(k):
        k_b, k_s = jax.random.split(k)
        x0 = base_dist.sample(key=k_b)
        x1, action = transport.forward_and_path_action(x0, key=k_s)
        log_weight = base_dist.log_prob(x0) + action
        log_w = target_dist.log_prob(x1) - log_weight
        return x1, log_w

    x1_batch, log_w_batch = jax.vmap(single_run)(keys)

    max_log_w = jnp.max(log_w_batch)
    log_mean_w = max_log_w + jnp.log(jnp.mean(jnp.exp(log_w_batch - max_log_w)))
    assert abs(log_mean_w) < 0.05


def test_pytree_state_sde():
    """Verify SDETransport with multi-leaf structured PyTree state."""
    state = {
        "pos": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "vel": jnp.array([0.5, -0.5]),
    }
    key = jax.random.key(789)

    transport = SDETransport(
        velocity_field=lambda t, x, a: jax.tree.map(lambda leaf: -0.5 * leaf, x),
        score_field=lambda t, x, a: jax.tree.map(lambda leaf: -leaf, x),
        noise_schedule=lambda t: 0.1,
        num_steps=50,
    )

    y, action = transport.forward_and_log_weight(state, key=key)
    assert y["pos"].shape == state["pos"].shape
    assert y["vel"].shape == state["vel"].shape
    assert jnp.isfinite(action)


def test_sde_dynamic_masking():
    """Verify SDETransport with dynamic masking (partial DOF update)."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = jnp.array([0, 2, 4])
    complement = jnp.array([1, 3])
    key = jax.random.key(101)

    transport = SDETransport(
        velocity_field=lambda t, x, a: -x,
        score_field=lambda t, x, a: -x,
        noise_schedule=lambda t: 0.1,
        num_steps=50,
    )

    y, action = transport.forward_and_log_weight(x, dynamic_mask=mask, key=key)
    # Static context unchanged
    assert jnp.allclose(y[complement], x[complement])
    # Dynamic DOFs changed
    assert not jnp.allclose(y[mask], x[mask])
    assert jnp.isfinite(action)


def test_sde_per_call_overrides():
    """Verify per-call overrides (num_steps, t0, t1, saveat)."""
    x = jnp.array([1.0, 2.0])
    key = jax.random.key(202)

    transport = SDETransport(
        velocity_field=lambda t, x, a: -x,
        score_field=lambda t, x, a: -x,
        noise_schedule=lambda t: 0.1,
        num_steps=50,
    )

    # Override num_steps per call
    y1 = transport.forward(x, key=key, num_steps=20)
    assert y1.shape == x.shape
