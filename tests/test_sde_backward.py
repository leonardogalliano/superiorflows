"""Mathematical and functional tests for backward SDE transport.

Validates the backward SDE integration and path action by testing against
known analytical Gaussian transport solutions in both 1D and 3D, verifying:

1. **Correctness of backward transport**: samples from N(μ₁, σ₁²) at t=1 are
   transported to N(0, σ₀²) at t=0, verified by comparing empirical mean/std
   against the analytical target.

2. **Exactness of backward Girsanov action**: the importance weight
   w = q₁(y)·exp(S^bwd) / q₀(x) satisfies E[w] = 1 by the Girsanov theorem,
   verified via log E[w] ≈ 0 and ESS/N ≈ 1.

3. **Forward–backward action antisymmetry**: E[S^fwd] + E[S^bwd] ≈ 0 on the
   same Gaussian interpolant, verifying S^bwd = −S^fwd.

4. **Non-regression of forward solvers**: existing forward transport must be
   unaffected by the bidirectional solver fixes.

5. **Backward with partial DOFs**: dynamic masking preserves static context.

6. **Backward with structured PyTree states**: multi-leaf backward integration.

7. **Bijector backward == inverse**: AbstractBijector bridges backward to inverse.

8. **PartialUpdater backward methods**: backward_update and backward_update_and_log_weight.
"""

import distreqx.distributions as dsx
import jax
import jax.numpy as jnp
import pytest

from superiorflows.sde import SDETransport

# ──────────────────────────────────────────────────────────────
# Shared Gaussian interpolant fixtures
# ──────────────────────────────────────────────────────────────


def _make_gaussian_transport(sigma0, mu1, sigma1, eps_val, num_steps=200):
    """Build an SDETransport with exact linear Gaussian interpolant.

    The interpolating distribution is:
        q_t = N(μ_t, σ²_t)   with   μ_t = t·μ₁,   σ²_t = (1−t)²σ₀² + t²σ₁²

    Velocity and score are analytically exact, so the SDE is
    consistent with the marginals q₀ = N(0, σ₀²) and q₁ = N(μ₁, σ₁²).
    """

    def mu_t(t):
        return t * mu1

    def sigma2_t(t):
        return (1.0 - t) ** 2 * sigma0**2 + t**2 * sigma1**2

    def velocity_field(t, x, args):
        s2 = sigma2_t(t)
        ds2 = -2.0 * (1.0 - t) * sigma0**2 + 2.0 * t * sigma1**2
        return mu1 + 0.5 * ds2 / s2 * (x - mu_t(t))

    def score_field(t, x, args):
        return -(x - mu_t(t)) / sigma2_t(t)

    transport = SDETransport(
        velocity_field=velocity_field,
        score_field=score_field,
        noise_schedule=lambda t: eps_val,
        num_steps=num_steps,
    )
    return transport


def _log_mean_w_and_ess(log_w_batch):
    """Numerically stable computation of log E[w] and ESS/N."""
    max_lw = jnp.max(log_w_batch)
    log_mean_w = max_lw + jnp.log(jnp.mean(jnp.exp(log_w_batch - max_lw)))
    w = jnp.exp(log_w_batch - max_lw)
    w_norm = w / jnp.sum(w)
    ess_fraction = 1.0 / jnp.sum(w_norm**2) / log_w_batch.shape[0]
    return log_mean_w, ess_fraction


# ══════════════════════════════════════════════════════════════
# 1D Gaussian backward transport
# ══════════════════════════════════════════════════════════════


class TestBackwardSDE1DGaussian:
    """1D Gaussian N(0,1) ↔ N(3, 0.25) with ε = 0.3."""

    sigma0, mu1, sigma1, eps_val = 1.0, 3.0, 0.5, 0.3
    B = 3000

    @pytest.fixture
    def transport(self):
        return _make_gaussian_transport(self.sigma0, self.mu1, self.sigma1, self.eps_val, num_steps=200)

    @pytest.fixture
    def base_dist(self):
        return dsx.Normal(0.0, self.sigma0)

    @pytest.fixture
    def target_dist(self):
        return dsx.Normal(self.mu1, self.sigma1)

    def test_backward_basic_shape_and_finiteness(self, transport, target_dist):
        """Backward output has correct shape with finite values."""
        key = jax.random.key(100)
        k_t, k_s = jax.random.split(key)
        y = target_dist.sample(key=k_t)
        x = transport.backward(y, key=k_s)
        assert x.shape == y.shape
        assert jnp.isfinite(x).all()

    def test_backward_samples_match_base_distribution(self, transport, target_dist, base_dist):
        """Backward-transported target samples recover the base distribution.

        If the SDE is correctly reversed, samples from q₁ = N(3, 0.25)
        should map to q₀ = N(0, 1).  We verify empirical mean and std
        against the analytical targets.
        """
        keys = jax.random.split(jax.random.key(200), self.B)

        def sample_backward(k):
            k_t, k_s = jax.random.split(k)
            y = target_dist.sample(key=k_t)
            return transport.backward(y, key=k_s)

        x0_batch = jax.vmap(sample_backward)(keys)

        mean_x0 = jnp.mean(x0_batch)
        std_x0 = jnp.std(x0_batch)
        assert abs(mean_x0) < 0.15, f"Mean = {mean_x0:.4f}, expected ≈ 0"
        assert abs(std_x0 - self.sigma0) < 0.15, f"Std = {std_x0:.4f}, expected ≈ {self.sigma0}"

    def test_backward_path_action_importance_weights(self, transport, target_dist, base_dist):
        """Backward Girsanov action yields unbiased importance weights.

        The backward importance weight is:
            log w = log q₀(x₀) − [log q₁(y) + S^bwd]

        By the Girsanov theorem, E[w] = 1 ⟹ log E[w] = 0.
        """
        keys = jax.random.split(jax.random.key(300), self.B)

        def single_backward(k):
            k_t, k_s = jax.random.split(k)
            y = target_dist.sample(key=k_t)
            x0, action_bwd = transport.backward_and_path_action(y, key=k_s)
            log_weight_bwd = target_dist.log_prob(y) + action_bwd
            log_w = base_dist.log_prob(x0) - log_weight_bwd
            return x0, log_w, action_bwd

        x0_batch, log_w_batch, actions = jax.vmap(single_backward)(keys)

        log_mean_w, ess_frac = _log_mean_w_and_ess(log_w_batch)
        assert abs(log_mean_w) < 0.05, f"log E[w] = {log_mean_w:.4f}, expected ≈ 0"
        assert ess_frac > 0.85, f"ESS/N = {ess_frac:.3f}, expected > 0.85"

    def test_forward_backward_action_antisymmetry(self, transport, base_dist, target_dist):
        """Forward and backward path actions are antisymmetric in expectation.

        For the exact interpolant, E[S^fwd] + E[S^bwd] ≈ 0 because the
        forward and backward SDEs generate identical marginals with
        complementary action contributions.
        """
        keys = jax.random.split(jax.random.key(400), self.B)

        def forward_action(k):
            k_b, k_s = jax.random.split(k)
            x0 = base_dist.sample(key=k_b)
            _, action = transport.forward_and_path_action(x0, key=k_s)
            return action

        def backward_action(k):
            k_t, k_s = jax.random.split(k)
            y = target_dist.sample(key=k_t)
            _, action = transport.backward_and_path_action(y, key=k_s)
            return action

        fwd_actions = jax.vmap(forward_action)(keys)
        bwd_actions = jax.vmap(backward_action)(keys)

        mean_sum = jnp.mean(fwd_actions) + jnp.mean(bwd_actions)
        assert abs(mean_sum) < 0.15, f"E[S_fwd] + E[S_bwd] = {mean_sum:.4f}, expected ≈ 0"

    def test_forward_not_broken_by_solver_fix(self, transport, base_dist, target_dist):
        """Forward transport still produces correct importance weights.

        Regression test: the bidirectional solver fix must not alter
        forward behaviour.
        """
        keys = jax.random.split(jax.random.key(500), self.B)

        def single_fwd(k):
            k_b, k_s = jax.random.split(k)
            x0 = base_dist.sample(key=k_b)
            x1, action = transport.forward_and_path_action(x0, key=k_s)
            log_q0 = base_dist.log_prob(x0)
            log_w = target_dist.log_prob(x1) - (log_q0 + action)
            return x1, log_w

        x1_batch, log_w_batch = jax.vmap(single_fwd)(keys)

        log_mean_w, ess_frac = _log_mean_w_and_ess(log_w_batch)
        assert abs(log_mean_w) < 0.05, f"Forward log E[w] = {log_mean_w:.4f}"
        assert ess_frac > 0.85, f"Forward ESS/N = {ess_frac:.3f}"


# ══════════════════════════════════════════════════════════════
# 3D Gaussian backward transport
# ══════════════════════════════════════════════════════════════


def test_3d_gaussian_backward_path_action():
    """Verify backward path action on isotropic 3D Gaussian transport.

    N(0, I) → N([1.5, −2, 0.5], 0.09I) with ε = 0.2.
    The multivariate setting tests that leaf-by-leaf residual accumulation
    works correctly in higher dimensions.
    """
    d = 3
    sigma0, sigma1, eps_val = 1.0, 0.3, 0.2
    mu1 = jnp.array([1.5, -2.0, 0.5])

    transport = _make_gaussian_transport(sigma0, mu1, sigma1, eps_val, num_steps=150)
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.full((d,), sigma0))
    target_dist = dsx.MultivariateNormalDiag(mu1, jnp.full((d,), sigma1))

    B = 2000
    keys = jax.random.split(jax.random.key(600), B)

    def single_backward(k):
        k_t, k_s = jax.random.split(k)
        y = target_dist.sample(key=k_t)
        x0, action_bwd = transport.backward_and_path_action(y, key=k_s)
        log_weight_bwd = target_dist.log_prob(y) + action_bwd
        log_w = base_dist.log_prob(x0) - log_weight_bwd
        return x0, log_w

    x0_batch, log_w_batch = jax.vmap(single_backward)(keys)

    log_mean_w, ess_frac = _log_mean_w_and_ess(log_w_batch)
    assert abs(log_mean_w) < 0.10, f"3D backward log E[w] = {log_mean_w:.4f}"
    assert ess_frac > 0.70, f"3D backward ESS/N = {ess_frac:.3f}"

    mean_x0 = jnp.mean(x0_batch, axis=0)
    assert jnp.allclose(mean_x0, jnp.zeros(d), atol=0.20), f"Mean x0 = {mean_x0}"


# ══════════════════════════════════════════════════════════════
# Backward with structured PyTree state
# ══════════════════════════════════════════════════════════════


def test_backward_pytree_state():
    """Backward transport on multi-leaf structured PyTree state.

    Verifies that the solver correctly handles PyTree states where x is
    a dict with heterogeneous-shaped leaves.
    """
    state = {
        "pos": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "vel": jnp.array([0.5, -0.5]),
    }
    key = jax.random.key(700)

    transport = SDETransport(
        velocity_field=lambda t, x, a: jax.tree.map(lambda leaf: -0.5 * leaf, x),
        score_field=lambda t, x, a: jax.tree.map(lambda leaf: -leaf, x),
        noise_schedule=lambda t: 0.1,
        num_steps=50,
    )

    x, action = transport.backward_and_log_weight(state, key=key)
    assert x["pos"].shape == state["pos"].shape
    assert x["vel"].shape == state["vel"].shape
    assert jnp.isfinite(action)
    assert not jnp.allclose(x["pos"], state["pos"])


# ══════════════════════════════════════════════════════════════
# Backward with dynamic masking (partial DOF update)
# ══════════════════════════════════════════════════════════════


def test_backward_sde_dynamic_masking():
    """Backward SDE preserves static DOFs under dynamic masking.

    Only indices [0, 2, 4] are dynamic; indices [1, 3] must remain
    unchanged after backward transport.
    """
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = jnp.array([0, 2, 4])
    complement = jnp.array([1, 3])
    key = jax.random.key(800)

    transport = SDETransport(
        velocity_field=lambda t, x, a: -x,
        score_field=lambda t, x, a: -x,
        noise_schedule=lambda t: 0.1,
        num_steps=50,
    )

    y, action = transport.backward_and_log_weight(x, dynamic_mask=mask, key=key)
    assert jnp.allclose(y[complement], x[complement]), "Static DOFs must be unchanged"
    assert not jnp.allclose(y[mask], x[mask]), "Dynamic DOFs must change"
    assert jnp.isfinite(action)


# ══════════════════════════════════════════════════════════════
# Bijector backward == inverse
# ══════════════════════════════════════════════════════════════


def test_bijector_backward_equals_inverse():
    """For bijective transports, backward() must equal inverse().

    Uses an ODEBijector to verify that the AbstractBijector bridge
    correctly delegates backward to inverse, including the log-weight
    sign convention: backward returns inv_logdet = −fwd_logdet.
    """
    from superiorflows.bijector import AbstractBijector
    from superiorflows.ode import ODEBijector

    def velocity(t, x, args):
        return -0.5 * x

    bijector = ODEBijector(velocity_field=velocity)
    assert isinstance(bijector, AbstractBijector)

    x0 = jnp.array([1.0, 2.0, -1.5])

    y, fwd_logdet = bijector.forward_and_log_det(x0)
    x_inv, inv_logdet = bijector.inverse_and_log_det(y)
    x_bwd, bwd_logweight = bijector.backward_and_log_weight(y)

    assert jnp.allclose(x_inv, x_bwd, atol=1e-5), "backward ≠ inverse"
    assert jnp.allclose(inv_logdet, bwd_logweight, atol=1e-5), "backward log-weight ≠ inverse log-det"
    assert jnp.allclose(fwd_logdet, -inv_logdet, atol=1e-4), "log-det signs inconsistent"


def test_bijector_backward_roundtrip():
    """Forward then backward recovers the original state."""
    from superiorflows.ode import ODEBijector

    bijector = ODEBijector(velocity_field=lambda t, x, a: -0.3 * x)
    x0 = jnp.array([2.0, -1.0, 0.5])

    y = bijector.forward(x0)
    x_rec = bijector.backward(y)
    assert jnp.allclose(x0, x_rec, atol=1e-4), f"Roundtrip error: {jnp.max(jnp.abs(x0 - x_rec)):.2e}"


# ══════════════════════════════════════════════════════════════
# PartialUpdater backward methods
# ══════════════════════════════════════════════════════════════


def test_partial_updater_backward_sde():
    """PartialUpdater.backward_update_and_log_weight with SDE transport.

    Verifies that backward_update correctly applies backward transport
    on the dynamic subset while preserving static context.
    """
    from superiorflows.partial import PartialUpdater

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.array([0, 2])
    complement = jnp.array([1, 3])

    transport = SDETransport(
        velocity_field=lambda t, x, a: -0.5 * x,
        score_field=lambda t, x, a: -x,
        noise_schedule=lambda t: 0.1,
        num_steps=30,
    )
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2))

    updater = PartialUpdater(transport=transport, base_distribution=base_dist)

    y, log_w = updater.backward_update_and_log_weight(x, dynamic_mask=mask, key=jax.random.key(900))
    assert jnp.allclose(y[complement], x[complement]), "Static DOFs must be unchanged"
    assert not jnp.allclose(y[mask], x[mask]), "Dynamic DOFs must change"
    assert jnp.isfinite(log_w)

    y_plain = updater.backward_update(x, dynamic_mask=mask, key=jax.random.key(901))
    assert jnp.allclose(y_plain[complement], x[complement])

    y_alias, log_w_alias = updater.pull_back_and_log_weight(x, dynamic_mask=mask, key=jax.random.key(900))
    assert jnp.allclose(y, y_alias, atol=1e-6)
    assert jnp.allclose(log_w, log_w_alias, atol=1e-6)


def test_partial_updater_backward_bijector():
    """PartialUpdater.backward_update with bijective transport.

    Verifies backward_update delegates correctly to the bijector's
    inverse operation through the AbstractBijector bridge.
    """
    from superiorflows.ode import ODEBijector
    from superiorflows.partial import PartialUpdater

    x = jnp.array([1.0, 2.0, 3.0])
    mask = jnp.array([0, 2])
    complement = jnp.array([1])

    bijector = ODEBijector(velocity_field=lambda t, x, a: -0.5 * x)
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2))

    updater = PartialUpdater(transport=bijector, base_distribution=base_dist)

    y, log_w = updater.backward_update_and_log_weight(x, dynamic_mask=mask)
    assert jnp.allclose(y[complement], x[complement])
    assert jnp.isfinite(log_w)


# ══════════════════════════════════════════════════════════════
# SDE backward per-call overrides
# ══════════════════════════════════════════════════════════════


def test_backward_per_call_overrides():
    """Backward transport respects per-call num_steps override."""
    x = jnp.array([1.0, 2.0])
    key = jax.random.key(1000)

    transport = SDETransport(
        velocity_field=lambda t, x, a: -x,
        score_field=lambda t, x, a: -x,
        noise_schedule=lambda t: 0.1,
        num_steps=50,
    )

    y1 = transport.backward(x, key=key, num_steps=10)
    y2 = transport.backward(x, key=key, num_steps=100)
    assert y1.shape == x.shape
    assert y2.shape == x.shape
    assert not jnp.allclose(y1, y2), "Different step counts should give different results"
