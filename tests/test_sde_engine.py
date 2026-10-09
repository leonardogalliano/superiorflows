"""Exactness tests for the SDETransport Euler–Maruyama engine.

The engine draws Brownian increments inside the scan from per-step keys and
carries the fields ``(b, u)`` evaluated at ``(t_{k+1}, X_{k+1})`` from the
backward kernel of step ``k`` to the forward kernel of step ``k + 1``. These
tests pin down that this is an implementation detail only:

1. Agreement to float64 round-off with a direct, step-by-step reference
   implementation of the Euler–Maruyama kernels (no field reuse), fed the
   same Brownian increments — forward/backward, score/osmotic, PyTree state,
   path action and its gradient.
2. The sampled step is drawn from exactly the forward kernel p⁺ used in the
   action: for an arbitrary (deliberately wrong) drift, the importance-sampling
   estimate of a normalised target's Z stays unbiased.
3. Field evaluation counts: ``M + 1`` with the path action, ``M`` without.
4. Compiled memory does not grow with the number of steps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from superiorflows.sde import SDETransport


def reference_em_with_action(transport, x0, step_noise, t0, t1, backward):
    """Textbook Euler–Maruyama with Girsanov action, every field evaluated afresh.

    ``step_noise[k]`` is the standard-normal PyTree used at step ``k``. Forward in
    time the generating drift is ``b + u`` and the ghost drift ``b − u``;
    backward in time the roles are swapped.
    """
    sign = -1.0 if backward else 1.0
    num_steps = len(step_noise)
    dt = (t1 - t0) / num_steps
    d_total = sum(leaf.size for leaf in jax.tree.leaves(x0))

    def drift(t, x, s):
        b = transport.velocity_field(t, x, None)
        if transport.osmotic_drift is not None:
            u = transport.osmotic_drift(t, x, None)
        else:
            u = jax.tree.map(lambda si: transport.noise_schedule(t) * si, transport.score_field(t, x, None))
        return jax.tree.map(lambda bi, ui: bi + s * ui, b, u)

    def log_gauss(y, mean, var):
        sq = sum(jnp.sum((yl - ml) ** 2) for yl, ml in zip(jax.tree.leaves(y), jax.tree.leaves(mean)))
        return -0.5 * sq / var - 0.5 * d_total * jnp.log(2.0 * jnp.pi * var)

    x, action = x0, 0.0
    for k in range(num_steps):
        t_k, t_k1 = abs(t0 + k * dt), abs(t0 + (k + 1) * dt)
        var_fwd = 2.0 * transport.noise_schedule(t_k) * abs(dt)
        var_bwd = 2.0 * transport.noise_schedule(t_k1) * abs(dt)
        mean_fwd = jax.tree.map(lambda xi, m: xi + m * dt, x, drift(t_k, x, sign))
        x_next = jax.tree.map(lambda m, w: m + jnp.sqrt(var_fwd) * w, mean_fwd, step_noise[k])
        mean_bwd = jax.tree.map(lambda xi, m: xi - m * dt, x_next, drift(t_k1, x_next, -sign))
        action = action + log_gauss(x_next, mean_fwd, var_fwd) - log_gauss(x, mean_bwd, var_bwd)
        x = x_next
    return x, action


def engine_noise(key, x0, num_steps):
    """The standard normals the engine draws at each step for a given ``key``."""
    step_keys = jax.random.split(key, num_steps)
    return [SDETransport._standard_normal_like(step_keys[k], x0) for k in range(num_steps)]


def make_fields(d, use_osmotic):
    """Nonlinear, time-dependent fields so that reusing a field at the wrong point would show."""
    w = jnp.linspace(-1.0, 1.0, d)

    def velocity(t, x, args):
        return jax.tree.map(lambda xi: jnp.sin(3.0 * t + w * xi) - (1.0 + t) * xi, x)

    def score(t, x, args):
        return jax.tree.map(lambda xi: -xi / (1.0 + t**2) + 0.3 * jnp.cos(xi), x)

    def noise_schedule(t):
        return 0.4 + 0.5 * t * (1.0 - t)

    if use_osmotic:

        def osmotic(t, x, args):
            return jax.tree.map(lambda si: noise_schedule(t) * si, score(t, x, args))

        return dict(velocity_field=velocity, osmotic_drift=osmotic, noise_schedule=noise_schedule)
    return dict(velocity_field=velocity, score_field=score, noise_schedule=noise_schedule)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("use_osmotic", [False, True])
@pytest.mark.parametrize("pytree", [False, True])
def test_engine_matches_reference_kernels(backward, use_osmotic, pytree):
    """Engine and step-by-step reference agree to float64 round-off, given the same noise."""
    d, num_steps = 5, 40
    t0, t1 = (1.0, 0.0) if backward else (0.0, 1.0)
    with jax.enable_x64(True):
        transport = SDETransport(**make_fields(d, use_osmotic), num_steps=num_steps)
        x0 = jax.random.normal(jax.random.key(0), (d,), dtype=jnp.float64)
        if pytree:
            x0 = {"a": x0, "b": 0.5 * x0[::-1]}
        key = jax.random.key(1)

        sol = transport._integrate_with_action(x0, key=key, t0=t0, t1=t1, backward=backward)
        x_engine = jax.tree.map(lambda v: v[-1], sol.ys["x"])
        action_engine = sol.ys["action"][-1]
        x_plain = jax.tree.map(lambda v: v[-1], transport._integrate(x0, key=key, t0=t0, t1=t1, backward=backward).ys)

        x_ref, action_ref = reference_em_with_action(transport, x0, engine_noise(key, x0, num_steps), t0, t1, backward)

        for a, b, c in zip(jax.tree.leaves(x_engine), jax.tree.leaves(x_ref), jax.tree.leaves(x_plain)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
            np.testing.assert_allclose(c, b, rtol=0, atol=1e-12)
        np.testing.assert_allclose(action_engine, action_ref, rtol=0, atol=1e-10)


def test_engine_gradient_matches_reference():
    """Reverse-mode gradients of the path action w.r.t. field parameters agree with the reference."""
    d, num_steps = 4, 25
    with jax.enable_x64(True):
        x0 = jax.random.normal(jax.random.key(0), (d,), dtype=jnp.float64)
        key = jax.random.key(1)
        noise = engine_noise(key, x0, num_steps)
        theta0 = jnp.array([0.7, -0.3, 1.2], dtype=jnp.float64)

        def build(theta):
            return SDETransport(
                velocity_field=lambda t, x, args: theta[0] * jnp.sin(x + t) - theta[1] * x,
                score_field=lambda t, x, args: -theta[2] * x,
                noise_schedule=lambda t: 0.5,
                num_steps=num_steps,
            )

        g_engine = jax.grad(lambda th: build(th)._integrate_with_action(x0, key=key).ys["action"][-1])(theta0)
        g_ref = jax.grad(lambda th: reference_em_with_action(build(th), x0, noise, 0.0, 1.0, False)[1])(theta0)
        np.testing.assert_allclose(g_engine, g_ref, rtol=1e-10, atol=1e-10)


def test_wrong_drift_still_unbiased():
    """Samples are drawn from exactly the p⁺ used in the action.

    For any drift, E[ρ(X_M) e^{−S_M} / q₀(X_0)] = Z when X_{k+1} ~ p⁺(·|X_k). A
    mismatch between the drift used to move the walkers and the drift used in
    log p⁺ would bias this. Target N(1, 0.5²) is normalised, so Z = 1.
    """
    transport = SDETransport(
        velocity_field=lambda t, x, args: 1.0 + 0.5 * jnp.sin(3.0 * x),
        score_field=lambda t, x, args: -x,
        noise_schedule=lambda t: 0.5,
        num_steps=20,
    )

    def log_weight(key):
        k0, k1 = jax.random.split(key)
        x0 = jax.random.normal(k0, (1,))
        sol = transport._integrate_with_action(x0, key=k1)
        y = sol.ys["x"][-1]
        log_q0 = -0.5 * jnp.sum(x0**2) - 0.5 * jnp.log(2.0 * jnp.pi)
        log_rho = jnp.sum(-0.5 * ((y - 1.0) / 0.5) ** 2 - jnp.log(0.5) - 0.5 * jnp.log(2.0 * jnp.pi))
        return log_rho - log_q0 - sol.ys["action"][-1]

    log_w = np.asarray(jax.jit(jax.vmap(log_weight))(jax.random.split(jax.random.key(0), 200_000)), np.float64)
    w = np.exp(log_w)
    z_hat, sem = w.mean(), w.std() / np.sqrt(len(w))
    ess = w.sum() ** 2 / (len(w) * (w**2).sum())
    assert ess > 0.1, f"ESS/N = {ess:.3f} too low for the test to be meaningful"
    assert abs(z_hat - 1.0) < 4.0 * sem, f"Z-hat = {z_hat:.4f} ± {sem:.4f}, expected 1"


@pytest.mark.parametrize("with_action", [False, True])
def test_field_evaluation_count(with_action):
    """M + 1 evaluations of each field with the path action, M without (counted eagerly)."""
    num_steps = 7
    calls = {"b": 0, "s": 0}

    def velocity(t, x, args):
        calls["b"] += 1
        return -x

    def score(t, x, args):
        calls["s"] += 1
        return -x

    transport = SDETransport(
        velocity_field=velocity, score_field=score, noise_schedule=lambda t: 1.0, num_steps=num_steps
    )
    with jax.disable_jit():
        if with_action:
            transport._integrate_with_action(jnp.ones(3), key=jax.random.key(0))
        else:
            transport._integrate(jnp.ones(3), key=jax.random.key(0))
    expected = num_steps + 1 if with_action else num_steps
    assert calls == {"b": expected, "s": expected}


def test_memory_independent_of_num_steps():
    """Compiled temporary memory must not scale with the number of steps."""
    d, batch = 512, 64

    def temp_bytes(num_steps):
        transport = SDETransport(
            velocity_field=lambda t, x, args: jnp.tanh(x) - x,
            score_field=lambda t, x, args: -x,
            noise_schedule=lambda t: 1.0,
            num_steps=num_steps,
        )

        def run(key):
            keys = jax.random.split(key, batch)
            x0 = jnp.zeros((batch, d))
            return jax.vmap(lambda x, k: transport._integrate_with_action(x, key=k).ys["action"])(x0, keys)

        analysis = jax.jit(run).lower(jax.random.key(0)).compile().memory_analysis()
        if analysis is None:
            pytest.skip("Backend does not report compiled memory.")
        return analysis.temp_size_in_bytes

    small, large = temp_bytes(100), temp_bytes(1000)
    # Pre-sampled noise would make this ratio ≈ 10; per-step keys only add 8 bytes per step and sample.
    assert large < 1.5 * small + batch * 1000 * 8, f"temp memory {small} B (M=100) vs {large} B (M=1000)"
