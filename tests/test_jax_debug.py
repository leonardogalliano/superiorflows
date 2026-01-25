import time

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from superiorflows import Flow

# ============================================================================
# Setup
# ============================================================================


class LinearVelocityField(eqx.Module):
    """Simple linear velocity field: v(t, x) = A @ x + b."""

    A: jax.Array
    b: jax.Array

    def __call__(self, t, x, args):
        return self.A @ x + self.b


@pytest.fixture
def debug_setup():
    """Sets up a simple flow for debugging."""
    # 2D problem
    d = 2

    # Base distribution: Standard Normal
    base_dist = dsx.MultivariateNormalDiag(loc=jnp.zeros(d), scale_diag=jnp.ones(d))

    # Velocity field parameters
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    A = jax.random.normal(k1, (d, d)) * 0.5
    b = jax.random.normal(k2, (d,)) * 0.5

    velocity = LinearVelocityField(A=A, b=b)

    flow = Flow(
        velocity_field=velocity,
        base_distribution=base_dist,
        t0=0.0,
        t1=1.0,
        solver=dfx.Tsit5(),  # Default solver
    )

    return flow, k3


# ============================================================================
# 1. Recompilation Check
# ============================================================================


def test_recompilation_check(debug_setup, capsys):
    """
    Pedagogical test to verify JIT compilation behavior.

    Goal: Ensure that `sample_and_log_prob` is compiled exactly once
    for a given input shape, and not recompiled on subsequent calls.
    """
    flow, key = debug_setup
    sample_shape = (10,)

    # We use a side-effect (printing) to detect compilation.
    # In JAX, Python side-effects inside a JIT-ed function run ONLY during tracing (compilation).

    @jax.jit
    def monitored_function(k):
        jax.debug.print("--- TRACING (Compiling) ---")
        return flow.sample_and_log_prob(seed=k, sample_shape=sample_shape)

    print("\n\n>>> Starting Recompilation Test")

    # 1st Call: Should trigger compilation
    print("Call 1 (Should compile):")
    k1, k2 = jax.random.split(key)
    monitored_function(k1)

    # 2nd Call: Should use cached executable
    print("Call 2 (Should use cache):")
    monitored_function(k2)

    # Check captured stdout
    # Check captured stdout
    _ = capsys.readouterr()

    # We expect "TRACING" to appear in the output (since jax.debug.print outputs to stdout/stderr)
    # Note: jax.debug.print behaves slightly differently than print() inside JIT.
    # Actually, jax.debug.print runs at EXECUTION time, designed for runtime values.
    # To check compilation, we should use a standard python print().

    recompilation_counter = {"count": 0}

    @jax.jit
    def traced_function(k):
        # This python print only executes during tracing
        print("!!! COMPILING !!!")
        # We can also update a mutable object, but that's a bit "unsafe" in general,
        # though standard for detecting trace-time execution.
        recompilation_counter["count"] += 1
        return flow.sample_and_log_prob(seed=k, sample_shape=sample_shape)

    print("Call A (Compile):")
    traced_function(k1)

    print("Call B (Cache):")
    traced_function(k2)

    # Verify we only compiled once
    assert (
        recompilation_counter["count"] == 1
    ), f"Function recompiled {recompilation_counter['count']} times! expected 1."

    print(">>> Recompilation Test Passed: Function compiled exactly once.")


# ============================================================================
# 2. Profiling (Compilation vs Execution)
# ============================================================================


def test_pedagogical_profiling(debug_setup):
    """
    Profiles the flow to show the cost of compilation vs execution.
    """
    flow, key = debug_setup
    sample_shape = (128,)

    @jax.jit
    def run_step(k):
        return flow.sample_and_log_prob(seed=k, sample_shape=sample_shape)

    print("\n\n>>> Starting Profiling Test")
    k1, k2 = jax.random.split(key)

    # --- Compilation Step ---
    start_time = time.perf_counter()
    samples, log_probs = run_step(k1)
    samples.block_until_ready()
    log_probs.block_until_ready()
    end_time = time.perf_counter()
    compile_time = end_time - start_time
    print(f"First run (Compilation + Exec): {compile_time:.4f} s")

    # --- Execution Step (Benchmarking) ---
    n_loops = 20
    times = []
    keys = jax.random.split(k2, n_loops)

    # Warmup (optional, but good practice)
    samples, log_probs = run_step(keys[0])
    samples.block_until_ready()
    log_probs.block_until_ready()

    # start_bench = time.perf_counter()
    for i in range(n_loops):
        t0 = time.perf_counter()
        samples, log_probs = run_step(keys[i])
        samples.block_until_ready()
        log_probs.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    # end_bench = time.perf_counter()

    avg_time = sum(times) / n_loops
    print(f"Subsequent runs (Avg Exec):     {avg_time:.4f} s")
    print(f"Speedup factor:                 {compile_time / avg_time:.1f}x")

    # Assert execution is significantly faster than compilation (heuristic)
    # This might fail on very slow machines or tiny models, but usually holds.
    if compile_time > 0.1:  # Only assert if compilation was actually measurable
        assert avg_time < compile_time, "Execution should be faster than compilation!"


# ============================================================================
# 3. Numerical Health Checks
# ============================================================================


def test_numerical_health(debug_setup):
    """
    Checks for NaNs and Infs in the output.
    """
    flow, key = debug_setup
    sample_shape = (100,)

    samples, log_probs = flow.sample_and_log_prob(seed=key, sample_shape=sample_shape)

    print("\n\n>>> Numerical Health Check")
    print(f"Sample stats: mean={jnp.mean(samples):.3f}, std={jnp.std(samples):.3f}")
    print(f"LogProb stats: mean={jnp.mean(log_probs):.3f}, min={jnp.min(log_probs):.3f}, max={jnp.max(log_probs):.3f}")

    # Check for NaNs
    nans_in_samples = jnp.isnan(samples).any()
    nans_in_logp = jnp.isnan(log_probs).any()

    if nans_in_samples:
        print("WARNING: NaNs found in samples!")
    if nans_in_logp:
        print("WARNING: NaNs found in log_probs!")

    assert not nans_in_samples, "Samples contain NaNs"
    assert not nans_in_logp, "Log probabilities contain NaNs"

    # Check for Infs
    infs_in_samples = jnp.isinf(samples).any()
    assert not infs_in_samples, "Samples contain Infs"

    print(">>> Numerical Health Passed: No NaNs or Infs found.")


# ============================================================================
# 4. Integration Debugging (ODE Solver Diagnostics)
# ============================================================================


def test_ode_diagnostics(debug_setup):
    """
    Inspects the internal steps taken by the ODE solver.
    """
    flow, key = debug_setup
    x0 = flow.base_distribution.sample(seed=key)

    # We want to see the trajectory steps.
    # Flow.integrate returns the solution object.

    print("\n\n>>> ODE Diagnostics")

    # We use SaveAt(steps=True) to save every step taken by the solver
    sol = flow.integrate(x0, saveat=dfx.SaveAt(steps=True))

    ts = sol.ts  # Times at which steps were taken
    # ys = sol.ys  # State values at those times

    # ts will have shape (num_steps_cap,) with inf padding if fewer steps taken
    # We count valid steps (where t is not inf/nan and <= t1)
    # Actually diffrax pads with inf usually.

    # Count valid steps
    valid_steps = jnp.isfinite(ts)
    num_steps = jnp.sum(valid_steps)

    print(f"Solver used {num_steps} steps (avg over batch if batched, but here time is scalar shared).")
    print(f"Time points (first 10): {ts[:10]}")

    # Check if we hit max steps (default is usually 4096 in diffrax)
    # If we are close to max steps, the system might be stiff.
    if num_steps > 1000:
        print("WARNING: High number of steps! System might be stiff.")

    # Check final time
    t_final = ts[jnp.argmax(valid_steps * jnp.arange(ts.shape[0]))]
    print(f"Final integration time reached: {t_final:.4f} (Target: {flow.t1})")

    assert jnp.isclose(t_final, flow.t1, atol=1e-3), "Solver did not reach final time t1!"
    print(">>> ODE Diagnostics Passed.")
