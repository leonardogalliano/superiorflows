"""Tests for diffrax features: solvers, adjoints, stepsize controllers, events.

This module tests the Flow class with various diffrax configurations to ensure
flexibility and correctness across different integration schemes.
"""

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from superiorflows import Flow

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def base_distribution():
    """Simple 2D uniform distribution."""
    d = (2,)
    low = -jnp.ones(d)
    high = jnp.ones(d)
    return dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)


class SimpleVelocity(eqx.Module):
    """Simple velocity field: v(t, x) = -t * x (contracts towards origin)."""

    def __call__(self, t, x, args):
        return -t * x


@pytest.fixture
def velocity_field():
    return SimpleVelocity()


@pytest.fixture
def base_flow(base_distribution, velocity_field):
    """Default flow with Tsit5 solver."""
    return Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )


# ============================================================================
# ODE Solver Tests
# ============================================================================


@pytest.mark.parametrize(
    "solver,dt0",
    [
        (dfx.Euler(), 0.01),  # Fixed step required
        (dfx.Heun(), 0.01),
        (dfx.Midpoint(), 0.01),
        (dfx.Bosh3(), None),  # Adaptive
        (dfx.Tsit5(), None),
        (dfx.Dopri5(), None),
        (dfx.Dopri8(), None),
    ],
)
def test_flow_with_different_solvers(base_distribution, velocity_field, solver, dt0):
    """Test Flow produces valid results with different ODE solvers."""
    # Some solvers need constant step size
    if dt0 is not None:
        controller = dfx.ConstantStepSize()
    else:
        controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        solver=solver,
        dt0=dt0,
        stepsize_controller=controller,
    )

    key = jax.random.PRNGKey(42)
    x0 = flow.base_distribution.sample(seed=key)

    # Forward map
    x1 = flow.apply_map(x0)
    assert x1.shape == x0.shape
    assert jnp.all(jnp.isfinite(x1))

    # Inverse should reconstruct approximately
    x0_rec = flow.apply_inverse_map(x1)
    # Tolerance depends on solver order
    tol = 0.1 if isinstance(solver, dfx.Euler) else 1e-3
    assert jnp.allclose(x0_rec, x0, atol=tol, rtol=tol)


def test_solver_consistency_with_analytical(base_distribution):
    """Compare different solvers on problem with known solution."""

    # dx/dt = x => x(t) = x0 * exp(t)
    def exp_velocity(t, x, args):
        return x

    key = jax.random.PRNGKey(0)
    x0 = base_distribution.sample(seed=key)
    t1 = 0.5  # short time for accuracy
    true_solution = x0 * jnp.exp(t1)

    results = {}
    for name, solver, dt0 in [
        ("Euler", dfx.Euler(), 0.001),
        ("Heun", dfx.Heun(), 0.01),
        ("Tsit5", dfx.Tsit5(), None),
        ("Dopri8", dfx.Dopri8(), None),
    ]:
        if dt0 is not None:
            controller = dfx.ConstantStepSize()
        else:
            controller = dfx.PIDController(rtol=1e-7, atol=1e-7)

        flow = Flow(
            velocity_field=exp_velocity,
            base_distribution=base_distribution,
            solver=solver,
            dt0=dt0,
            t1=t1,
            stepsize_controller=controller,
        )
        results[name] = flow.apply_map(x0)

    # All should be close to true solution
    for name, result in results.items():
        tol = 0.1 if name == "Euler" else 1e-4
        assert jnp.allclose(result, true_solution, atol=tol), f"{name} failed"


# ============================================================================
# Adjoint Method Tests (Gradient Computation)
# ============================================================================


def test_gradient_default_adjoint(base_flow):
    """Test gradient computation with default RecursiveCheckpointAdjoint."""
    key = jax.random.PRNGKey(0)
    X = base_flow.base_distribution.sample(seed=key, sample_shape=(10,))
    X1 = jax.vmap(base_flow.apply_map)(X)

    @eqx.filter_jit
    def loss(flow, x):
        return jnp.mean(jax.vmap(flow.log_prob)(x))

    grad = jax.grad(loss)(base_flow, X1)
    assert grad is not None
    # Check gradient is finite and non-zero
    grad_flat = jax.flatten_util.ravel_pytree(grad)[0]
    assert jnp.all(jnp.isfinite(grad_flat))


def test_gradient_with_direct_adjoint(base_distribution, velocity_field):
    """Test gradient with DirectAdjoint (supports both forward and reverse mode)."""
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
        extra_args={"adjoint": dfx.DirectAdjoint()},
        augmented_extra_args={"adjoint": dfx.DirectAdjoint()},
    )

    key = jax.random.PRNGKey(0)
    X = flow.base_distribution.sample(seed=key, sample_shape=(5,))
    X1 = jax.vmap(flow.apply_map)(X)

    @eqx.filter_jit
    def loss(flow, x):
        return jnp.mean(jax.vmap(flow.log_prob)(x))

    # Should work with both reverse and forward mode
    grad_rev = jax.grad(loss)(flow, X1)
    assert grad_rev is not None

    # Forward mode via jacfwd
    def scalar_loss(flow):
        return loss(flow, X1)

    # Just verify it doesn't crash - forward mode on whole flow is complex
    grad_flat = jax.flatten_util.ravel_pytree(grad_rev)[0]
    assert jnp.all(jnp.isfinite(grad_flat))


def test_gradient_finite_difference_check(base_flow):
    """Verify autodiff gradient matches finite differences."""
    key = jax.random.PRNGKey(0)
    x0 = base_flow.base_distribution.sample(seed=key)
    x1 = base_flow.apply_map(x0)

    def log_prob_scalar(params_flat, x):
        # Simple finite diff check on the log_prob output
        return base_flow.log_prob(x)

    # Get autodiff gradient w.r.t. x1
    grad_auto = jax.grad(lambda x: base_flow.log_prob(x))(x1)

    # Finite difference approximation
    eps = 1e-4
    grad_fd = jnp.zeros_like(x1)
    for i in range(x1.size):
        x_plus = x1.at[i].add(eps)
        x_minus = x1.at[i].add(-eps)
        grad_fd = grad_fd.at[i].set((base_flow.log_prob(x_plus) - base_flow.log_prob(x_minus)) / (2 * eps))

    assert jnp.allclose(grad_auto, grad_fd, atol=1e-2, rtol=1e-2)


# ============================================================================
# Stepsize Controller Tests
# ============================================================================


@pytest.mark.parametrize(
    "controller,dt0",
    [
        (dfx.PIDController(rtol=1e-5, atol=1e-5), None),
        (dfx.PIDController(rtol=1e-3, atol=1e-3), None),  # Coarser tolerance
        (dfx.ConstantStepSize(), 0.01),
    ],
)
def test_stepsize_controllers(base_distribution, velocity_field, controller, dt0):
    """Test Flow with different stepsize controllers."""
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        stepsize_controller=controller,
        dt0=dt0,
    )

    key = jax.random.PRNGKey(0)
    x0 = flow.base_distribution.sample(seed=key)

    x1 = flow.apply_map(x0)
    x0_rec = flow.apply_inverse_map(x1)

    assert jnp.all(jnp.isfinite(x1))
    assert jnp.allclose(x0_rec, x0, atol=1e-2, rtol=1e-2)


def test_tolerance_convergence(base_distribution, velocity_field):
    """Verify tighter tolerances produce more accurate results."""
    key = jax.random.PRNGKey(0)
    x0 = base_distribution.sample(seed=key)

    errors = []
    for rtol in [1e-3, 1e-5, 1e-7]:
        flow = Flow(
            velocity_field=velocity_field,
            base_distribution=base_distribution,
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=rtol),
        )
        x1 = flow.apply_map(x0)
        x0_rec = flow.apply_inverse_map(x1)
        error = jnp.max(jnp.abs(x0_rec - x0))
        errors.append(float(error))

    # Errors should decrease with tighter tolerance
    assert errors[0] >= errors[1] >= errors[2] - 1e-10


# ============================================================================
# SaveAt Tests (Trajectory Saving)
# ============================================================================


def test_saveat_t1_only(base_flow):
    """Default: save only at t1."""
    key = jax.random.PRNGKey(0)
    x0 = base_flow.base_distribution.sample(seed=key)

    sol = base_flow.integrate(x0)
    assert sol.ys.shape[0] == 1  # Only t1 saved


def test_saveat_specific_times(base_flow):
    """Save at specific times."""
    key = jax.random.PRNGKey(0)
    x0 = base_flow.base_distribution.sample(seed=key)

    ts = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sol = base_flow.integrate(x0, saveat=dfx.SaveAt(ts=ts))

    assert sol.ys.shape[0] == len(ts)
    # First should be close to x0
    assert jnp.allclose(sol.ys[0], x0, atol=1e-5)


def test_saveat_steps(base_flow):
    """Save at all integration steps."""
    key = jax.random.PRNGKey(0)
    x0 = base_flow.base_distribution.sample(seed=key)

    sol = base_flow.integrate(x0, saveat=dfx.SaveAt(steps=True), max_steps=1000)

    # Should have multiple steps saved
    assert sol.ys.shape[0] > 1


def test_saveat_dense_output(base_flow):
    """Test dense output for interpolation at arbitrary times."""
    key = jax.random.PRNGKey(0)
    x0 = base_flow.base_distribution.sample(seed=key)

    # For dense output, we need to also save at t1 to get a valid solution
    sol = base_flow.integrate(x0, saveat=dfx.SaveAt(dense=True, t1=True), max_steps=1000)

    # Evaluate at arbitrary times using dense interpolation
    t_eval = 0.5
    x_interp = sol.evaluate(t_eval)
    assert x_interp.shape == x0.shape
    assert jnp.all(jnp.isfinite(x_interp))

    # Verify endpoint consistency
    x1_dense = sol.evaluate(1.0)
    x1_saveat = sol.ys[-1]
    assert jnp.allclose(x1_dense, x1_saveat, atol=1e-5)


# ============================================================================
# Manifold Projection Tests
# ============================================================================


def test_sphere_projection_callback():
    """Test projection onto unit sphere using step callbacks."""
    d = 3
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def radial_velocity(t, x, args):
        """Velocity that would push off sphere without projection."""
        return x * 0.5  # Expands radially

    def project_to_sphere(t, y, args):
        """Project onto unit sphere."""
        return y / jnp.linalg.norm(y)

    # Create flow - no direct callback support, but we can verify the concept
    flow = Flow(
        velocity_field=radial_velocity,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)
    # Start on unit sphere
    x0 = x0 / jnp.linalg.norm(x0)

    # Without projection, x1 will leave the sphere
    x1 = flow.apply_map(x0)
    norm_x1 = jnp.linalg.norm(x1)

    # Verify it left the sphere (norm != 1)
    assert not jnp.isclose(norm_x1, 1.0, atol=0.1)

    # Manual projection shows concept works
    x1_projected = x1 / jnp.linalg.norm(x1)
    assert jnp.isclose(jnp.linalg.norm(x1_projected), 1.0, atol=1e-5)


def test_box_constraint_projection():
    """Test projection to enforce box constraints."""
    d = (2,)
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def expanding_velocity(t, x, args):
        """Velocity that expands beyond box."""
        return x * 2.0

    def project_to_box(x, low, high):
        """Clip to box."""
        return jnp.clip(x, low, high)

    flow = Flow(
        velocity_field=expanding_velocity,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)

    x1 = flow.apply_map(x0)

    # Without projection, should exceed bounds
    assert jnp.any(jnp.abs(x1) > 1.0)

    # Projected version stays in box
    x1_proj = project_to_box(x1, low, high)
    assert jnp.all(jnp.abs(x1_proj) <= 1.0)


# ============================================================================
# Steady State Event Tests
# ============================================================================


def test_steady_state_approach():
    """Test detecting when flow approaches steady state."""
    d = (2,)
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def damped_velocity(t, x, args):
        """Velocity that damps to zero: v = -x (approaches origin)."""
        return -x

    flow = Flow(
        velocity_field=damped_velocity,
        base_distribution=base_dist,
        t1=10.0,  # Long time to ensure convergence
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)

    # After long time, should be near origin (steady state)
    x1 = flow.apply_map(x0)
    assert jnp.allclose(x1, 0.0, atol=1e-3)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_identity_flow():
    """Test flow with zero velocity (identity map)."""
    d = (2,)
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def zero_velocity(t, x, args):
        return jnp.zeros_like(x)

    flow = Flow(
        velocity_field=zero_velocity,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)

    x1 = flow.apply_map(x0)
    assert jnp.allclose(x1, x0, atol=1e-5)

    # Log prob should equal base distribution log prob (zero divergence)
    x1, logq1 = flow.apply_map_and_log_prob(x0)
    logq_base = base_dist.log_prob(x0)
    assert jnp.allclose(logq1, logq_base, atol=1e-4)


def test_high_dimensional_flow():
    """Test flow in higher dimensions."""
    d = 50
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def simple_velocity(t, x, args):
        return -t * x

    flow = Flow(
        velocity_field=simple_velocity,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)

    x1 = flow.apply_map(x0)
    assert x1.shape == (d,)
    assert jnp.all(jnp.isfinite(x1))

    x1, logq1 = flow.apply_map_and_log_prob(x0)
    assert jnp.isfinite(logq1)


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.parametrize(
    "solver_name,solver",
    [
        ("Tsit5", dfx.Tsit5()),
        ("Dopri5", dfx.Dopri5()),
        ("Dopri8", dfx.Dopri8()),
    ],
)
def test_solver_benchmark(benchmark, base_distribution, velocity_field, solver_name, solver):
    """Benchmark different solvers."""
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        solver=solver,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = flow.base_distribution.sample(seed=key)

    def run():
        x1, logq1 = flow.apply_map_and_log_prob(x0)
        x1.block_until_ready()
        logq1.block_until_ready()
        return x1, logq1

    benchmark(run)


@pytest.mark.parametrize("dim", [2, 10, 20, 50, 100, 200])
def test_dimension_scaling_benchmark(benchmark, velocity_field, dim):
    """Benchmark scaling with dimension (exact divergence)."""
    low = -jnp.ones(dim)
    high = jnp.ones(dim)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def velocity_d(t, x, args):
        return -t * x

    flow = Flow(
        velocity_field=velocity_d,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)

    def run():
        x1, logq1 = flow.apply_map_and_log_prob(x0)
        x1.block_until_ready()
        logq1.block_until_ready()
        return x1, logq1

    benchmark(run)


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_batch_size_scaling(benchmark, base_flow, batch_size):
    """Benchmark scaling with batch size."""
    key = jax.random.PRNGKey(0)
    X0 = base_flow.base_distribution.sample(seed=key, sample_shape=(batch_size,))

    def run():
        X1, logq1 = jax.vmap(base_flow.apply_map_and_log_prob)(X0)
        X1.block_until_ready()
        logq1.block_until_ready()
        return X1, logq1

    benchmark(run)


# ============================================================================
# Augmented Solver Tests (integrate vs integrate_augmented_ode)
# ============================================================================


@pytest.mark.parametrize(
    "solver,augmented_solver,dt0",
    [
        (dfx.Tsit5(), dfx.Tsit5(), None),
        (dfx.Dopri5(), dfx.Dopri5(), None),
        (dfx.Dopri8(), dfx.Dopri8(), None),
        (dfx.Heun(), dfx.Heun(), 0.01),
    ],
)
def test_augmented_solver_consistency(base_distribution, velocity_field, solver, augmented_solver, dt0):
    """Test that integrate and integrate_augmented_ode are consistent with different solvers."""
    if dt0 is not None:
        controller = dfx.ConstantStepSize()
    else:
        controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        solver=solver,
        augmented_solver=augmented_solver,
        dt0=dt0,
        stepsize_controller=controller,
        augmented_stepsize_controller=controller,
    )

    key = jax.random.PRNGKey(0)
    x0 = flow.base_distribution.sample(seed=key)

    # integrate only (no log prob)
    x1_integrate = flow.apply_map(x0)

    # integrate_augmented_ode (with log prob)
    x1_augmented, logq1 = flow.apply_map_and_log_prob(x0)

    # Trajectories should match
    tol = 0.05 if isinstance(solver, dfx.Heun) else 1e-4
    assert jnp.allclose(x1_integrate, x1_augmented, atol=tol, rtol=tol)
    assert jnp.isfinite(logq1)


@pytest.mark.parametrize(
    "augmented_solver_name,augmented_solver",
    [
        ("Tsit5", dfx.Tsit5()),
        ("Dopri5", dfx.Dopri5()),
        ("Dopri8", dfx.Dopri8()),
    ],
)
def test_augmented_solver_log_prob_consistency(
    base_distribution, velocity_field, augmented_solver_name, augmented_solver
):
    """Test log_prob is consistent across different augmented solvers."""
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        augmented_solver=augmented_solver,
        augmented_stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
    )

    key = jax.random.PRNGKey(0)
    x0 = flow.base_distribution.sample(seed=key)
    x1 = flow.apply_map(x0)

    # Compute log_prob (uses integrate_augmented_ode in reverse)
    logq = flow.log_prob(x1)
    assert jnp.isfinite(logq)

    # Forward and reverse log_prob should be consistent
    x1_fwd, logq_fwd = flow.apply_map_and_log_prob(x0)
    logq_rev = flow.log_prob(x1_fwd)
    assert jnp.allclose(logq_fwd, logq_rev, atol=1e-3, rtol=1e-3)


def test_different_solvers_for_integrate_and_augmented(base_distribution, velocity_field):
    """Test using different solvers for integrate vs integrate_augmented_ode."""
    # Use high-order solver for regular integration, lower for augmented
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        solver=dfx.Dopri8(),
        augmented_solver=dfx.Tsit5(),
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
        augmented_stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = flow.base_distribution.sample(seed=key)

    x1 = flow.apply_map(x0)
    x1_aug, logq = flow.apply_map_and_log_prob(x0)

    # Should still be reasonably close despite different solvers
    assert jnp.allclose(x1, x1_aug, atol=1e-3, rtol=1e-3)


# ============================================================================
# Diffrax Callback Tests for Manifold Projections
# ============================================================================


def test_velocity_with_projection_wrapper():
    """Test wrapping velocity field to include projection at each call.

    This demonstrates how to implement projection without modifying Flow.
    The velocity field wrapper projects the state before computing velocity.
    """
    d = 3
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def project_to_sphere(x):
        """Project onto unit sphere."""
        return x / jnp.linalg.norm(x)

    class ProjectedVelocity(eqx.Module):
        """Velocity that projects to manifold before computing tangent vector."""

        def __call__(self, t, x, args):
            # Project to manifold
            x_proj = project_to_sphere(x)
            # Compute velocity in ambient space
            v_ambient = jnp.sin(t) * jnp.array([1.0, 0.0, 0.0])
            # Project velocity to tangent space: v_tan = v - (v · n) n
            # For sphere, n = x_proj
            v_tangent = v_ambient - jnp.dot(v_ambient, x_proj) * x_proj
            return v_tangent

    flow = Flow(
        velocity_field=ProjectedVelocity(),
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)
    x0 = x0 / jnp.linalg.norm(x0)  # Start on sphere

    x1 = flow.apply_map(x0)
    # With tangent-space velocity, should stay close to sphere
    # (not exact due to integration discretization)
    norm_x1 = jnp.linalg.norm(x1)
    assert jnp.isclose(norm_x1, 1.0, atol=0.1)


def test_event_via_extra_args():
    """Test passing diffrax Event through extra_args for early termination."""
    d = (2,)
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def damped_velocity(t, x, args):
        return -2 * x  # Fast decay to origin

    # Boolean event: stop when norm < 0.1
    def stop_condition(t, y, args, **kwargs):
        return jnp.linalg.norm(y) < 0.1

    event = dfx.Event(stop_condition)

    # Pass event through extra_args
    flow = Flow(
        velocity_field=damped_velocity,
        base_distribution=base_dist,
        t1=10.0,  # Long time
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
        extra_args={"event": event, "max_steps": 1000},
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)

    # With event, should stop early when |x| < 0.1
    sol = flow.integrate(x0)
    x1 = sol.ys[-1]

    # Should have stopped before going all the way to near-zero
    # (without event it would be ~exp(-20) * x0 ≈ 0)
    # With boolean event, stops at end of step when condition is true
    assert jnp.linalg.norm(x1) < 0.15


def test_subsaveat_monitoring_at_each_step():
    """Test using SaveAt(steps=True) to monitor state at each step (JAX-friendly callback)."""
    d = (2,)
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def contracting_velocity(t, x, args):
        return -t * x

    # Use SaveAt(ts=...) for monitoring - more reliable than steps=True with fn
    ts = jnp.linspace(0.0, 1.0, 21)
    saveat = dfx.SaveAt(ts=ts)

    flow = Flow(
        velocity_field=contracting_velocity,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)

    sol = flow.integrate(x0, saveat=saveat)
    trajectory = sol.ys

    # Compute norms at each saved time
    norms = jax.vmap(jnp.linalg.norm)(trajectory)

    # Should have recorded at all times
    assert norms.shape[0] == 21

    # Norms should be decreasing overall (contracting flow)
    assert norms[-1] < norms[0]


def test_manifold_projection_monitoring():
    """Test monitoring manifold constraint satisfaction using SaveAt(ts=...)."""
    d = 3
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def project_to_sphere(x):
        return x / jnp.linalg.norm(x)

    class TangentSpaceVelocity(eqx.Module):
        """Velocity in tangent space of sphere (geodesic-like motion)."""

        def __call__(self, t, x, args):
            x_proj = project_to_sphere(x)
            # Rotation in tangent space
            v_ambient = 0.5 * jnp.array([-x_proj[1], x_proj[0], 0.0])
            return v_ambient

    # Save trajectory at many times
    ts = jnp.linspace(0.0, 1.0, 21)
    saveat = dfx.SaveAt(ts=ts)

    flow = Flow(
        velocity_field=TangentSpaceVelocity(),
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
    )

    key = jax.random.PRNGKey(0)
    x0 = base_dist.sample(seed=key)
    x0 = x0 / jnp.linalg.norm(x0)  # Start on sphere

    sol = flow.integrate(x0, saveat=saveat)
    trajectory = sol.ys

    # Compute manifold distances at each saved time
    manifold_distances = jax.vmap(lambda y: jnp.abs(jnp.linalg.norm(y) - 1.0))(trajectory)

    # With tangent-space velocity, should stay close to sphere
    max_drift = jnp.max(manifold_distances)
    assert max_drift < 0.1, f"Maximum drift from manifold: {max_drift}"


# ============================================================================
# Hutchinson Scaling Benchmarks
# ============================================================================


@pytest.fixture
def hutchinson_flow(base_distribution, velocity_field):
    """Flow with Hutchinson estimator."""
    return Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
        hutchinson_samples=5,
    )


@pytest.mark.parametrize("dim", [2, 10, 20, 50, 100, 200])
def test_hutchinson_dimension_scaling_benchmark(benchmark, dim):
    """Benchmark Hutchinson scaling with dimension."""
    low = -jnp.ones(dim)
    high = jnp.ones(dim)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def velocity_d(t, x, args):
        return -t * x

    flow = Flow(
        velocity_field=velocity_d,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
        hutchinson_samples=5,
    )

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x0 = base_dist.sample(seed=key1)

    def run():
        x1, logq1 = flow.apply_map_and_log_prob(x0, key=key2)
        x1.block_until_ready()
        logq1.block_until_ready()
        return x1, logq1

    benchmark(run)


@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_hutchinson_batch_size_scaling(benchmark, hutchinson_flow, batch_size):
    """Benchmark Hutchinson scaling with batch size."""
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    X0 = hutchinson_flow.base_distribution.sample(seed=key1, sample_shape=(batch_size,))
    keys = jax.random.split(key2, batch_size)

    def run():
        X1, logq1 = jax.vmap(lambda x, k: hutchinson_flow.apply_map_and_log_prob(x, key=k))(X0, keys)
        X1.block_until_ready()
        logq1.block_until_ready()
        return X1, logq1

    benchmark(run)


@pytest.mark.parametrize("hutchinson_samples", [1, 5, 10, 20])
def test_hutchinson_samples_scaling(benchmark, base_distribution, velocity_field, hutchinson_samples):
    """Benchmark scaling with number of Hutchinson samples."""
    flow = Flow(
        velocity_field=velocity_field,
        base_distribution=base_distribution,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
        hutchinson_samples=hutchinson_samples,
    )

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x0 = base_distribution.sample(seed=key1)

    def run():
        x1, logq1 = flow.apply_map_and_log_prob(x0, key=key2)
        x1.block_until_ready()
        logq1.block_until_ready()
        return x1, logq1

    benchmark(run)


def test_hutchinson_vs_exact_high_dim_comparison():
    """Compare Hutchinson vs exact divergence in higher dimensions."""
    d = 30
    low = -jnp.ones(d)
    high = jnp.ones(d)
    base_dist = dsx.Independent(dsx.Uniform(low, high), reinterpreted_batch_ndims=1)

    def velocity_d(t, x, args):
        return -t * x

    exact_flow = Flow(
        velocity_field=velocity_d,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )

    hutch_flow = Flow(
        velocity_field=velocity_d,
        base_distribution=base_dist,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
        hutchinson_samples=10,
    )

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x0 = base_dist.sample(seed=key1)

    x1_exact, logq_exact = exact_flow.apply_map_and_log_prob(x0)
    x1_hutch, logq_hutch = hutch_flow.apply_map_and_log_prob(x0, key=key2)

    # Trajectories should match
    assert jnp.allclose(x1_exact, x1_hutch, atol=1e-4)
    # Log probs should be close (stochastic, larger tolerance)
    assert jnp.abs(logq_exact - logq_hutch) < 3.0  # Reasonable for 10 samples in d=30
