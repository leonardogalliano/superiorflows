"""Continuous Normalizing Flows (CNF) implementation in JAX.

This module provides a flexible, JAX-friendly implementation of continuous
normalizing flows that can handle arbitrary pytree structures as inputs.
This is particularly useful for complex state structures like physical systems.
"""
from typing import Any, Callable, Dict, Optional

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["Flow"]


def _divergence_exact(velocity_field, t, x, args):
    """Compute exact divergence using full Jacobian (O(d²) complexity)."""
    x_flat, unravel = jax.flatten_util.ravel_pytree(x)

    def v_flat(x_flat_):
        x_unravelled = unravel(x_flat_)
        v = velocity_field(t, x_unravelled, args)
        v_flattened, _ = jax.flatten_util.ravel_pytree(v)
        return v_flattened

    y_flat = v_flat(x_flat)
    v = unravel(y_flat)
    jac = jax.jacfwd(v_flat)(x_flat)
    divergence = jnp.trace(jac)
    return v, divergence


def _divergence_hutchinson(velocity_field, t, x, args, random_vectors):
    """Compute stochastic divergence estimate using Hutchinson's trace estimator.

    Uses the identity: trace(J) = E[v^T J v] where v has E[v v^T] = I.
    Rademacher vectors (±1) are optimal for this estimator.

    Complexity: O(d * n_samples) instead of O(d²) for exact.

    Args:
        velocity_field: The velocity field callable
        t: Current time
        x: State pytree
        args: Additional arguments
        random_vectors: Array of shape (n_samples, d) with Rademacher vectors

    Returns:
        Tuple of (velocity, divergence_estimate)
    """
    x_flat, unravel = jax.flatten_util.ravel_pytree(x)

    def v_flat(x_flat_):
        x_unravelled = unravel(x_flat_)
        v = velocity_field(t, x_unravelled, args)
        v_flattened, _ = jax.flatten_util.ravel_pytree(v)
        return v_flattened

    # Compute velocity and JVP function
    y_flat, jvp_fn = jax.linearize(v_flat, x_flat)
    v = unravel(y_flat)

    # Estimate trace: trace(J) ≈ mean(v^T J v) = mean(v^T jvp(v))
    def estimate_single(rand_vec):
        jvp_result = jvp_fn(rand_vec)
        return jnp.dot(rand_vec, jvp_result)

    estimates = jax.vmap(estimate_single)(random_vectors)
    divergence = jnp.mean(estimates)
    return v, divergence


def _compute_velocity_and_divergence(velocity_field, t, x, args, random_vectors=None):
    """Compute velocity field and its divergence for arbitrary pytree inputs.

    Uses jax.jacfwd for exact divergence (when random_vectors is None) or
    Hutchinson's trace estimator for stochastic approximation.

    Args:
        velocity_field: Callable (t, x, args) -> velocity pytree with same structure as x
        t: Current time (scalar)
        x: State pytree (dynamic components only)
        args: Additional arguments passed to velocity_field
        random_vectors: Optional array of shape (n_samples, d) for Hutchinson estimator.
            If None, uses exact divergence computation.

    Returns:
        Tuple of (velocity, divergence) where velocity has same structure as x
        and divergence is a scalar.
    """
    if random_vectors is None:
        return _divergence_exact(velocity_field, t, x, args)
    return _divergence_hutchinson(velocity_field, t, x, args, random_vectors)


def _augmented_dynamics(t, y, args, velocity_field, random_vectors=None):
    """Augmented ODE dynamics for computing the change of variables formula.

    Solves the coupled system:
        dx/dt = v(t, x)
        d(log q)/dt = -div(v(t, x))

    where div(v) is the divergence of the velocity field.

    Args:
        t: Current time (scalar)
        y: Dictionary with keys "x" (state) and "logq" (log probability)
        args: Context pytree passed to velocity_field
        velocity_field: The velocity field callable
        random_vectors: Optional Rademacher vectors for Hutchinson estimator

    Returns:
        Dictionary with derivatives {"x": v, "logq": -div_v}
    """
    v, div_v = _compute_velocity_and_divergence(velocity_field, t, y["x"], args, random_vectors)
    return {"x": v, "logq": -div_v}


class Flow(eqx.Module, dsx.Distribution):
    """Continuous Normalizing Flow for generative modeling.

    A Flow transforms samples from a base distribution through a learned
    velocity field, enabling exact density evaluation via the instantaneous
    change of variables formula.

    The flow supports arbitrary pytree structures as inputs, making it suitable
    for particle systems with complex state structures (e.g., positions, species,
    box vectors). Use the `dynamic_mask` to specify which parts of the pytree
    should be transformed (dynamic) vs. kept constant (context).

    Attributes:
        velocity_field: Callable with signature (t, x, args) -> velocity pytree.
            The velocity field defining the flow dynamics. Must return a pytree
            with the same structure as the dynamic part of x.
        base_distribution: A distrax.Distribution representing the base/prior.
        dynamic_mask: A pytree or callable that returns True for leaves to flow,
            False for static context. Defaults to flowing all inexact arrays.
        hutchinson_samples: Number of random vectors for stochastic divergence
            estimation. If None (default), uses exact O(d²) computation.
            Set to a positive integer (e.g., 1-10) for O(d) stochastic estimation.
        solver: Diffrax solver for forward/inverse integration (default: Tsit5).
        augmented_solver: Solver for augmented ODE with log-prob (default: Tsit5).
        t0: Start time of the flow (default: 0.0).
        t1: End time of the flow (default: 1.0).
        dt0: Initial step size (optional, None for adaptive).
        stepsize_controller: Controller for adaptive stepping (default: PIDController).
        augmented_stepsize_controller: Controller for augmented ODE.
        extra_args: Additional kwargs passed to diffrax.diffeqsolve.
        augmented_extra_args: Additional kwargs for augmented ODE solve.

    """

    velocity_field: Callable
    base_distribution: dsx.Distribution
    dynamic_mask: Callable = eqx.field(
        default=lambda x: jax.tree.map(eqx.is_inexact_array, x),
        static=True,
    )
    hutchinson_samples: Optional[int] = eqx.field(default=None, static=True)
    solver: dfx.AbstractSolver = eqx.field(
        default_factory=lambda: dfx.Tsit5(),
        static=True,
    )
    augmented_solver: dfx.AbstractSolver = eqx.field(
        default_factory=lambda: dfx.Tsit5(),
        static=True,
    )
    t0: float = 0.0
    t1: float = 1.0
    dt0: Optional[float] = None
    stepsize_controller: dfx.AbstractStepSizeController = eqx.field(
        default_factory=lambda: dfx.PIDController(rtol=1e-5, atol=1e-5),
        static=True,
    )
    augmented_stepsize_controller: dfx.AbstractStepSizeController = eqx.field(
        default_factory=lambda: dfx.PIDController(rtol=1e-5, atol=1e-5),
        static=True,
    )
    extra_args: Dict[str, Any] = eqx.field(default_factory=dict, static=True)
    augmented_extra_args: Dict[str, Any] = eqx.field(default_factory=dict, static=True)

    @property
    def event_shape(self):
        return self.base_distribution.event_shape

    def _sample_n(self, key, n):
        x0 = self.base_distribution.sample(seed=key, sample_shape=(n,))
        x1 = jax.vmap(self.apply_map)(x0)
        return x1

    def _sample_n_and_log_prob(self, key, n):
        if self.hutchinson_samples is not None:
            # Need separate keys for sampling and for Hutchinson estimation
            key1, key2 = jax.random.split(key)
            x0 = self.base_distribution.sample(seed=key1, sample_shape=(n,))
            keys = jax.random.split(key2, n)
            x1, logq1 = jax.vmap(lambda x, k: self.apply_map_and_log_prob(x, key=k))(x0, keys)
        else:
            x0 = self.base_distribution.sample(seed=key, sample_shape=(n,))
            x1, logq1 = jax.vmap(self.apply_map_and_log_prob)(x0)
        return x1, logq1

    def _merge_solution(self, ys, ctx):
        if ys is None:
            return None
        leaves = jax.tree.leaves(ys)
        if not leaves:
            return ys
        T = leaves[0].shape[0]

        return jax.tree.map(
            lambda y, c: y if y is not None else jnp.broadcast_to(c, (T,) + c.shape),
            ys,
            ctx,
            is_leaf=lambda x: x is None,
        )

    @eqx.filter_jit
    def integrate(self, x0, **kwargs):
        """Integrate the ODE from t0 to t1 (or as specified in kwargs).

        Args:
            x0: Initial state (pytree matching base_distribution.event_shape).
            **kwargs: Override solver parameters. Common options:
                - t0, t1: Override integration bounds
                - dt0: Override initial step size
                - saveat: diffrax.SaveAt for saving intermediate states
                - args: User arguments passed to velocity_field

        Returns:
            diffrax solution object with .ys containing the trajectory.
        """
        solver_args = dict(
            solver=self.solver,
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            stepsize_controller=self.stepsize_controller,
            **self.extra_args,
        )
        solver_args.update(kwargs)
        if solver_args["dt0"] is not None:
            solver_args["dt0"] = jnp.sign(solver_args["t1"] - solver_args["t0"]) * abs(solver_args["dt0"])

        y0, ctx = eqx.partition(x0, self.dynamic_mask)

        user_args = solver_args.get("args")
        if user_args is not None:
            solver_args["args"] = (ctx, user_args)
        else:
            solver_args["args"] = ctx

        term = dfx.ODETerm(self.velocity_field)
        sol = dfx.diffeqsolve(term, y0=y0, **solver_args)
        return eqx.tree_at(lambda s: s.ys, sol, self._merge_solution(sol.ys, ctx))

    @eqx.filter_jit
    def apply_map(self, x0, **kwargs):
        """Apply the forward flow transformation x0 -> x1.

        Transforms a sample from the base distribution (at t0) to the
        target distribution (at t1).

        Args:
            x0: Initial state at t0.
            **kwargs: Override solver parameters (t0, t1, dt0, etc.).

        Returns:
            Transformed state x1 at t1.
        """
        saveat = kwargs.pop("saveat", dfx.SaveAt(t1=True))
        sol = self.integrate(x0, saveat=saveat, **kwargs)
        return jax.tree.map(lambda y: y[-1], sol.ys)

    @eqx.filter_jit
    def apply_inverse_map(self, x1, **kwargs):
        """Apply the inverse flow transformation x1 -> x0.

        Transforms a sample from the target distribution (at t1) back to
        the base distribution (at t0).

        Args:
            x1: State at t1.
            **kwargs: Override solver parameters.

        Returns:
            Reconstructed state x0 at t0.
        """
        t0 = kwargs.pop("t0", self.t0)
        t1 = kwargs.pop("t1", self.t1)
        saveat = kwargs.pop("saveat", dfx.SaveAt(t1=True))
        sol = self.integrate(x1, t0=t1, t1=t0, saveat=saveat, **kwargs)
        return jax.tree.map(lambda y: y[-1], sol.ys)

    @eqx.filter_jit
    def integrate_augmented_ode(self, x0, logq0=None, *, key=None, **kwargs):
        """Integrate the augmented ODE for log-probability computation.

        Args:
            x0: Initial state.
            logq0: Optional initial log probability. If None, computed from base_distribution.
            key: Optional PRNG key for Hutchinson estimator. Required if hutchinson_samples is set.
            **kwargs: Override solver parameters.

        Returns:
            diffrax solution object with .ys containing {"x": trajectory, "logq": log_probs}.
        """
        solver_args = dict(
            solver=self.augmented_solver,
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            stepsize_controller=self.augmented_stepsize_controller,
            **self.augmented_extra_args,
        )
        solver_args.update(kwargs)
        if solver_args["dt0"] is not None:
            solver_args["dt0"] = jnp.sign(solver_args["t1"] - solver_args["t0"]) * abs(solver_args["dt0"])

        if logq0 is None:
            logq0 = self.base_distribution.log_prob(x0)

        y0, ctx = eqx.partition(x0, self.dynamic_mask)
        u0 = {"x": y0, "logq": logq0}

        random_vectors = None
        if self.hutchinson_samples is not None:
            if key is None:
                raise ValueError("key is required when hutchinson_samples is set")
            y0_flat, _ = jax.flatten_util.ravel_pytree(y0)
            d = y0_flat.size
            random_vectors = jax.random.rademacher(key, shape=(self.hutchinson_samples, d)).astype(y0_flat.dtype)

        user_args = solver_args.get("args")
        if user_args is not None:
            solver_args["args"] = (ctx, user_args)
        else:
            solver_args["args"] = ctx

        term_func = jax.tree_util.Partial(
            _augmented_dynamics, velocity_field=self.velocity_field, random_vectors=random_vectors
        )

        term = dfx.ODETerm(term_func)
        sol = dfx.diffeqsolve(term, y0=u0, **solver_args)
        return eqx.tree_at(lambda s: s.ys["x"], sol, self._merge_solution(sol.ys["x"], ctx))

    @eqx.filter_jit
    def apply_map_and_log_prob(self, x0, *, key=None, **kwargs):
        """Apply forward flow and compute log probability simultaneously.

        More efficient than calling apply_map and log_prob separately
        when you need both the transformed sample and its log probability.

        Args:
            x0: Initial state sampled from base_distribution.
            key: PRNG key for Hutchinson estimator (required if hutchinson_samples is set).
            **kwargs: Override solver parameters.

        Returns:
            Tuple of (x1, log_prob) where x1 is the transformed state
            and log_prob is the log probability density at x1.
        """
        saveat = kwargs.pop("saveat", dfx.SaveAt(t1=True))
        sol = self.integrate_augmented_ode(x0, saveat=saveat, key=key, **kwargs)
        x1 = jax.tree.map(lambda y: y[-1], sol.ys["x"])
        logq1 = sol.ys["logq"][-1]
        return x1, logq1

    @eqx.filter_jit
    def log_prob(self, x1, *, key=None, **kwargs):
        """Compute log probability density of samples under the flow.

        Inverts the flow to map x1 back to the base distribution and
        applies the change of variables formula to compute the exact
        log probability.

        Automatically handles batched inputs: if x1 has an extra leading
        dimension compared to event_shape, it will vmap over that dimension.

        Args:
            x1: Sample(s) from the target distribution.
            key: PRNG key for Hutchinson estimator (required if hutchinson_samples is set).
            **kwargs: Override solver parameters.

        Returns:
            Log probability density. Shape is () for single sample,
            (batch_size,) for batched input.
        """
        event_shape = self.event_shape

        def _log_prob(x, k):
            f = jnp.zeros(())
            t0 = kwargs.pop("t0", self.t0)
            t1 = kwargs.pop("t1", self.t1)
            saveat = kwargs.pop("saveat", dfx.SaveAt(t1=True))
            sol = self.integrate_augmented_ode(x, t0=t1, t1=t0, logq0=f, saveat=saveat, key=k, **kwargs)
            x0 = jax.tree.map(lambda y: y[-1], sol.ys["x"])
            f0 = sol.ys["logq"][-1]
            logq0 = self.base_distribution.log_prob(x0)
            return logq0 - f0

        def _has_batch_dimension(x1_leaves, shape_leaves):
            """Check if all array leaves have a consistent batch dimension."""
            if len(x1_leaves) == 0 or len(x1_leaves) != len(shape_leaves):
                return False, 0

            def is_shape_tuple(x):
                return isinstance(x, tuple) and all(isinstance(i, int) for i in x)

            batch_sizes = []
            for arr, shape in zip(x1_leaves, shape_leaves):
                if not is_shape_tuple(shape):
                    continue
                if arr.ndim == len(shape) + 1:
                    batch_sizes.append(arr.shape[0])
                elif arr.ndim == len(shape):
                    return False, 0
                else:
                    return False, 0

            if len(batch_sizes) == 0:
                return False, 0
            if len(set(batch_sizes)) == 1:
                return True, batch_sizes[0]
            return False, 0

        def is_shape_tuple(x):
            return isinstance(x, tuple) and all(isinstance(i, int) for i in x)

        leaves_x1 = jax.tree.leaves(x1)
        leaves_shape = jax.tree.leaves(event_shape, is_leaf=is_shape_tuple)

        has_batch, batch_size = _has_batch_dimension(leaves_x1, leaves_shape)
        if has_batch:
            # Split keys for batch if hutchinson_samples is set
            if key is not None:
                keys = jax.random.split(key, batch_size)
            else:
                keys = jnp.zeros((batch_size,), dtype=jnp.uint32)  # Placeholder, won't be used
                keys = None  # Actually set to None array for vmap
            if keys is not None:
                return jax.vmap(_log_prob)(x1, keys)
            else:
                return jax.vmap(lambda x: _log_prob(x, None))(x1)
        return _log_prob(x1, key)
