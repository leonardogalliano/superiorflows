"""ODE-based bijector for continuous normalising flows.

Implements an invertible transformation defined by the flow map of an
ordinary differential equation.  The velocity field ``v(t, x, args)``
generates the dynamics ``dx/dt = v``; integrating from ``t₀`` to ``t₁``
gives the forward map, and integrating backward gives the inverse.

The log absolute determinant of the Jacobian is computed via the
instantaneous change-of-variables formula using an augmented ODE that
simultaneously integrates ``d(log q)/dt = −div(v)``.

Three divergence computation strategies are supported:

- **Exact** (default): diagonal JVP extraction, O(d²) FLOPs.
- **Hutchinson**: stochastic trace estimator with Rademacher vectors,
  O(d · n_samples).  Activated by setting ``hutchinson_samples``.
- **Analytical**: user-supplied closed-form divergence.  Activated by
  setting ``divergence_fn``.
"""

from typing import Any, Callable, Dict, Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp

from superiorflows.bijector import AbstractBijector

__all__ = ["ODEBijector"]


def _divergence_exact(velocity_field, t, x, args):
    """Compute exact divergence via diagonal JVP extraction.

    Uses jax.linearize to obtain the JVP map, then extracts only the diagonal
    elements J_ii via unit-vector probes. This avoids materialising the full
    d×d Jacobian (O(d) memory instead of O(d²)), while the FLOP count remains
    O(d²) — each of the d probes costs O(d).
    """
    x_flat, unravel = jax.flatten_util.ravel_pytree(x)

    def v_flat(x_flat_):
        x_unravelled = unravel(x_flat_)
        v = velocity_field(t, x_unravelled, args)
        v_flattened, _ = jax.flatten_util.ravel_pytree(v)
        return v_flattened

    y_flat, jvp_fn = jax.linearize(v_flat, x_flat)
    v = unravel(y_flat)

    def diag_element(i):
        e = jnp.zeros_like(x_flat).at[i].set(1.0)
        return jvp_fn(e)[i]

    divergence = jnp.sum(jax.vmap(diag_element)(jnp.arange(x_flat.size)))
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

    y_flat, jvp_fn = jax.linearize(v_flat, x_flat)
    v = unravel(y_flat)

    def estimate_single(rand_vec):
        jvp_result = jvp_fn(rand_vec)
        return jnp.dot(rand_vec, jvp_result)

    estimates = jax.vmap(estimate_single)(random_vectors)
    divergence = jnp.mean(estimates)
    return v, divergence


def _augmented_dynamics(t, y, args, divergence_fn):
    """Augmented ODE dynamics for the change of variables formula.

    Solves the coupled system:
        dx/dt = v(t, x)
        d(log q)/dt = -div(v(t, x))

    Args:
        t: Current time (scalar)
        y: Dictionary ``{"x": state, "logq": log_det_accumulator}``
        args: Context pytree passed to velocity_field
        divergence_fn: Callable ``(t, x, args) → (velocity, divergence)``

    Returns:
        Dictionary ``{"x": v, "logq": -div_v}``
    """
    v, div_v = divergence_fn(t, y["x"], args)
    return {"x": v, "logq": -div_v}


class ODEBijector(AbstractBijector):
    """Bijector defined by an ODE flow map.

    Wraps a velocity field and ODE solver configuration into an invertible
    transformation with exact (or stochastic) log-det-Jacobian computation.

    The forward map integrates the ODE ``dx/dt = v(t, x)`` from ``t₀`` to
    ``t₁``; the inverse integrates backward from ``t₁`` to ``t₀``.

    Dynamic-mask partitioning is handled by the parent
    :class:`AbstractBijector` — this class operates exclusively on the
    pre-partitioned dynamic state.  Static context is received via
    ``args`` and forwarded to the velocity field.

    Attributes:
        velocity_field: Callable ``(t, x_dynamic, args) → velocity_pytree``.
        divergence_fn: Optional callable
            ``(velocity_field, t, x, args) → (v, div_v)`` providing an
            analytical divergence.  Mutually exclusive with
            ``hutchinson_samples``.
        hutchinson_samples: Number of Rademacher vectors for stochastic
            divergence estimation.  ``None`` (default) uses exact O(d²)
            computation.
        solver: Diffrax solver for plain forward/inverse integration.
        augmented_solver: Diffrax solver for the augmented ODE (log-det).
        t0: Start time of the flow (default: 0.0).
        t1: End time of the flow (default: 1.0).
        dt0: Initial step size (``None`` for adaptive).
        stepsize_controller: Controller for plain integration.
        augmented_stepsize_controller: Controller for augmented integration.
        extra_args: Additional kwargs passed to ``diffrax.diffeqsolve``
            for plain integration.
        augmented_extra_args: Additional kwargs for augmented integration.
    """

    velocity_field: Callable
    divergence_fn: Optional[Callable] = eqx.field(default=None, static=True)
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

    def __check_init__(self):
        if self.divergence_fn is not None and self.hutchinson_samples is not None:
            raise ValueError("Cannot set both divergence_fn and hutchinson_samples. Choose one divergence strategy.")

    # --- AbstractBijector extension points --------------------------------
    #
    # All methods receive pre-partitioned dynamic state from the parent.
    # Context is in ``args``; no partition/merge logic here.

    def _forward(self, x, *, args=None, **kwargs):
        kw = dict(kwargs)
        saveat = kw.pop("saveat", dfx.SaveAt(t1=True))
        sol = self._integrate(x, args=args, saveat=saveat, **kw)
        return jax.tree.map(lambda y: y[-1], sol.ys)

    def _inverse(self, y, *, args=None, **kwargs):
        kw = dict(kwargs)
        t0 = kw.pop("t0", self.t0)
        t1 = kw.pop("t1", self.t1)
        saveat = kw.pop("saveat", dfx.SaveAt(t1=True))
        sol = self._integrate(y, args=args, t0=t1, t1=t0, saveat=saveat, **kw)
        return jax.tree.map(lambda y: y[-1], sol.ys)

    def _forward_and_log_det(self, x, *, args=None, **kwargs):
        kw = dict(kwargs)
        key = kw.pop("key", None)
        saveat = kw.pop("saveat", dfx.SaveAt(t1=True))
        logq0 = jnp.zeros(())
        sol = self._integrate_augmented(x, logq0, key=key, args=args, saveat=saveat, **kw)
        y = jax.tree.map(lambda arr: arr[-1], sol.ys["x"])
        logq1 = sol.ys["logq"][-1]
        return y, -logq1

    def _inverse_and_log_det(self, y, *, args=None, **kwargs):
        kw = dict(kwargs)
        key = kw.pop("key", None)
        t0 = kw.pop("t0", self.t0)
        t1 = kw.pop("t1", self.t1)
        saveat = kw.pop("saveat", dfx.SaveAt(t1=True))
        logq0 = jnp.zeros(())
        sol = self._integrate_augmented(y, logq0, key=key, args=args, t0=t1, t1=t0, saveat=saveat, **kw)
        x = jax.tree.map(lambda arr: arr[-1], sol.ys["x"])
        f0 = sol.ys["logq"][-1]
        return x, -f0

    # --- ODE integration backend -----------------------------------------

    def _integrate(self, x0, *, args=None, **kwargs):
        """Integrate the ODE on pre-partitioned dynamic state.

        Args:
            x0: Dynamic state (static leaves are ``None``).
            args: Context from partition, forwarded to velocity field.
            **kwargs: Override solver parameters (``t0``, ``t1``, ``dt0``,
                ``saveat``).

        Returns:
            Diffrax solution object with ``.ys`` containing the dynamic
            trajectory.
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
        solver_args["args"] = args

        term = dfx.ODETerm(self.velocity_field)
        return dfx.diffeqsolve(term, y0=x0, **solver_args)

    def _integrate_augmented(self, x0, logq0, *, key=None, args=None, **kwargs):
        """Integrate the augmented ODE on pre-partitioned dynamic state.

        Args:
            x0: Dynamic state (static leaves are ``None``).
            logq0: Initial value of the log-det accumulator (scalar).
            key: PRNG key for Hutchinson estimator.
            args: Context from partition, forwarded to velocity field.
            **kwargs: Override solver parameters.

        Returns:
            Diffrax solution with ``.ys`` containing
            ``{"x": dynamic_trajectory, "logq": accumulated_logdet}``.
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

        u0 = {"x": x0, "logq": logq0}

        random_vectors = None
        if self.hutchinson_samples is not None:
            if key is None:
                raise ValueError("key is required when hutchinson_samples is set")
            y0_flat, _ = jax.flatten_util.ravel_pytree(x0)
            d = y0_flat.size
            random_vectors = jax.random.rademacher(key, shape=(self.hutchinson_samples, d)).astype(y0_flat.dtype)

        solver_args["args"] = args

        div_fn = self._make_divergence_fn(random_vectors=random_vectors)
        term_func = jax.tree_util.Partial(_augmented_dynamics, divergence_fn=div_fn)
        term = dfx.ODETerm(term_func)
        return dfx.diffeqsolve(term, y0=u0, **solver_args)

    # --- Private helpers --------------------------------------------------

    def _make_divergence_fn(self, random_vectors=None):
        """Resolve the divergence strategy into a single callable.

        Returns a callable ``(t, x, args) → (velocity, divergence)``
        that is fully bound and ready to be passed to ``_augmented_dynamics``.
        """
        if self.divergence_fn is not None:
            vf = self.velocity_field

            def analytical(t, x, args):
                return self.divergence_fn(vf, t, x, args)

            return analytical

        if self.hutchinson_samples is not None:
            vf = self.velocity_field
            rv = random_vectors

            def hutchinson(t, x, args):
                return _divergence_hutchinson(vf, t, x, args, rv)

            return hutchinson

        vf = self.velocity_field

        def exact(t, x, args):
            return _divergence_exact(vf, t, x, args)

        return exact
