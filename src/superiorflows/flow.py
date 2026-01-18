from typing import Any, Callable, Dict, Optional

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["Flow"]


def _compute_velocity_and_divergence(velocity_field, t, x, args):
    x_flat, unravel = jax.flatten_util.ravel_pytree(x)

    def v_flat(x_flat_):
        x_unravelled = unravel(x_flat_)
        v = velocity_field(t, x_unravelled, args)
        v_flattened, _ = jax.flatten_util.ravel_pytree(v)
        return v_flattened

    y_flat, jvp_fun = jax.linearize(v_flat, x_flat)
    v = unravel(y_flat)
    identity = jnp.eye(x_flat.size)
    cols = jax.vmap(jvp_fun)(identity)
    divergence = jnp.trace(cols)
    return v, divergence


def _augmented_dynamics(t, y, args, velocity_field):
    v, div_v = _compute_velocity_and_divergence(velocity_field, t, y["x"], args)
    return {"x": v, "logq": -div_v}


class Flow(eqx.Module, dsx.Distribution):
    velocity_field: Callable
    base_distribution: dsx.Distribution
    dynamic_mask: Callable = eqx.field(
        default=lambda x: jax.tree.map(eqx.is_inexact_array, x),
        static=True,
    )
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
        saveat = kwargs.pop("saveat", dfx.SaveAt(t1=True))
        sol = self.integrate(x0, saveat=saveat, **kwargs)
        return jax.tree.map(lambda y: y[-1], sol.ys)

    @eqx.filter_jit
    def apply_inverse_map(self, x1, **kwargs):
        t0 = kwargs.pop("t0", self.t0)
        t1 = kwargs.pop("t1", self.t1)
        saveat = kwargs.pop("saveat", dfx.SaveAt(t1=True))
        sol = self.integrate(x1, t0=t1, t1=t0, saveat=saveat, **kwargs)
        return jax.tree.map(lambda y: y[-1], sol.ys)

    @eqx.filter_jit
    def integrate_augmented_ode(self, x0, logq0=None, **kwargs):
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

        user_args = solver_args.get("args")
        if user_args is not None:
            solver_args["args"] = (ctx, user_args)
        else:
            solver_args["args"] = ctx

        term_func = jax.tree_util.Partial(_augmented_dynamics, velocity_field=self.velocity_field)

        term = dfx.ODETerm(term_func)
        sol = dfx.diffeqsolve(term, y0=u0, **solver_args)
        return eqx.tree_at(lambda s: s.ys["x"], sol, self._merge_solution(sol.ys["x"], ctx))

    @eqx.filter_jit
    def apply_map_and_log_prob(self, x0, **kwargs):
        saveat = kwargs.pop("saveat", dfx.SaveAt(t1=True))
        sol = self.integrate_augmented_ode(x0, saveat=saveat, **kwargs)
        x1 = jax.tree.map(lambda y: y[-1], sol.ys["x"])
        logq1 = sol.ys["logq"][-1]
        return x1, logq1

    @eqx.filter_jit
    def log_prob(self, x1, **kwargs):
        event_shape = self.event_shape

        def _log_prob(x):
            f = jnp.zeros(())
            t0 = kwargs.pop("t0", self.t0)
            t1 = kwargs.pop("t1", self.t1)
            saveat = kwargs.pop("saveat", dfx.SaveAt(t1=True))
            sol = self.integrate_augmented_ode(x, t0=t1, t1=t0, logq0=f, saveat=saveat, **kwargs)
            x0 = jax.tree.map(lambda y: y[-1], sol.ys["x"])
            f0 = sol.ys["logq"][-1]
            logq0 = self.base_distribution.log_prob(x0)
            return logq0 - f0

        def is_shape_tuple(x):
            return isinstance(x, tuple) and all(isinstance(i, int) for i in x)

        leaves_x1 = jax.tree.leaves(x1)
        leaves_shape = jax.tree.leaves(event_shape, is_leaf=is_shape_tuple)

        has_batch_dim = False
        if len(leaves_x1) > 0 and len(leaves_x1) == len(leaves_shape):
            arr, shape = leaves_x1[0], leaves_shape[0]
            if isinstance(shape, tuple) and arr.ndim == len(shape) + 1:
                has_batch_dim = True

        if has_batch_dim:
            return jax.vmap(_log_prob)(x1)
        return _log_prob(x1)
