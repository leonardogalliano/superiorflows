"""SDE-based transport with Girsanov path action.

Implements stochastic transport via Euler–Maruyama discretisation of the
forward SDE:

    dX = [b̂(t,X) + ε(t)ŝ(t,X)] dt + √(2ε(t)) dW

where b̂ is the velocity field, ŝ is the score field, and ε is the noise
schedule.  The path action (Girsanov/Crooks reweighting factor) is
accumulated in-step via a custom diffrax solver.

Key components:

- :class:`SDETerm`: custom ``AbstractTerm`` wrapping drift + Brownian
  motion for both plain and augmented states.
- :class:`EulerMaruyama`: custom ``AbstractItoSolver`` for plain SDE
  integration.
- :class:`EulerMaruyamaWithAction`: custom ``AbstractItoSolver`` that
  simultaneously integrates the SDE and accumulates the path action
  ``Σ(log p⁺ − log p⁻)`` step by step.
- :class:`SDETransport`: concrete ``AbstractTransport`` subclass
  composing these pieces.
"""

from typing import Any, Callable, ClassVar, Dict

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from superiorflows.transport import AbstractTransport

__all__ = ["SDETerm", "EulerMaruyama", "EulerMaruyamaWithAction", "SDETransport"]


class SDETerm(dfx.AbstractTerm):
    """Combined SDE term operating on plain or augmented state.

    Wraps drift + Brownian motion into a single ``AbstractTerm`` with
    elementwise (diagonal) noise over arbitrary PyTrees.
    """

    forward_drift_fn: Callable
    noise_coefficient_fn: Callable
    brownian_motion: dfx.AbstractBrownianPath

    def vf(self, t, y, args):
        x = y["x"] if isinstance(y, dict) and "x" in y else y
        return self.forward_drift_fn(t, x, args)

    def contr(self, t0, t1, **kwargs):
        dt = t1 - t0
        dW = self.brownian_motion.evaluate(t0, t1)
        return (dt, dW)

    def prod(self, vf, control):
        dt, _ = control
        return jax.tree.map(lambda v: v * dt, vf)

    def vf_prod(self, t, y, args, control):
        dt, dW = control
        x = y["x"] if isinstance(y, dict) and "x" in y else y
        mu = self.forward_drift_fn(t, x, args)
        sigma = self.noise_coefficient_fn(t, x, args)
        return jax.tree.map(lambda m, s, dw: m * dt + s * dw, mu, sigma, dW)


class EulerMaruyama(dfx.AbstractItoSolver):
    """Euler–Maruyama solver operating on SDETerm."""

    term_structure: ClassVar = SDETerm
    interpolation_cls: ClassVar[type] = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        dt = t1 - t0
        _, dW = terms.contr(t0, t1)
        dy = terms.vf_prod(t0, y0, args, (dt, dW))
        y1 = jax.tree.map(lambda y, dy_: y + dy_, y0, dy)
        return y1, None, dict(y0=y0, y1=y1), None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class EulerMaruyamaWithAction(dfx.AbstractItoSolver):
    """Euler–Maruyama solver with in-step Girsanov path action accumulation.

    Operates on augmented state ``{"x": pytree, "action": scalar}``.
    Each step computes:

    1. Standard EM update: ``x₁ = x₀ + μ⁺(t₀,x₀)·dt + σ(t₀)·dW``
    2. Path action increment: ``log p⁺(x₁|x₀) − log p⁻(x₀|x₁)``

    The forward and backward transition kernels are isotropic Gaussians:

    - ``p⁺(x₁|x₀) = N(x₀ + μ⁺·dt, 2ε(t₀)·dt·I)``
    - ``p⁻(x₀|x₁) = N(x₁ − μ⁻·dt, 2ε(t₁)·dt·I)``

    The path action is accumulated into ``y["action"]`` as a running sum.
    Residual norms are computed leaf-by-leaf over the PyTree (zero-copy).
    """

    term_structure: ClassVar = SDETerm
    interpolation_cls: ClassVar[type] = dfx.LocalLinearInterpolation

    forward_drift_fn: Callable
    backward_drift_fn: Callable
    noise_schedule: Callable

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        x0 = y0["x"]
        action0 = y0["action"]

        dt = t1 - t0
        _, dW = terms.contr(t0, t1)

        eps_t0 = self.noise_schedule(t0)
        sigma_t0 = jnp.sqrt(2.0 * eps_t0)

        mu_fwd_x0 = self.forward_drift_fn(t0, x0, args)
        x1 = jax.tree.map(lambda x, m, dw: x + m * dt + sigma_t0 * dw, x0, mu_fwd_x0, dW)

        eps_t1 = self.noise_schedule(t1)
        sigma2_fwd = 2.0 * eps_t0 * dt
        sigma2_bwd = 2.0 * eps_t1 * dt
        d = sum(leaf.size for leaf in jax.tree.leaves(x0))

        res_fwd_sq = sum(
            jnp.sum(jnp.square(x1_l - x0_l - m_l * dt))
            for x1_l, x0_l, m_l in zip(jax.tree.leaves(x1), jax.tree.leaves(x0), jax.tree.leaves(mu_fwd_x0))
        )
        log_p_fwd = -0.5 * res_fwd_sq / sigma2_fwd - 0.5 * d * jnp.log(2.0 * jnp.pi * sigma2_fwd)

        mu_bwd_x1 = self.backward_drift_fn(t1, x1, args)
        res_bwd_sq = sum(
            jnp.sum(jnp.square(x0_l - (x1_l - m_b_l * dt)))
            for x0_l, x1_l, m_b_l in zip(jax.tree.leaves(x0), jax.tree.leaves(x1), jax.tree.leaves(mu_bwd_x1))
        )
        log_p_bwd = -0.5 * res_bwd_sq / sigma2_bwd - 0.5 * d * jnp.log(2.0 * jnp.pi * sigma2_bwd)

        action1 = action0 + (log_p_fwd - log_p_bwd)
        y1 = {"x": x1, "action": action1}
        return y1, None, dict(y0=y0, y1=y1), None, dfx.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class SDETransport(AbstractTransport):
    """Stochastic transport via Euler–Maruyama SDE integration.

    Integrates the forward SDE::

        dX = [b̂(t,X) + ε(t)ŝ(t,X)] dt + √(2ε(t)) dW

    ``velocity_field`` and ``score_field`` are arbitrary callables with
    signature ``(t, x, args)`` — they can be learned models or analytical
    oracles for testing.  ``noise_schedule`` is an arbitrary callable
    ``ε(t) → ℝ≥0``.

    Configuration mirrors :class:`ODEBijector`: module-level defaults for
    solver parameters, per-call kwargs overrides for ``saveat``/``t0``/
    ``t1``/``num_steps``/``key``, and ``extra_args`` dicts forwarded to
    ``diffrax.diffeqsolve``.

    Attributes:
        velocity_field: Callable ``(t, x_dynamic, args) → velocity_pytree``.
        score_field: Callable ``(t, x_dynamic, args) → score_pytree``.
        noise_schedule: Callable ``t → ε ≥ 0``.
        t0: Start time of the SDE (default: 0.0).
        t1: End time of the SDE (default: 1.0).
        num_steps: Number of Euler–Maruyama steps M (default: 100).
            Determines ``dt = (t1 − t0) / M``.  Overridable per call.
        extra_args: Additional kwargs passed to ``diffrax.diffeqsolve``
            for plain forward integration (no path action).
        action_extra_args: Additional kwargs passed to ``diffrax.diffeqsolve``
            for forward + path action integration.
    """

    velocity_field: Callable
    score_field: Callable
    noise_schedule: Callable
    t0: float = 0.0
    t1: float = 1.0
    num_steps: int = 100
    extra_args: Dict[str, Any] = eqx.field(default_factory=dict, static=True)
    action_extra_args: Dict[str, Any] = eqx.field(default_factory=dict, static=True)

    # --- AbstractTransport extension points ---

    def _forward(self, x, *, args=None, key, **kwargs):
        kw = dict(kwargs)
        saveat = kw.pop("saveat", dfx.SaveAt(t1=True))
        sol = self._integrate(x, args=args, key=key, saveat=saveat, **kw)
        return jax.tree.map(lambda y: y[-1], sol.ys)

    def _forward_and_log_weight(self, x, *, args=None, key, **kwargs):
        kw = dict(kwargs)
        saveat = kw.pop("saveat", dfx.SaveAt(t1=True))
        sol = self._integrate_with_action(x, args=args, key=key, saveat=saveat, **kw)
        x1 = jax.tree.map(lambda arr: arr[-1], sol.ys["x"])
        action = sol.ys["action"][-1]
        return x1, action

    # --- SDE-native alias ---

    def forward_and_path_action(self, x, **kwargs):
        """SDE-native alias for :meth:`forward_and_log_weight`."""
        return self.forward_and_log_weight(x, **kwargs)

    # --- Internal helpers ---

    def _make_drifts(self):
        """Build forward and backward drift closures from velocity + score + noise.

        Returns:
            Tuple ``(forward_drift, backward_drift)`` with signature
            ``(t, x, args) → pytree``.
        """

        def forward_drift(t, x, _args):
            b = self.velocity_field(t, x, _args)
            s = self.score_field(t, x, _args)
            eps = self.noise_schedule(t)
            return jax.tree.map(lambda bi, si: bi + eps * si, b, s)

        def backward_drift(t, x, _args):
            b = self.velocity_field(t, x, _args)
            s = self.score_field(t, x, _args)
            eps = self.noise_schedule(t)
            return jax.tree.map(lambda bi, si: bi - eps * si, b, s)

        return forward_drift, backward_drift

    def _make_brownian_tree(self, x0, key, t0, t1, dt0):
        """Create a VirtualBrownianTree matching the PyTree structure of ``x0``."""
        shape = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), x0)
        return dfx.VirtualBrownianTree(t0=t0, t1=t1, tol=dt0 / 2, shape=shape, key=key)

    def _noise_coefficient(self, t, x, _args):
        """Isotropic diffusion coefficient ``σ(t) = √(2ε(t))``."""
        sigma = jnp.sqrt(2.0 * self.noise_schedule(t))
        return jax.tree.map(lambda xi: sigma * jnp.ones_like(xi), x)

    # --- SDE integration backend ---

    def _integrate(self, x0, *, args=None, key, **kwargs):
        """Plain EM integration on pre-partitioned dynamic state.

        Args:
            x0: Dynamic state.
            args: Context from partition, forwarded to velocity/score.
            key: PRNG key for Brownian motion.
            **kwargs: Override parameters (``t0``, ``t1``, ``num_steps``,
                ``saveat``).

        Returns:
            Diffrax ``Solution`` object.
        """
        t0 = kwargs.pop("t0", self.t0)
        t1 = kwargs.pop("t1", self.t1)
        num_steps = kwargs.pop("num_steps", self.num_steps)
        dt0 = (t1 - t0) / num_steps

        fwd_drift, _ = self._make_drifts()
        bm = self._make_brownian_tree(x0, key, t0, t1, dt0)

        term = SDETerm(
            forward_drift_fn=fwd_drift,
            noise_coefficient_fn=self._noise_coefficient,
            brownian_motion=bm,
        )

        solver = EulerMaruyama()

        solver_args = dict(
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            stepsize_controller=dfx.ConstantStepSize(),
            args=args,
            **self.extra_args,
        )
        solver_args.update(kwargs)

        return dfx.diffeqsolve(term, y0=x0, **solver_args)

    def _integrate_with_action(self, x0, *, args=None, key, **kwargs):
        """EM integration with path action on pre-partitioned dynamic state.

        Args:
            x0: Dynamic state.
            args: Context from partition, forwarded to velocity/score.
            key: PRNG key for Brownian motion.
            **kwargs: Override parameters (``t0``, ``t1``, ``num_steps``,
                ``saveat``).

        Returns:
            Diffrax ``Solution`` with ``.ys = {"x": ..., "action": ...}``.
        """
        t0 = kwargs.pop("t0", self.t0)
        t1 = kwargs.pop("t1", self.t1)
        num_steps = kwargs.pop("num_steps", self.num_steps)
        dt0 = (t1 - t0) / num_steps

        fwd_drift, bwd_drift = self._make_drifts()
        bm = self._make_brownian_tree(x0, key, t0, t1, dt0)

        term = SDETerm(
            forward_drift_fn=fwd_drift,
            noise_coefficient_fn=self._noise_coefficient,
            brownian_motion=bm,
        )

        solver = EulerMaruyamaWithAction(
            forward_drift_fn=fwd_drift,
            backward_drift_fn=bwd_drift,
            noise_schedule=self.noise_schedule,
        )

        u0 = {"x": x0, "action": jnp.zeros(())}

        solver_args = dict(
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            stepsize_controller=dfx.ConstantStepSize(),
            args=args,
            **self.action_extra_args,
        )
        solver_args.update(kwargs)

        return dfx.diffeqsolve(term, y0=u0, **solver_args)
