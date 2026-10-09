"""SDE-based transport with in-step Girsanov path action.

Implements stochastic continuous transport via a native JAX (``jax.lax.scan``)
Euler–Maruyama discretisation of the forward and time-reversed SDE:

    dX = [b̂(t,X) + u(t,X)] dt + √(2ε(t)) dW

where b̂ is the velocity field, u is the osmotic drift (either supplied directly
or computed as ε(t)ŝ(t,X) from a score field ŝ), and ε is the diffusion noise
schedule.

The path action (Girsanov/Crooks path reweighting factor) is accumulated in-step
during the numerical integration:

    S_M = Σ_{k=0}^{M-1} (log p⁺(X_{k+1}|X_k) − log p⁻(X_k|X_{k+1}))

Key components:
- :class:`SDESolution`: lightweight container matching Diffrax solution structures.
- :class:`SDETransport`: concrete :class:`AbstractTransport` subclass providing
  forward, backward, and importance sampling path action evaluations.
"""

from typing import Any, Callable, Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from superiorflows.transport import AbstractTransport

__all__ = ["SDESolution", "SDETransport"]


class SDESolution(eqx.Module):
    """Lightweight solution container providing the ``.ys`` attribute.

    Attributes:
        ys: PyTree of integrated states (and path action when evaluated).
    """

    ys: Any


class SDETransport(AbstractTransport):
    """Stochastic transport via native JAX Euler–Maruyama integration.

    Integrates the forward SDE:

        dX = [b̂(t,X) + u(t,X)] dt + √(2ε(t)) dW

    where ``u(t, X)`` is the osmotic drift. The osmotic drift can either
    be provided directly via ``osmotic_drift``, or computed from a score
    field via ``u(t, X) = ε(t) ŝ(t, X)`` when ``score_field`` is supplied.

    ``velocity_field``, ``score_field``, and ``osmotic_drift`` are arbitrary
    callables with signature ``(t, x_dynamic, args)``. ``noise_schedule`` is an
    arbitrary callable ``ε(t) → ℝ≥0``.

    The integration engine compiles to a single ``jax.lax.scan``. Brownian
    increments are drawn inside the loop from per-step keys
    (``jax.random.split(key, num_steps)``), so memory does not grow with the
    number of steps beyond the keys themselves. With the path action, the
    fields ``(b, u)`` at ``(t_{k+1}, X_{k+1})`` serve both the backward kernel of
    step ``k`` and the forward kernel of step ``k + 1``, so they are evaluated
    ``num_steps + 1`` times in total rather than ``2 * num_steps``.

    Attributes:
        velocity_field: Callable ``(t, x_dynamic, args) → velocity_pytree``.
        noise_schedule: Callable ``t → ε ≥ 0``.
        score_field: Optional callable ``(t, x_dynamic, args) → score_pytree``.
        osmotic_drift: Optional callable
            ``(t, x_dynamic, args) → osmotic_drift_pytree`` evaluating
            ``u(t, x) = ε(t) s(t, x)`` directly without division.
        t0: Start time of the SDE (default: 0.0).
        t1: End time of the SDE (default: 1.0).
        num_steps: Number of Euler–Maruyama steps M (default: 100).
            Determines ``dt = (t1 − t0) / M``. Overridable per call.
        extra_args: Unused compatibility dict for solver kwargs.
        action_extra_args: Unused compatibility dict for solver kwargs.
    """

    velocity_field: Callable
    noise_schedule: Callable
    score_field: Optional[Callable] = None
    osmotic_drift: Optional[Callable] = None
    t0: float = 0.0
    t1: float = 1.0
    num_steps: int = 100
    extra_args: Dict[str, Any] = eqx.field(default_factory=dict, static=True)
    action_extra_args: Dict[str, Any] = eqx.field(default_factory=dict, static=True)

    def __post_init__(self):
        if self.score_field is None and self.osmotic_drift is None:
            raise ValueError("Either `score_field` or `osmotic_drift` must be provided to SDETransport.")

    # --- AbstractTransport extension points ---

    def _forward(self, x, *, args=None, key, **kwargs):
        sol = self._integrate(x, args=args, key=key, **kwargs)
        return jax.tree.map(lambda y: y[-1], sol.ys)

    def _forward_and_log_weight(self, x, *, args=None, key, **kwargs):
        sol = self._integrate_with_action(x, args=args, key=key, **kwargs)
        x1 = jax.tree.map(lambda arr: arr[-1], sol.ys["x"])
        action = sol.ys["action"][-1]
        return x1, action

    def forward_and_path_action(self, x, **kwargs):
        """SDE-native alias for :meth:`forward_and_log_weight`."""
        return self.forward_and_log_weight(x, **kwargs)

    def _backward(self, y, *, args=None, key, **kwargs):
        kw = dict(kwargs)
        t0 = kw.pop("t0", self.t1)
        t1 = kw.pop("t1", self.t0)
        sol = self._integrate(y, args=args, key=key, t0=t0, t1=t1, backward=True, **kw)
        return jax.tree.map(lambda arr: arr[-1], sol.ys)

    def _backward_and_log_weight(self, y, *, args=None, key, **kwargs):
        kw = dict(kwargs)
        t0 = kw.pop("t0", self.t1)
        t1 = kw.pop("t1", self.t0)
        sol = self._integrate_with_action(y, args=args, key=key, t0=t0, t1=t1, backward=True, **kw)
        x0 = jax.tree.map(lambda arr: arr[-1], sol.ys["x"])
        action = sol.ys["action"][-1]
        return x0, action

    def backward_and_path_action(self, y, **kwargs):
        """SDE-native alias for :meth:`backward_and_log_weight`."""
        return self.backward_and_log_weight(y, **kwargs)

    # --- Internal field evaluation ---

    def _fields(self, t, x, args):
        """Evaluate the velocity and the osmotic drift at a single point.

        Returns:
            Tuple ``(b, u)`` of PyTrees matching ``x``: ``b = b̂(t, x)`` and
            ``u = u(t, x)`` (given directly, or ``ε(t) ŝ(t, x)``). The forward
            drift is ``b + u`` and the backward drift ``b − u``.
        """
        b = self.velocity_field(t, x, args)
        if self.osmotic_drift is not None:
            return b, self.osmotic_drift(t, x, args)
        eps = self.noise_schedule(t)
        s = self.score_field(t, x, args)
        return b, jax.tree.map(lambda si: eps * si, s)

    @staticmethod
    def _standard_normal_like(key, x):
        """Draw independent standard normals with the structure, shapes and dtypes of ``x``."""
        leaves, treedef = jax.tree.flatten(x)
        if not leaves:
            return x
        keys = jax.random.split(key, len(leaves))
        return jax.tree.unflatten(
            treedef, [jax.random.normal(k, shape=leaf.shape, dtype=leaf.dtype) for k, leaf in zip(keys, leaves)]
        )

    # --- SDE integration core routines ---

    def _integrate(self, x0, *, args=None, key, **kwargs):
        """Plain Euler–Maruyama integration on pre-partitioned dynamic state.

        Args:
            x0: Dynamic state PyTree.
            args: Context from partition, forwarded to velocity/score.
            key: PRNG key for Brownian motion increments.
            **kwargs: Override parameters (``t0``, ``t1``, ``num_steps``, ``backward``).

        Returns:
            :class:`SDESolution` whose ``.ys`` holds the final state with a leading axis of length 1.
        """
        kw = dict(kwargs)
        kw.pop("saveat", None)
        t0 = kw.pop("t0", self.t0)
        t1 = kw.pop("t1", self.t1)
        num_steps = kw.pop("num_steps", self.num_steps)
        backward = kw.pop("backward", False)

        # Integrating backward in time follows the backward drift b − u.
        sign = -1.0 if backward else 1.0
        dt = (t1 - t0) / num_steps
        sqrt_dt = jnp.sqrt(jnp.abs(dt))

        def time_at(step_idx):
            return jnp.abs(t0 + step_idx * dt)

        def step_fn(x_curr, inputs):
            step_key, step_idx = inputs
            t_curr = time_at(step_idx)
            sigma_curr = jnp.sqrt(2.0 * self.noise_schedule(t_curr))

            b, u = self._fields(t_curr, x_curr, args)
            dw = self._standard_normal_like(step_key, x_curr)
            x_next = jax.tree.map(
                lambda x, bi, ui, w: x + (bi + sign * ui) * dt + sigma_curr * sqrt_dt * w,
                x_curr,
                b,
                u,
                dw,
            )
            return x_next, None

        step_keys = jax.random.split(key, num_steps)
        x_final, _ = jax.lax.scan(step_fn, x0, (step_keys, jnp.arange(num_steps)))
        ys = jax.tree.map(lambda leaf: jnp.expand_dims(leaf, 0), x_final)
        return SDESolution(ys=ys)

    def _integrate_with_action(self, x0, *, args=None, key, **kwargs):
        """Euler–Maruyama integration with in-step Girsanov path action accumulation.

        Step ``k`` draws ``X_{k+1}`` from the forward kernel
        ``p⁺(·|X_k) = N(X_k + μ_gen(t_k, X_k) Δt, 2ε(t_k)|Δt|)`` and adds
        ``log p⁺(X_{k+1}|X_k) − log p⁻(X_k|X_{k+1})`` to the action, where
        ``p⁻(·|X_{k+1}) = N(X_{k+1} − μ_ghost(t_{k+1}, X_{k+1}) Δt, 2ε(t_{k+1})|Δt|)``.
        Forward in time ``μ_gen = b + u`` and ``μ_ghost = b − u``; backward in time
        the roles are swapped. The fields at ``(t_{k+1}, X_{k+1})`` are carried
        to step ``k + 1``, whose forward kernel needs them at the same point.

        Args:
            x0: Dynamic state PyTree.
            args: Context from partition, forwarded to velocity/score.
            key: PRNG key for Brownian motion increments.
            **kwargs: Override parameters (``t0``, ``t1``, ``num_steps``, ``backward``).

        Returns:
            :class:`SDESolution` with ``.ys = {"x": ..., "action": ...}``, each with a leading axis of length 1.
        """
        kw = dict(kwargs)
        kw.pop("saveat", None)
        t0 = kw.pop("t0", self.t0)
        t1 = kw.pop("t1", self.t1)
        num_steps = kw.pop("num_steps", self.num_steps)
        backward = kw.pop("backward", False)

        sign = -1.0 if backward else 1.0
        dt = (t1 - t0) / num_steps
        abs_dt = jnp.abs(dt)
        sqrt_dt = jnp.sqrt(abs_dt)
        d_total = sum(leaf.size for leaf in jax.tree.leaves(x0))

        def time_at(step_idx):
            # A single expression for the grid, so the time at which the carried
            # fields were evaluated is bitwise the time of the next forward kernel.
            return jnp.abs(t0 + step_idx * dt)

        def step_fn(carry, inputs):
            x_curr, b_curr, u_curr, action_curr = carry
            step_key, step_idx = inputs
            t_curr = time_at(step_idx)
            t_next = time_at(step_idx + 1)

            eps_curr = self.noise_schedule(t_curr)
            eps_next = self.noise_schedule(t_next)
            sigma_curr = jnp.sqrt(2.0 * eps_curr)
            sigma2_fwd = jnp.maximum(2.0 * eps_curr * abs_dt, 1e-20)
            sigma2_bwd = jnp.maximum(2.0 * eps_next * abs_dt, 1e-20)

            mu_gen = jax.tree.map(lambda bi, ui: bi + sign * ui, b_curr, u_curr)
            dw = self._standard_normal_like(step_key, x_curr)
            x_next = jax.tree.map(
                lambda x, m, w: x + m * dt + sigma_curr * sqrt_dt * w,
                x_curr,
                mu_gen,
                dw,
            )

            b_next, u_next = self._fields(t_next, x_next, args)
            mu_ghost = jax.tree.map(lambda bi, ui: bi - sign * ui, b_next, u_next)

            res_fwd_sq = sum(
                jnp.sum(jnp.square(x1_l - x0_l - m_l * dt))
                for x1_l, x0_l, m_l in zip(jax.tree.leaves(x_next), jax.tree.leaves(x_curr), jax.tree.leaves(mu_gen))
            )
            log_p_fwd = -0.5 * res_fwd_sq / sigma2_fwd - 0.5 * d_total * jnp.log(2.0 * jnp.pi * sigma2_fwd)

            res_bwd_sq = sum(
                jnp.sum(jnp.square(x0_l - (x1_l - m_b_l * dt)))
                for x0_l, x1_l, m_b_l in zip(
                    jax.tree.leaves(x_curr), jax.tree.leaves(x_next), jax.tree.leaves(mu_ghost)
                )
            )
            log_p_bwd = -0.5 * res_bwd_sq / sigma2_bwd - 0.5 * d_total * jnp.log(2.0 * jnp.pi * sigma2_bwd)

            action_next = action_curr + log_p_fwd - log_p_bwd
            return (x_next, b_next, u_next, action_next), None

        b_init, u_init = self._fields(time_at(jnp.asarray(0)), x0, args)
        step_keys = jax.random.split(key, num_steps)
        (x_final, _, _, total_action), _ = jax.lax.scan(
            step_fn, (x0, b_init, u_init, jnp.array(0.0)), (step_keys, jnp.arange(num_steps))
        )
        ys_x = jax.tree.map(lambda leaf: jnp.expand_dims(leaf, 0), x_final)
        ys_action = jnp.expand_dims(total_action, 0)
        return SDESolution(ys={"x": ys_x, "action": ys_action})
