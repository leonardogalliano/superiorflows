"""SDE-based transport with in-step Girsanov path action.

Implements stochastic continuous transport via a zero-allocation native JAX
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
    """Stochastic transport via zero-allocation native JAX Euler–Maruyama integration.

    Integrates the forward SDE:

        dX = [b̂(t,X) + u(t,X)] dt + √(2ε(t)) dW

    where ``u(t, X)`` is the osmotic drift. The osmotic drift can either
    be provided directly via ``osmotic_drift``, or computed from a score
    field via ``u(t, X) = ε(t) ŝ(t, X)`` when ``score_field`` is supplied.

    ``velocity_field``, ``score_field``, and ``osmotic_drift`` are arbitrary
    callables with signature ``(t, x_dynamic, args)``. ``noise_schedule`` is an
    arbitrary callable ``ε(t) → ℝ≥0``.

    The integration engine compiles to a single, non-allocating ``jax.lax.scan``
    loop with pre-sampled Brownian increments, ensuring optimal memory
    coalescing and arithmetic throughput across CPU and GPU hardware.

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

    # --- Internal drift construction ---

    def _make_drifts(self):
        """Build forward and backward drift closures.

        If ``osmotic_drift`` is provided, it is evaluated directly:
            forward_drift  = b + u
            backward_drift = b - u
        Otherwise, osmotic drift is evaluated via the score and diffusion schedule:
            forward_drift  = b + ε s
            backward_drift = b - ε s

        Returns:
            Tuple ``(forward_drift, backward_drift)`` with signature
            ``(t, x, args) → pytree``.
        """
        if self.osmotic_drift is not None:

            def forward_drift(t, x, _args):
                b = self.velocity_field(t, x, _args)
                u = self.osmotic_drift(t, x, _args)
                return jax.tree.map(lambda bi, ui: bi + ui, b, u)

            def backward_drift(t, x, _args):
                b = self.velocity_field(t, x, _args)
                u = self.osmotic_drift(t, x, _args)
                return jax.tree.map(lambda bi, ui: bi - ui, b, u)

            return forward_drift, backward_drift

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

    # --- SDE integration core routines ---

    def _integrate(self, x0, *, args=None, key, **kwargs):
        """Plain Euler–Maruyama integration on pre-partitioned dynamic state.

        Args:
            x0: Dynamic state PyTree.
            args: Context from partition, forwarded to velocity/score.
            key: PRNG key for Brownian motion increments.
            **kwargs: Override parameters (``t0``, ``t1``, ``num_steps``, ``backward``).

        Returns:
            :class:`SDESolution` with ``.ys`` containing the trajectory with leading time axis.
        """
        kw = dict(kwargs)
        kw.pop("saveat", None)
        t0 = kw.pop("t0", self.t0)
        t1 = kw.pop("t1", self.t1)
        num_steps = kw.pop("num_steps", self.num_steps)
        backward = kw.pop("backward", False)

        fwd_drift, bwd_drift = self._make_drifts()
        gen_drift = bwd_drift if backward else fwd_drift

        dt = (t1 - t0) / num_steps
        abs_dt = jnp.abs(dt)
        sqrt_dt = jnp.sqrt(abs_dt)

        leaves, treedef = jax.tree.flatten(x0)
        subkeys = jax.random.split(key, len(leaves)) if leaves else []
        dW_leaves = [
            jax.random.normal(k, shape=(num_steps,) + leaf.shape, dtype=leaf.dtype) for k, leaf in zip(subkeys, leaves)
        ]

        def step_fn(x_curr, step_idx):
            t_curr = t0 + step_idx * dt
            t0_phys = jnp.abs(t_curr)
            eps_t0 = self.noise_schedule(t0_phys)
            sigma_t0 = jnp.sqrt(2.0 * eps_t0)

            dw_step = jax.tree.unflatten(treedef, [leaf_arr[step_idx] for leaf_arr in dW_leaves])
            mu = gen_drift(t0_phys, x_curr, args)
            x_next = jax.tree.map(
                lambda x, m, dw: x + m * dt + sigma_t0 * sqrt_dt * dw,
                x_curr,
                mu,
                dw_step,
            )
            return x_next, None

        x_final, _ = jax.lax.scan(step_fn, x0, jnp.arange(num_steps))
        ys = jax.tree.map(lambda leaf: jnp.expand_dims(leaf, 0), x_final)
        return SDESolution(ys=ys)

    def _integrate_with_action(self, x0, *, args=None, key, **kwargs):
        """Euler–Maruyama integration with in-step Girsanov path action accumulation.

        Args:
            x0: Dynamic state PyTree.
            args: Context from partition, forwarded to velocity/score.
            key: PRNG key for Brownian motion increments.
            **kwargs: Override parameters (``t0``, ``t1``, ``num_steps``, ``backward``).

        Returns:
            :class:`SDESolution` with ``.ys = {"x": ..., "action": ...}``.
        """
        kw = dict(kwargs)
        kw.pop("saveat", None)
        t0 = kw.pop("t0", self.t0)
        t1 = kw.pop("t1", self.t1)
        num_steps = kw.pop("num_steps", self.num_steps)
        backward = kw.pop("backward", False)

        fwd_drift, bwd_drift = self._make_drifts()
        gen_drift = bwd_drift if backward else fwd_drift
        ghost_drift = fwd_drift if backward else bwd_drift

        dt = (t1 - t0) / num_steps
        abs_dt = jnp.abs(dt)
        sqrt_dt = jnp.sqrt(abs_dt)

        leaves, treedef = jax.tree.flatten(x0)
        d_total = sum(leaf.size for leaf in leaves)
        subkeys = jax.random.split(key, len(leaves)) if leaves else []
        dW_leaves = [
            jax.random.normal(k, shape=(num_steps,) + leaf.shape, dtype=leaf.dtype) for k, leaf in zip(subkeys, leaves)
        ]

        def step_fn(carry, step_idx):
            x_curr, action_curr = carry
            t_curr = t0 + step_idx * dt
            t_next = t_curr + dt

            t0_phys = jnp.abs(t_curr)
            t1_phys = jnp.abs(t_next)

            eps_t0 = self.noise_schedule(t0_phys)
            eps_t1 = self.noise_schedule(t1_phys)

            sigma_t0 = jnp.sqrt(2.0 * eps_t0)
            sigma2_fwd = jnp.maximum(2.0 * eps_t0 * abs_dt, 1e-20)
            sigma2_bwd = jnp.maximum(2.0 * eps_t1 * abs_dt, 1e-20)

            dw_step = jax.tree.unflatten(treedef, [leaf_arr[step_idx] for leaf_arr in dW_leaves])
            mu_gen = gen_drift(t0_phys, x_curr, args)
            x_next = jax.tree.map(
                lambda x, m, dw: x + m * dt + sigma_t0 * sqrt_dt * dw,
                x_curr,
                mu_gen,
                dw_step,
            )

            mu_ghost = ghost_drift(t1_phys, x_next, args)

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

            delta_action = log_p_fwd - log_p_bwd
            action_next = action_curr + delta_action
            return (x_next, action_next), None

        (x_final, total_action), _ = jax.lax.scan(step_fn, (x0, jnp.array(0.0)), jnp.arange(num_steps))
        ys_x = jax.tree.map(lambda leaf: jnp.expand_dims(leaf, 0), x_final)
        ys_action = jnp.expand_dims(total_action, 0)
        return SDESolution(ys={"x": ys_x, "action": ys_action})
