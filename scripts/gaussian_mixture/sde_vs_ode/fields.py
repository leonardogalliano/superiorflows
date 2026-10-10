"""Fields (oracle, learnt, hybrid) and samplers for the SDE vs ODE study.

A *field source* provides a velocity ``b(t, x, args)`` and a score ``s(t, x, args)`` for the
interpolant ``x_t = (1 − t) x0 + t x1 + γ(t) z`` with ``γ = configs.gamma_fn``. Samplers built
here return ``(y, log_w)`` with ``log_w = log ρ★(y) − log_weight(y)`` (ρ★ normalised, Z = 1).
"""

import json
from pathlib import Path
from typing import Any, Callable

import diffrax as dfx
import distreqx.distributions as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from scripts.gaussian_mixture.analytical import (
    AnalyticalScoreField,
    AnalyticalVelocityField,
    louis_mixture_denoiser,
)
from scripts.gaussian_mixture.sde_vs_ode import configs
from scripts.gaussian_mixture.train_gaussian_mixture import (
    build_louis_mixture,
    build_model,
    restore_evaluation_model,
)
from superiorflows import NeuralSampler, ODEBijector, SDETransport


class ExactEndpointScore(eqx.Module):
    """Learnt score ``s = −η_z / γ`` with no floor, and the exact scores where γ = 0.

    At ``t = 0`` and ``t = 1`` the formula is 0/0 (and ±∞ with a network, since η_z ≠ 0 there);
    the true values are known: ``∇log ρ0`` and ``∇log ρ★``. The SDE grid ``t_k = k/M`` hits 0.0
    and 1.0 exactly in float32 (checked in ``validate_pipeline``), so exact comparisons are safe.
    """

    denoiser: Any
    base_score_fn: Any = eqx.field(static=True)
    target_score_fn: Any = eqx.field(static=True)

    def __call__(self, t, x, args=None):
        t = jnp.asarray(t)
        gamma = configs.gamma_fn(t)
        # The denominator only matters where 0 < t < 1, i.e. where γ > 0; at the endpoints the
        # interior branch is discarded, and dividing by 1 there keeps it finite for autodiff.
        interior = -self.denoiser(t, x, args) / jnp.where(gamma > 0, gamma, 1.0)
        return jnp.where(t <= 0.0, self.base_score_fn(x), jnp.where(t >= 1.0, self.target_score_fn(x), interior))


class SmoothBlendedScoreField(eqx.Module):
    """Gemini's boundary blend, copied verbatim from its scratch ``investigate_si_sde.py``.

    ``s = w0 s_base + w1 s_target + max(1 − w0 − w1, 0) s_raw`` with Gaussian envelopes
    ``w0 = exp(−t²/2σ²)``, ``w1 = exp(−(1 − t)²/2σ²)``.
    """

    score_fn: Any
    target_score_fn: Any = eqx.field(static=True)
    base_score_fn: Any = eqx.field(static=True)
    sigma_bnd: float = eqx.field(static=True, default=0.08)

    def __call__(self, t, x, args=None):
        t_arr = jnp.asarray(t, dtype=jnp.float32)
        w0 = jnp.exp(-0.5 * (t_arr / self.sigma_bnd) ** 2)
        w1 = jnp.exp(-0.5 * ((1.0 - t_arr) / self.sigma_bnd) ** 2)
        w_mid = jnp.maximum(1.0 - w0 - w1, 0.0)
        return w0 * self.base_score_fn(x) + w1 * self.target_score_fn(x) + w_mid * self.score_fn(t, x, args)


class TwoSidedBlend(eqx.Module):
    """Main score candidate (PLAN.md section 5.2): base blend near t = 0, rescaled target score near t = 1.

    ``s = w1 S1 + (1 − w1) [w0 (−x) + (1 − w0) s_raw]`` with ``w0 = exp(−t²/2σ_b²)`` and
    ``S1 = ∇log ρ★(x/t)/t``, ``w1 = exp(−r/κ)``, ``r = ((1 − t)² + γ²) / (t² σ²_min)``. Near t = 1,
    ``x_t = t x1 + n`` with a small centred blur n of variance ``(1 − t)² + γ²``, so ρ_t is ρ★ rescaled
    by t and slightly blurred; ``w1`` trusts S1 only while that blur is small compared with the
    target's narrowest variance.
    """

    raw: ExactEndpointScore
    base_score_fn: Any = eqx.field(static=True)
    target_score_fn: Any = eqx.field(static=True)
    sigma_bnd: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)
    sigma2_min: float = eqx.field(static=True)

    def __call__(self, t, x, args=None):
        t = jnp.asarray(t)
        w0 = jnp.exp(-0.5 * (t / self.sigma_bnd) ** 2)
        inner = w0 * self.base_score_fn(x) + (1 - w0) * self.raw(t, x, args)
        # S1 and r are only used where t > 0 (w1 = 0 at t = 0); dividing by 1 there keeps them finite.
        t_safe = jnp.where(t > 0, t, 1.0)
        blur = (1 - t) ** 2 + configs.gamma_fn(t) ** 2
        w1 = jnp.where(t > 0, jnp.exp(-blur / (t_safe**2 * self.sigma2_min) / self.kappa), 0.0)
        s1 = self.target_score_fn(x / t_safe) / t_safe
        return w1 * s1 + (1 - w1) * inner


class Problem:
    """Target, base and exact boundary scores for dimension ``d``.

    A plain class, not an ``eqx.Module``: its bound methods are passed as static fields of the
    score wrappers, and the arrays they close over are compile-time constants by design.
    """

    def __init__(self, d: int):
        self.d = d
        self.target = build_louis_mixture(d, **configs.TARGET)
        self.base = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))

    def target_score(self, x):
        return jax.grad(self.target.log_prob)(x)

    def base_score(self, x):
        return -x


def oracle_fields(problem: Problem) -> tuple[Callable, Callable]:
    """Exact ``(b, s)`` of the γ-interpolant (closed form, no singularity at the boundaries)."""
    t = configs.TARGET
    common = (problem.d, t["a"], t["sigma2_max"], t["sigma2_min"], t["weight"], None, configs.gamma_fn)
    return AnalyticalVelocityField(*common), AnalyticalScoreField(*common)


def oracle_denoiser(problem: Problem) -> Callable:
    """Exact noise denoiser ``η_z(t, x) = E[z | x_t = x]``."""
    t = configs.TARGET

    def eta_z(time, x, args=None):
        return louis_mixture_denoiser(
            time, x, problem.d, t["a"], t["sigma2_max"], t["sigma2_min"], t["weight"], None, configs.gamma_fn
        )[2]

    return eta_z


def restore_model(run_dir: Path, step: int | None = None):
    """Restore a trained ``VelocityDenoiserPair`` at ``step`` (latest if None), at its evaluation parameters.

    For schedule-free AdamW this is the averaged iterate x, not the training iterate y.
    Returns ``(model, config, step)``.
    """
    cfg = json.loads((run_dir / "config.json").read_text())
    template = build_model(cfg, cfg["target"]["d"], key=jax.random.key(0))
    manager = ocp.CheckpointManager(run_dir.resolve(), item_names=("model", "optimizer", "metadata"))
    step = manager.latest_step() if step is None else step
    return restore_evaluation_model(manager, step, template, cfg), cfg, step


def learnt_fields(model, problem: Problem, treatment: str) -> tuple[Callable, Callable]:
    """Learnt ``(b, s)`` with one of the score treatments in ``configs.SCORE_TREATMENTS``."""
    raw = ExactEndpointScore(model.denoiser, problem.base_score, problem.target_score)
    if treatment == "raw":
        score = raw
    elif treatment == "two_sided":
        score = TwoSidedBlend(
            raw=raw,
            base_score_fn=problem.base_score,
            target_score_fn=problem.target_score,
            sigma_bnd=configs.BLEND_SIGMA,
            kappa=configs.KAPPA,
            sigma2_min=configs.TARGET["sigma2_min"],
        )
    elif treatment == "gemini":
        score = SmoothBlendedScoreField(
            score_fn=raw,
            target_score_fn=problem.target_score,
            base_score_fn=problem.base_score,
            sigma_bnd=configs.BLEND_SIGMA,
        )
    else:
        raise ValueError(f"Unknown score treatment '{treatment}'. Available: {configs.SCORE_TREATMENTS}")
    return model.velocity_field, score


def ode_transport(velocity: Callable, num_steps: int) -> ODEBijector:
    """Euler with exact (autodiff) divergence, constant step 1/M, for both solver pairs."""
    return ODEBijector(
        velocity_field=velocity,
        solver=dfx.Euler(),
        augmented_solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        augmented_stepsize_controller=dfx.ConstantStepSize(),
        dt0=1.0 / num_steps,
    )


def sde_transport(velocity: Callable, score: Callable, eps: float, num_steps: int) -> SDETransport:
    """Euler–Maruyama with in-step path action and constant diffusion ε."""
    return SDETransport(
        velocity_field=velocity,
        score_field=score,
        noise_schedule=lambda t: eps,
        num_steps=num_steps,
    )


def compiled_sampler(transport, problem: Problem, n: int):
    """Ahead-of-time compiled ``key → (y, log_w)`` for ``n`` samples, so timing excludes compilation."""
    sampler = NeuralSampler(transport=transport, base_distribution=problem.base)

    def run(key):
        y, log_weight = jax.vmap(sampler.sample_and_log_weight)(jax.random.split(key, n))
        return y, jax.vmap(problem.target.log_prob)(y) - log_weight

    return jax.jit(run).lower(jax.random.key(0)).compile()
