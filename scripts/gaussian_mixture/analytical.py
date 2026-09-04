"""Closed-form analytical fields for the Gaussian mixture benchmark (Louis mixture).

Provides exact, pure JAX reference implementations of:
- Marginal density q(t, x)
- Score field s(t, x) = ∇_x log q(t, x)
- Denoiser fields η₀(t, x) = E[X₀ | X_t = x], η₁(t, x) = E[X₁ | X_t = x], η_z(t, x) = E[Z | X_t = x]
- Velocity field b(t, x) = E[dX_t/dt | X_t = x]
- Oracle ODE bijector and SDE transport constructors

Domain of analytical validity:
1. Base distribution: q₀(x) = N(0, I_d).
2. Target distribution: ρ*(x) = ∑_k w_k N(m_k, Σ_k) (Louis mixture).
3. Coupling: Independent pairing (x₀ ⊥ x₁).
4. Interpolant: Linear state combination I_t(x₀, x₁) = α(t) x₀ + β(t) x₁ + γ(t) z.
"""

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from superiorflows.ode import ODEBijector
from superiorflows.sde import SDETransport


def build_louis_parameters(
    d: int,
    a: float,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute component weights, means, and diagonal covariance vectors for Louis mixture."""
    i = jnp.arange(1, d + 1)
    var1 = (i / d) * sigma2_max + ((d - i) / d) * sigma2_min
    var2 = var1[::-1]

    weights = jnp.array([weight, 1.0 - weight])
    means = jnp.stack([-a * jnp.ones(d), a * jnp.ones(d)])
    vars_ = jnp.stack([var1, var2])
    return weights, means, vars_


def _eval_schedules(
    t: float,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
) -> Tuple[float, float, float, float, float, float]:
    """Evaluate interpolant coefficients and their exact time derivatives via autodiff."""
    t_arr = jnp.asarray(t, dtype=jnp.float32)

    if s_fn is not None:
        s_t, dot_s_t = jax.jvp(s_fn, (t_arr,), (jnp.ones_like(t_arr),))
    else:
        s_t = t_arr
        dot_s_t = jnp.ones_like(t_arr)

    alpha_t = 1.0 - s_t
    dot_alpha_t = -dot_s_t
    beta_t = s_t
    dot_beta_t = dot_s_t

    if gamma_fn is not None:
        gamma_t, dot_gamma_t = jax.jvp(gamma_fn, (t_arr,), (jnp.ones_like(t_arr),))
    else:
        gamma_t = jnp.zeros_like(t_arr)
        dot_gamma_t = jnp.zeros_like(t_arr)

    return alpha_t, beta_t, dot_alpha_t, dot_beta_t, gamma_t, dot_gamma_t


def louis_mixture_marginal_components(
    t: float,
    d: int,
    a: float,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
):
    """Compute time-dependent means and diagonal covariances of each mixture component."""
    weights, means, vars_ = build_louis_parameters(d, a, sigma2_max, sigma2_min, weight)
    alpha_t, beta_t, dot_alpha_t, dot_beta_t, gamma_t, dot_gamma_t = _eval_schedules(t, s_fn, gamma_fn)

    # μ_k(t) = β(t) m_k
    mu_t = beta_t * means

    # C_k(t) = β(t)² Σ_k + (α(t)² + γ(t)²) I
    base_var = alpha_t**2 + gamma_t**2
    cov_t = (beta_t**2) * vars_ + base_var

    return (
        weights,
        mu_t,
        cov_t,
        means,
        vars_,
        (alpha_t, beta_t, dot_alpha_t, dot_beta_t, gamma_t, dot_gamma_t),
    )


def _log_gaussian_diag(x: jnp.ndarray, mu: jnp.ndarray, cov_diag: jnp.ndarray) -> float:
    """Log-density of a diagonal Gaussian."""
    d = x.shape[-1]
    diff = x - mu
    maha = jnp.sum((diff**2) / cov_diag, axis=-1)
    log_det = jnp.sum(jnp.log(cov_diag), axis=-1)
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + maha)


def louis_mixture_responsibilities(
    t: float,
    x: jnp.ndarray,
    d: int,
    a: float,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, float]:
    """Compute posterior responsibilities γ_k(t, x) and total marginal log-prob log q(t, x)."""
    weights, mu_t, cov_t, _, _, _ = louis_mixture_marginal_components(
        t, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn
    )

    log_p0 = _log_gaussian_diag(x, mu_t[0], cov_t[0])
    log_p1 = _log_gaussian_diag(x, mu_t[1], cov_t[1])

    logits = jnp.array([jnp.log(weights[0]) + log_p0, jnp.log(weights[1]) + log_p1])
    log_q = jax.nn.logsumexp(logits)
    gamma = jax.nn.softmax(logits)
    return gamma, log_q


def louis_mixture_marginal_log_prob(
    t: float,
    x: jnp.ndarray,
    d: int,
    a: float,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
) -> float:
    """Exact marginal log-probability log q(t, x) of the Louis mixture at time t."""
    _, log_q = louis_mixture_responsibilities(t, x, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)
    return log_q


def louis_mixture_score(
    t: float,
    x: jnp.ndarray,
    d: int,
    a: float,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """Exact score field s(t, x) = ∇_x log q(t, x)."""
    gamma, _ = louis_mixture_responsibilities(t, x, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)
    _, mu_t, cov_t, _, _, _ = louis_mixture_marginal_components(t, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)

    grad_0 = -(x - mu_t[0]) / cov_t[0]
    grad_1 = -(x - mu_t[1]) / cov_t[1]

    return gamma[0] * grad_0 + gamma[1] * grad_1


def louis_mixture_denoiser(
    t: float,
    x: jnp.ndarray,
    d: int,
    a: float,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact denoisers (η₀, η₁, η_z) representing E[X₀|X_t=x], E[X₁|X_t=x], E[Z|X_t=x]."""
    gamma, _ = louis_mixture_responsibilities(t, x, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)
    _, mu_t, cov_t, means, vars_, sched = louis_mixture_marginal_components(
        t, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn
    )
    alpha_t, beta_t, _, _, gamma_t, _ = sched

    # Component-conditional expectations:
    # E[X₀ | X_t=x, k] = α(t) C_k(t)⁻¹ (x - μ_k(t))
    # E[X₁ | X_t=x, k] = m_k + β(t) Σ_k C_k(t)⁻¹ (x - μ_k(t))
    # E[Z  | X_t=x, k] = γ(t) C_k(t)⁻¹ (x - μ_k(t))
    diff_0 = (x - mu_t[0]) / cov_t[0]
    diff_1 = (x - mu_t[1]) / cov_t[1]

    eta_0_k0 = alpha_t * diff_0
    eta_0_k1 = alpha_t * diff_1
    eta_0 = gamma[0] * eta_0_k0 + gamma[1] * eta_0_k1

    eta_1_k0 = means[0] + beta_t * vars_[0] * diff_0
    eta_1_k1 = means[1] + beta_t * vars_[1] * diff_1
    eta_1 = gamma[0] * eta_1_k0 + gamma[1] * eta_1_k1

    eta_z_k0 = gamma_t * diff_0
    eta_z_k1 = gamma_t * diff_1
    eta_z = gamma[0] * eta_z_k0 + gamma[1] * eta_z_k1

    return eta_0, eta_1, eta_z


def louis_mixture_velocity(
    t: float,
    x: jnp.ndarray,
    d: int,
    a: float,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
) -> jnp.ndarray:
    """Exact velocity field b(t, x) = E[dI_t/dt | X_t = x]."""
    eta_0, eta_1, eta_z = louis_mixture_denoiser(t, x, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)
    _, _, _, _, _, sched = louis_mixture_marginal_components(t, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)
    _, _, dot_alpha_t, dot_beta_t, _, dot_gamma_t = sched

    return dot_alpha_t * eta_0 + dot_beta_t * eta_1 + dot_gamma_t * eta_z


# ── Oracle Bijector & SDE Transport Constructors ─────────────────────────────


class AnalyticalVelocityField(eqx.Module):
    """Pure JAX wrapper turning exact analytical velocity into an Equinox callable."""

    d: int = eqx.field(static=True)
    a: float = eqx.field(static=True)
    sigma2_max: float = eqx.field(static=True)
    sigma2_min: float = eqx.field(static=True)
    weight: float = eqx.field(static=True)
    s_fn: Optional[Callable] = eqx.field(static=True)
    gamma_fn: Optional[Callable] = eqx.field(static=True)

    def __call__(self, t: float, x: jnp.ndarray, args=None) -> jnp.ndarray:
        return louis_mixture_velocity(
            t,
            x,
            self.d,
            self.a,
            self.sigma2_max,
            self.sigma2_min,
            self.weight,
            self.s_fn,
            self.gamma_fn,
        )


class AnalyticalScoreField(eqx.Module):
    """Pure JAX wrapper turning exact analytical score into an Equinox callable."""

    d: int = eqx.field(static=True)
    a: float = eqx.field(static=True)
    sigma2_max: float = eqx.field(static=True)
    sigma2_min: float = eqx.field(static=True)
    weight: float = eqx.field(static=True)
    s_fn: Optional[Callable] = eqx.field(static=True)
    gamma_fn: Optional[Callable] = eqx.field(static=True)

    def __call__(self, t: float, x: jnp.ndarray, args=None) -> jnp.ndarray:
        return louis_mixture_score(
            t,
            x,
            self.d,
            self.a,
            self.sigma2_max,
            self.sigma2_min,
            self.weight,
            self.s_fn,
            self.gamma_fn,
        )


def make_oracle_bijector(
    d: int,
    a: float,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
    **kwargs,
) -> ODEBijector:
    """Instantiate an exact ODEBijector using closed-form analytical velocity."""
    vf = AnalyticalVelocityField(d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)
    return ODEBijector(
        velocity_field=vf,
        **kwargs,
    )


def make_oracle_sde_transport(
    d: int,
    a: float,
    noise_schedule: Callable,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
    s_fn: Optional[Callable] = None,
    gamma_fn: Optional[Callable] = None,
    num_steps: int = 100,
    osmotic_drift: Optional[Callable] = None,
    **kwargs,
) -> SDETransport:
    """Instantiate an exact SDETransport using closed-form analytical velocity and score."""
    vf = AnalyticalVelocityField(d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)
    if osmotic_drift is not None:
        return SDETransport(
            velocity_field=vf,
            noise_schedule=noise_schedule,
            osmotic_drift=osmotic_drift,
            num_steps=num_steps,
            **kwargs,
        )
    sf = AnalyticalScoreField(d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn)
    return SDETransport(
        velocity_field=vf,
        score_field=sf,
        noise_schedule=noise_schedule,
        num_steps=num_steps,
        **kwargs,
    )
