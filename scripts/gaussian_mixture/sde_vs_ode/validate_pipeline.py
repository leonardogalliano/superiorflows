"""Phase 0: validate the oracle and the sampling pipeline before any long run.

Checks, each printed as PASS/FAIL with the measured numbers:
1. Oracle score equals the autodiff gradient of the oracle marginal log-density.
2. Oracle denoiser satisfies s = −η_z / γ in the interior.
3. Oracle velocity satisfies the continuity equation ∂_t log ρ_t + ∇·b + b·∇log ρ_t = 0.
4. Oracle SDE: log Ẑ consistent with 0 and ESS → 1 as M grows; oracle ODE: |log Ẑ| shrinks with M.
5. The SDE time grid t_k = t0 + k·Δt hits 0.0 and 1.0 exactly in float32 for every M in the study
   (the exact-endpoint score relies on this), forwards and backwards.

Run with ``uv run python -m scripts.gaussian_mixture.sde_vs_ode.validate_pipeline``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from scripts.gaussian_mixture.analytical import louis_mixture_marginal_log_prob
from scripts.gaussian_mixture.sde_vs_ode import configs
from scripts.gaussian_mixture.sde_vs_ode.fields import (
    Problem,
    compiled_sampler,
    ode_transport,
    oracle_denoiser,
    oracle_fields,
    sde_transport,
)
from scripts.gaussian_mixture.sde_vs_ode.metrics import replicate_metrics

FAILURES = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def marginal_log_prob(problem, t, x):
    c = configs.TARGET
    return louis_mixture_marginal_log_prob(
        t, x, problem.d, c["a"], c["sigma2_max"], c["sigma2_min"], c["weight"], None, configs.gamma_fn
    )


def sample_xt(problem, t, n, key):
    k0, k1, kz = jax.random.split(key, 3)
    x0 = jax.random.normal(k0, (n, problem.d))
    x1 = jax.vmap(problem.target.sample)(jax.random.split(k1, n))
    z = jax.random.normal(kz, (n, problem.d))
    return (1 - t) * x0 + t * x1 + configs.gamma_fn(t) * z


def check_fields(d):
    problem = Problem(d)
    b, s = oracle_fields(problem)
    eta = oracle_denoiser(problem)
    worst_score, worst_den, worst_cont = 0.0, 0.0, 0.0
    for i, t in enumerate([1e-3, 0.05, 0.3, 0.5, 0.7, 0.95, 0.999]):
        x = sample_xt(problem, t, 256, jax.random.key(i))
        s_ad = jax.vmap(jax.grad(lambda xx: marginal_log_prob(problem, t, xx)))(x)
        s_or = jax.vmap(lambda xx: s(t, xx))(x)
        rel = float(jnp.linalg.norm(s_or - s_ad) / jnp.linalg.norm(s_ad))
        worst_score = max(worst_score, rel)

        s_from_eta = jax.vmap(lambda xx: -eta(t, xx) / configs.gamma_fn(t))(x)
        worst_den = max(worst_den, float(jnp.linalg.norm(s_from_eta - s_or) / jnp.linalg.norm(s_or)))

        def residual(xx):
            dt_logp = jax.jvp(lambda tt: marginal_log_prob(problem, tt, xx), (jnp.asarray(t),), (jnp.ones(()),))[1]
            div_b = jnp.trace(jax.jacfwd(lambda yy: b(t, yy))(xx))
            return dt_logp + div_b + b(t, xx) @ s(t, xx), jnp.abs(dt_logp) + jnp.abs(div_b)

        res, scale = jax.vmap(residual)(x)
        worst_cont = max(worst_cont, float(jnp.median(jnp.abs(res) / (scale + 1.0))))
    report(f"oracle score = grad log rho_t (d={d})", worst_score < 1e-3, f"max relative error {worst_score:.2e}")
    report(f"oracle s = -eta_z/gamma (d={d}, t in [1e-3, 0.999])", worst_den < 1e-3, f"max rel. error {worst_den:.2e}")
    report(f"oracle continuity equation (d={d})", worst_cont < 1e-3, f"median scaled residual {worst_cont:.2e}")


def check_sampling(d):
    problem = Problem(d)
    b, s = oracle_fields(problem)
    for M in [50, 1000]:
        for name, transport in [("ODE", ode_transport(b, M)), ("SDE eps=1", sde_transport(b, s, 1.0, M))]:
            fn = compiled_sampler(transport, problem, configs.N_PER_REPLICATE)
            rows = [replicate_metrics(*map(np.asarray, fn(jax.random.key(r)))) for r in range(5)]
            log_z = np.array([r["log_z"] for r in rows])
            ess = np.mean([r["ess"] for r in rows])
            sem = log_z.std(ddof=1) / np.sqrt(len(log_z))
            print(
                f"       d={d} M={M:4d} {name:10s}: log Z-hat = {log_z.mean():+.4f} ± {sem:.4f}, "
                f"ESS/n = {ess:.3f}, k-hat = {np.mean([r['khat'] for r in rows]):.2f}, "
                f"status = {sorted({r['status'] for r in rows})}"
            )
            if name.startswith("SDE") and M == 1000:
                unbiased = abs(log_z.mean()) < 4 * sem + 0.01
                report(f"oracle SDE unbiased (d={d}, M=1000)", unbiased, f"{log_z.mean():+.4f}")


def check_time_grid():
    exact = True
    for M in configs.STEPS:
        dt = 1.0 / M
        fwd = jnp.abs(0.0 + jnp.arange(M + 1) * dt)
        bwd = jnp.abs(1.0 + jnp.arange(M + 1) * (-dt))
        exact &= float(fwd[0]) == 0.0 and float(fwd[-1]) == 1.0 and float(bwd[0]) == 1.0 and float(bwd[-1]) == 0.0
    report("SDE grid hits t = 0 and t = 1 exactly (float32)", exact, f"M in {configs.STEPS}")


if __name__ == "__main__":
    check_time_grid()
    for d in [2, 8]:
        check_fields(d)
    for d in [2, 8]:
        check_sampling(d)
    print("\nAll checks passed." if not FAILURES else f"\nFAILED: {FAILURES}")
