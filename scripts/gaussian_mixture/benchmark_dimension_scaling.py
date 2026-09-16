"""Systematic dimension-scaling benchmark: Analytical Oracles and Trained Models across dimensions.

Evaluates scaling across dimensions d in {4, 8, 16, 32, 64} (powers of two)
and integration steps M in {100, 200, 500, 1000} on the Louis Gaussian mixture (a=5.25):

1. Oracle ODE (Euler, dt = 1/M)
2. Oracle SDE (Native SDETransport with Girsanov path action, M steps)
3. Trained Run 6 ODE (Full Stochastic Interpolant with TSM, Euler)
4. Trained Run 6 SDE (Full Stochastic Interpolant with TSM, Native SDE)

Measures:
- Free energy bias: Ẑ, SEM(Ẑ), and Student's t-statistic
- Variance & Sampling quality: Effective Sample Size (ESS %)
- Computational efficiency: Wall-clock time (s) and Throughput (ESS / second)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import diffrax as dfx
import distreqx.distributions as dsx
import jax
import jax.numpy as jnp
import numpy as np
import typer

from scripts.gaussian_mixture.analytical import (
    build_louis_parameters,
    make_oracle_bijector,
    make_oracle_sde_transport,
)
from scripts.gaussian_mixture.compare_sampling_methods import (
    BoundaryExactScoreField,
    extract_model_fields,
    resolve_checkpoint_dir,
    restore_trained_model,
)
from superiorflows.ode import ODEBijector
from superiorflows.sampler import NeuralSampler
from superiorflows.sde import SDETransport

app = typer.Typer(pretty_exceptions_show_locals=False)


def build_target_dist(d: int, a: float, weight: float = 2.0 / 3.0):
    """Instantiate the analytical bimodal Gaussian mixture target."""
    weights, means, vars_ = build_louis_parameters(d, a, weight=weight)
    comp1 = dsx.MultivariateNormalDiag(means[0], jnp.sqrt(vars_[0]))
    comp2 = dsx.MultivariateNormalDiag(means[1], jnp.sqrt(vars_[1]))

    class LouisTarget:
        def log_prob(self, x):
            lp1 = comp1.log_prob(x) + jnp.log(weights[0])
            lp2 = comp2.log_prob(x) + jnp.log(weights[1])
            return jax.nn.logsumexp(jnp.stack([lp1, lp2]))

    return LouisTarget(), weights


def evaluate_sampler(
    sampler: NeuralSampler,
    target_dist: Any,
    num_samples: int,
    batch_size: int,
    key: jax.Array,
) -> Dict[str, float]:
    """Evaluate partition function estimate, SEM, t-statistic, ESS, and wall-clock time."""

    @jax.jit
    def sample_batch(rng):
        keys = jax.random.split(rng, batch_size)
        return jax.vmap(sampler.sample_and_log_weight)(keys)

    # Warmup / JIT compilation
    k_warm, k_eval = jax.random.split(key)
    _ = sample_batch(k_warm)[0].block_until_ready()

    num_batches = max(1, num_samples // batch_size)
    batch_keys = jax.random.split(k_eval, num_batches)
    all_samples = []
    all_log_weights = []

    t0 = time.time()
    for b_idx in range(num_batches):
        x_b, log_w_b = sample_batch(batch_keys[b_idx])
        all_samples.append(np.asarray(x_b.block_until_ready()))
        all_log_weights.append(np.asarray(log_w_b.block_until_ready()))
    wall_time = time.time() - t0

    samples = np.concatenate(all_samples, axis=0)
    log_weights = np.concatenate(all_log_weights, axis=0)

    log_rho = np.asarray(jax.vmap(target_dist.log_prob)(jnp.asarray(samples)))
    log_ratios = log_rho - log_weights

    max_ratio = np.max(log_ratios)
    unnorm_w = np.exp(log_ratios - max_ratio)
    denom = len(unnorm_w) * np.sum(unnorm_w**2)
    ess = float((np.sum(unnorm_w) ** 2) / max(denom, 1e-12)) * 100.0

    weights = np.exp(log_ratios)
    z_hat = float(np.mean(weights))
    abs_log_z = float(np.abs(np.log(np.maximum(z_hat, 1e-12))))

    sem_z = float(np.std(weights) / np.sqrt(len(weights)))
    t_stat = float((z_hat - 1.0) / max(sem_z, 1e-12))

    eff_samples = (ess / 100.0) * len(weights)
    ess_rate = float(eff_samples / max(wall_time, 1e-6))

    return {
        "z_hat": z_hat,
        "abs_log_z": abs_log_z,
        "sem_z": sem_z,
        "t_stat": t_stat,
        "ess": ess,
        "wall_time": wall_time,
        "ess_rate": ess_rate,
    }


@app.command()
def main(
    dims: str = typer.Option("4,8,16,32,64", help="Comma-separated dimensions (powers of two)."),
    steps: str = typer.Option("100,500", help="Comma-separated step counts."),
    a: float = typer.Option(5.25, help="Separation parameter."),
    num_samples: int = typer.Option(2048, help="Number of samples (matching PhD student benchmark)."),
    batch_size: int = typer.Option(512, help="Batch size for parallel generation."),
    ckpt_dir: Path = typer.Option(Path("tmp/ckpt_powers_of_two"), help="Directory containing trained models."),
    output_json: Optional[Path] = typer.Option(None, help="Output JSON results path."),
    seed: int = typer.Option(42, help="PRNG seed."),
    boundary_threshold: float = typer.Option(1e-4, help="Boundary score safety threshold."),
):
    """Run systematic high-dimensional scaling comparison across dimensions and steps."""
    dim_list = [int(d.strip()) for d in dims.split(",") if d.strip()]
    step_list = [int(s.strip()) for s in steps.split(",") if s.strip()]

    print("\n" + "=" * 115)
    print(f"DIMENSION SCALING BENCHMARK: ORACLE vs. TRAINED RUN 6 (a={a}, N={num_samples})")
    print("=" * 115 + "\n")

    headers = [
        "d",
        "M",
        "Paradigm",
        "Method",
        "Ẑ",
        "SEM(Ẑ)",
        "t-stat",
        "|log Ẑ|",
        "ESS (%)",
        "Time (s)",
        "ESS / sec",
    ]
    header_str = f"| {' | '.join(headers)} |"
    sep_str = f"|{'-|-'.join([':' + '-' * (len(h) - 2) + ':' for h in headers])}|"

    print(header_str)
    print(sep_str)

    results: Dict[str, Any] = {}
    master_key = jax.random.key(seed)

    for d in dim_list:
        results[f"d_{d}"] = {}
        base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))
        target_dist, _ = build_target_dist(d, a)

        def target_score_fn(x):
            return jax.grad(target_dist.log_prob)(x)

        def base_score_fn(x):
            return jax.grad(base_dist.log_prob)(x)

        # 1. Load trained model for this dimension if available
        d_ckpt_dir = ckpt_dir / f"d{d}_a{a}"
        resolved_ckpt = resolve_checkpoint_dir(d_ckpt_dir)
        trained_fields = None
        if resolved_ckpt is not None:
            cfg = json.load(open(resolved_ckpt / "config.json"))
            model = restore_trained_model(resolved_ckpt, cfg, d)
            vf_t, sf_t, _, _, _, _ = extract_model_fields(model, cfg)
            safe_sf_t = BoundaryExactScoreField(
                score_fn=sf_t,
                target_score_fn=target_score_fn,
                base_score_fn=base_score_fn,
                boundary_threshold=boundary_threshold,
            )
            trained_fields = (vf_t, safe_sf_t)

        for M in step_list:
            step_key = f"M_{M}"
            results[f"d_{d}"][step_key] = {}

            # A. Oracle ODE
            oracle_ode_bij = make_oracle_bijector(
                d=d,
                a=a,
                solver=dfx.Euler(),
                augmented_solver=dfx.Euler(),
                stepsize_controller=dfx.ConstantStepSize(),
                augmented_stepsize_controller=dfx.ConstantStepSize(),
                dt0=1.0 / M,
            )
            oracle_ode_sampler = NeuralSampler(transport=oracle_ode_bij, base_distribution=base_dist)
            master_key, k_eval = jax.random.split(master_key)
            m_res = evaluate_sampler(oracle_ode_sampler, target_dist, num_samples, batch_size, k_eval)
            results[f"d_{d}"][step_key]["oracle_ode"] = m_res
            row = [
                f"{d}",
                f"{M}",
                "Oracle",
                "ODE",
                f"{m_res['z_hat']:.4f}",
                f"{m_res['sem_z']:.4f}",
                f"{m_res['t_stat']:+.2f}",
                f"{m_res['abs_log_z']:.4f}",
                f"{m_res['ess']:.1f}%",
                f"{m_res['wall_time']:.3f}",
                f"{m_res['ess_rate']:.1f}",
            ]
            print(f"| {' | '.join(row)} |")

            # B. Oracle SDE (Native SDETransport)
            oracle_sde_trans = make_oracle_sde_transport(
                d=d,
                a=a,
                noise_schedule=lambda t: 1.0,
                num_steps=M,
            )
            oracle_sde_sampler = NeuralSampler(transport=oracle_sde_trans, base_distribution=base_dist)
            master_key, k_eval = jax.random.split(master_key)
            m_res = evaluate_sampler(oracle_sde_sampler, target_dist, num_samples, batch_size, k_eval)
            results[f"d_{d}"][step_key]["oracle_sde"] = m_res
            row = [
                f"{d}",
                f"{M}",
                "Oracle",
                "SDE",
                f"{m_res['z_hat']:.4f}",
                f"{m_res['sem_z']:.4f}",
                f"{m_res['t_stat']:+.2f}",
                f"{m_res['abs_log_z']:.4f}",
                f"{m_res['ess']:.1f}%",
                f"{m_res['wall_time']:.3f}",
                f"{m_res['ess_rate']:.1f}",
            ]
            print(f"| {' | '.join(row)} |")

            # C. Trained Run 6 ODE
            if trained_fields is not None:
                vf_t, safe_sf_t = trained_fields
                trained_ode_bij = ODEBijector(
                    velocity_field=vf_t,
                    solver=dfx.Euler(),
                    augmented_solver=dfx.Euler(),
                    stepsize_controller=dfx.ConstantStepSize(),
                    augmented_stepsize_controller=dfx.ConstantStepSize(),
                    dt0=1.0 / M,
                )
                trained_ode_sampler = NeuralSampler(transport=trained_ode_bij, base_distribution=base_dist)
                master_key, k_eval = jax.random.split(master_key)
                m_res = evaluate_sampler(trained_ode_sampler, target_dist, num_samples, batch_size, k_eval)
                results[f"d_{d}"][step_key]["trained_ode"] = m_res
                row = [
                    f"{d}",
                    f"{M}",
                    "Trained (Run 6)",
                    "ODE",
                    f"{m_res['z_hat']:.4f}",
                    f"{m_res['sem_z']:.4f}",
                    f"{m_res['t_stat']:+.2f}",
                    f"{m_res['abs_log_z']:.4f}",
                    f"{m_res['ess']:.1f}%",
                    f"{m_res['wall_time']:.3f}",
                    f"{m_res['ess_rate']:.1f}",
                ]
                print(f"| {' | '.join(row)} |")

                # D. Trained Run 6 SDE
                trained_sde_trans = SDETransport(
                    velocity_field=vf_t,
                    score_field=safe_sf_t,
                    noise_schedule=lambda t: 1.0,
                    num_steps=M,
                )
                trained_sde_sampler = NeuralSampler(transport=trained_sde_trans, base_distribution=base_dist)
                master_key, k_eval = jax.random.split(master_key)
                m_res = evaluate_sampler(trained_sde_sampler, target_dist, num_samples, batch_size, k_eval)
                results[f"d_{d}"][step_key]["trained_sde"] = m_res
                row = [
                    f"{d}",
                    f"{M}",
                    "Trained (Run 6)",
                    "SDE",
                    f"{m_res['z_hat']:.4f}",
                    f"{m_res['sem_z']:.4f}",
                    f"{m_res['t_stat']:+.2f}",
                    f"{m_res['abs_log_z']:.4f}",
                    f"{m_res['ess']:.1f}%",
                    f"{m_res['wall_time']:.3f}",
                    f"{m_res['ess_rate']:.1f}",
                ]
                print(f"| {' | '.join(row)} |")

    print("\n" + "=" * 115 + "\n")

    if output_json is not None:

        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        with open(output_json, "w") as f:
            json.dump(results, f, indent=2, default=to_serializable)
        print(f"Saved benchmark results to {output_json}")


if __name__ == "__main__":
    app()
