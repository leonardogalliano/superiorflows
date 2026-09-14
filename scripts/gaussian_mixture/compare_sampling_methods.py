"""Systematic comparison of generative sampling methods on Gaussian mixture benchmark.

Compares:
1. Analytical Oracles: Exact ODE and Exact SDE (across step counts).
2. Deterministic ODE Flow Matching (Run 1: gamma=0, Run 2: gamma>0, Run 3: probability flow).
3. Stochastic SDE with Constant Noise Schedule (eps = 1.0).
4. Stochastic SDE with Exact Osmotic Drift (eps(t) = lambda * scale(t), u(t, x) = -lambda * eta(t, x), lambda=1.0).
5. Vector Field & Denoiser Accuracy: E_b(t), raw denoiser E_eta(t), and score E_s(t).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import diffrax as dfx
import distreqx.distributions as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import scipy.optimize
import typer

from scripts.gaussian_mixture.analytical import (
    louis_mixture_denoiser,
    louis_mixture_score,
    louis_mixture_velocity,
    make_oracle_bijector,
    make_oracle_sde_transport,
)
from scripts.gaussian_mixture.train_gaussian_mixture import (
    ReconstructedScoreFromDenoiser,
    ReconstructedVelocityFromDenoiser,
    ScoreFromGamma,
    VelocityDenoiserPair,
    build_louis_mixture,
    build_model,
    build_schedule_fn,
)
from superiorflows.ode import ODEBijector
from superiorflows.sampler import NeuralSampler
from superiorflows.sde import SDETransport

app = typer.Typer(pretty_exceptions_show_locals=False)

T_EVAL_GRID = [0.02, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.98]


class BoundaryExactScoreField(eqx.Module):
    """Wraps score field with exact boundary values at t=0 and t=1 to prevent 0/0 singularities.

    At t=1, q_1 = rho_1, so s(1, x) = grad log rho_1(x).
    At t=0, q_0 = N(0, I), so s(0, x) = grad log q_0(x) = -x.
    On the interior t in (boundary_threshold, 1 - boundary_threshold), evaluates the model score field.
    """

    score_fn: Any
    target_score_fn: Any = eqx.field(static=True)
    base_score_fn: Any = eqx.field(static=True)
    boundary_threshold: float = eqx.field(static=True, default=1e-4)

    def __call__(self, t: float, x: jnp.ndarray, args=None) -> jnp.ndarray:
        t_arr = jnp.asarray(t)
        return jnp.where(
            t_arr >= 1.0 - self.boundary_threshold,
            self.target_score_fn(x),
            jnp.where(
                t_arr <= self.boundary_threshold,
                self.base_score_fn(x),
                self.score_fn(t, x, args),
            ),
        )


def resolve_checkpoint_dir(path: Optional[Path]) -> Optional[Path]:
    """Resolve a checkpoint path or latest subdirectory containing config.json."""
    if path is None or not path.exists():
        return None
    if (path / "config.json").exists():
        return path
    subdirs = [p for p in path.iterdir() if p.is_dir() and (p / "config.json").exists()]
    if not subdirs:
        return None
    return sorted(subdirs, key=lambda p: p.stat().st_mtime)[-1]


def compute_wasserstein_2(samples_p: np.ndarray, samples_q: np.ndarray) -> float:
    """Compute 2-Wasserstein distance W_2 between two empirical point clouds."""
    n = min(len(samples_p), len(samples_q))
    p = samples_p[:n]
    q = samples_q[:n]
    cost_matrix = np.sum((p[:, None, :] - q[None, :, :]) ** 2, axis=-1)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    mean_sq_dist = cost_matrix[row_ind, col_ind].mean()
    return float(np.sqrt(np.maximum(mean_sq_dist, 0.0)))


def restore_trained_model(ckpt_dir: Path, config: dict, d: int):
    """Restore trained equinox model from orbax checkpoint."""
    model_template = build_model(config, d, key=jax.random.key(0))
    model_params = eqx.filter(model_template, eqx.is_array)
    static_model = eqx.filter(model_template, eqx.is_array, inverse=True)

    checkpointer = ocp.CheckpointManager(ckpt_dir.resolve(), item_names=("model", "optimizer", "metadata"))
    latest_step = checkpointer.latest_step()
    restore_args = ocp.args.Composite(model=ocp.args.StandardRestore(model_params))
    restored = checkpointer.restore(latest_step, args=restore_args)
    return eqx.combine(restored.model, static_model)


def extract_model_fields(trained_model: Any, config: dict):
    """Extract velocity, score, and raw denoiser from model."""
    si_cfg = config.get("stochastic_interpolant", {})
    s_expr = si_cfg.get("interpolant_scheduler", "t")
    gamma_expr = si_cfg.get("noise_scheduler", "None")
    s_fn = build_schedule_fn(s_expr)
    gamma_fn = build_schedule_fn(gamma_expr)

    denoiser_only = si_cfg.get("denoiser_only", False)
    learn_denoiser = si_cfg.get("learn_denoiser", False)

    denoiser_raw = None
    denoiser_type = None

    if denoiser_only:
        vf = ReconstructedVelocityFromDenoiser(trained_model, s_fn)
        sf = ReconstructedScoreFromDenoiser(trained_model, s_fn)
        denoiser_raw = trained_model
        denoiser_type = "0"
    elif learn_denoiser and isinstance(trained_model, VelocityDenoiserPair):
        vf = trained_model.velocity_field
        if gamma_fn is not None:
            sf = ScoreFromGamma(trained_model.denoiser, gamma_fn)
            denoiser_type = "z"
        else:
            sf = ReconstructedScoreFromDenoiser(trained_model.denoiser, s_fn)
            denoiser_type = "0"
        denoiser_raw = trained_model.denoiser
    elif hasattr(trained_model, "velocity_field"):
        vf = trained_model.velocity_field
        sf = None
    else:
        vf = trained_model
        sf = None

    return vf, sf, denoiser_raw, denoiser_type, s_fn, gamma_fn


def evaluate_field_accuracy(
    vf: Optional[Any],
    sf: Optional[Any],
    denoiser_raw: Optional[Any],
    denoiser_type: Optional[str],
    d: int,
    a: float,
    s_fn: Optional[Any],
    gamma_fn: Optional[Any],
    target_dist: Any,
    base_dist: Any,
    *,
    key: jax.Array,
    t_points: List[float] = T_EVAL_GRID,
    n_points: int = 512,
    sigma2_max: float = 0.2,
    sigma2_min: float = 0.01,
    weight: float = 2.0 / 3.0,
) -> Dict[str, Any]:
    """Evaluate L2 error of velocity, denoiser, and score fields against analytical oracle."""
    results = {"t": t_points, "error_b": [], "error_eta": [], "error_s": []}

    oracle_v_vmap = jax.vmap(
        lambda t_val, x_val: louis_mixture_velocity(t_val, x_val, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn),
        in_axes=(None, 0),
    )
    oracle_s_vmap = jax.vmap(
        lambda t_val, x_val: louis_mixture_score(t_val, x_val, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn),
        in_axes=(None, 0),
    )
    oracle_den_vmap = jax.vmap(
        lambda t_val, x_val: louis_mixture_denoiser(t_val, x_val, d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn),
        in_axes=(None, 0),
    )

    for t_val in t_points:
        key, k0, k1, kz = jax.random.split(key, 4)
        x0 = jax.vmap(base_dist.sample)(jax.random.split(k0, n_points))
        x1 = jax.vmap(target_dist.sample)(jax.random.split(k1, n_points))
        z = jax.random.normal(kz, shape=(n_points, d))

        s_t = s_fn(t_val) if s_fn is not None else t_val
        alpha_t = 1.0 - s_t
        beta_t = s_t
        gamma_t = gamma_fn(t_val) if gamma_fn is not None else 0.0

        xt = alpha_t * x0 + beta_t * x1 + gamma_t * z

        # 1. Velocity error
        if vf is not None:
            v_true = oracle_v_vmap(t_val, xt)
            v_pred = jax.vmap(lambda x: vf(t_val, x))(xt)
            err_b = float(jnp.mean(jnp.sum((v_pred - v_true) ** 2, axis=-1)))
        else:
            err_b = None
        results["error_b"].append(err_b)

        # 2. Raw Denoiser error
        if denoiser_raw is not None and denoiser_type is not None:
            eta_0_true, _, eta_z_true = oracle_den_vmap(t_val, xt)
            target_eta_true = eta_z_true if denoiser_type == "z" else eta_0_true
            eta_pred = jax.vmap(lambda x: denoiser_raw(t_val, x))(xt)
            err_eta = float(jnp.mean(jnp.sum((eta_pred - target_eta_true) ** 2, axis=-1)))
        else:
            err_eta = None
        results["error_eta"].append(err_eta)

        # 3. Score error
        if sf is not None:
            s_true = oracle_s_vmap(t_val, xt)
            s_pred = jax.vmap(lambda x: sf(t_val, x))(xt)
            err_s = float(jnp.mean(jnp.sum((s_pred - s_true) ** 2, axis=-1)))
        else:
            err_s = None
        results["error_s"].append(err_s)

    return results


def evaluate_sampler_metrics(
    sampler: NeuralSampler,
    target_dist: Any,
    d: int,
    *,
    key: jax.Array,
    num_samples: int = 1000,
    batch_size: int = 500,
    sample_kwargs: Optional[dict] = None,
) -> Dict[str, float]:
    """Generate samples and evaluate Z, ESS, W_2, and Mode allocation."""
    skw = sample_kwargs or {}
    batch_sz = min(num_samples, batch_size)
    batches = max(1, num_samples // batch_sz)
    all_samples = []
    all_log_weights = []

    @eqx.filter_jit
    def sample_batch(rng):
        keys = jax.random.split(rng, batch_sz)
        return jax.vmap(lambda k: sampler.sample_and_log_weight(k, **skw))(keys)

    for _ in range(batches):
        key, subkey = jax.random.split(key)
        samples_b, log_w_b = sample_batch(subkey)
        all_samples.append(np.asarray(samples_b))
        all_log_weights.append(np.asarray(log_w_b))

    samples = np.concatenate(all_samples, axis=0)[:num_samples]
    log_weights = np.concatenate(all_log_weights, axis=0)[:num_samples]

    log_rho = np.asarray(jax.vmap(target_dist.log_prob)(jnp.asarray(samples)))
    log_ratios = log_rho - log_weights

    max_ratio = np.max(log_ratios)
    unnorm_w = np.exp(log_ratios - max_ratio)
    z_hat = float(np.mean(np.exp(log_ratios)))
    abs_log_z = float(np.abs(np.log(np.maximum(z_hat, 1e-12))))

    denom = len(unnorm_w) * np.sum(unnorm_w**2)
    ess = float((np.sum(unnorm_w) ** 2) / max(denom, 1e-12)) * 100.0

    key, subkey = jax.random.split(key)
    true_samples = np.asarray(jax.vmap(target_dist.sample)(jax.random.split(subkey, len(samples))))
    w2 = compute_wasserstein_2(samples, true_samples)

    mode1_count = np.sum(np.sum(samples, axis=-1) < 0.0)
    p_mode1 = float(mode1_count / len(samples))
    mode_err = float(np.abs(p_mode1 - (2.0 / 3.0)))

    return {
        "z_hat": z_hat,
        "abs_log_z": abs_log_z,
        "ess": ess,
        "w2": w2,
        "p_mode1": p_mode1,
        "mode_err": mode_err,
    }


def run_comparison_suite(
    ckpt_run1: Optional[Path] = None,
    ckpt_run2: Optional[Path] = None,
    ckpt_run3: Optional[Path] = None,
    ckpt_run4: Optional[Path] = None,
    ckpt_run5: Optional[Path] = None,
    ckpt_run6: Optional[Path] = None,
    benchmark_dir: Optional[Path] = Path("tmp/ckpt_benchmark"),
    steps_list: List[int] = [50, 100, 200],
    num_samples: int = 1000,
    batch_size: int = 500,
    seed: int = 42,
    include_oracles: bool = True,
    boundary_threshold: float = 1e-4,
) -> Dict[str, Any]:
    """Execute systematic sampling method comparison across models and integration steps."""
    d = 2
    a = 2.875
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))
    target_dist = build_louis_mixture(d, a)

    def target_score_fn(x):
        return jax.grad(target_dist.log_prob)(x)

    def base_score_fn(x):
        return jax.grad(base_dist.log_prob)(x)

    key = jax.random.key(seed)
    results: Dict[str, Any] = {
        "benchmark1_oracles": {},
        "benchmark2_training_error": {},
        "benchmark3_trained_models": {},
        "summary_table": {},
        "sde_scaling": {},
        "field_accuracy": {},
    }

    # Resolve paths
    c1 = resolve_checkpoint_dir(ckpt_run1) or (
        resolve_checkpoint_dir(benchmark_dir / "01_ode_velocity_no_gamma") if benchmark_dir else None
    )
    c2 = resolve_checkpoint_dir(ckpt_run2) or (
        resolve_checkpoint_dir(benchmark_dir / "02_ode_velocity_with_gamma") if benchmark_dir else None
    )
    c3 = resolve_checkpoint_dir(ckpt_run3) or (
        resolve_checkpoint_dir(benchmark_dir / "03_sde_full_velocity_and_score_with_gamma") if benchmark_dir else None
    )
    c4 = resolve_checkpoint_dir(ckpt_run4) or (
        resolve_checkpoint_dir(benchmark_dir / "04_sde_one_sided_denoiser_only") if benchmark_dir else None
    )
    c5 = resolve_checkpoint_dir(ckpt_run5) or (
        resolve_checkpoint_dir(benchmark_dir / "05_sde_one_sided_tsm") if benchmark_dir else None
    )
    c6 = resolve_checkpoint_dir(ckpt_run6) or (
        resolve_checkpoint_dir(benchmark_dir / "06_sde_full_with_tsm") if benchmark_dir else None
    )

    # ── Benchmark 1: Analytical Oracles (Varying Integration Steps) ───────────
    if include_oracles:
        print("\n" + "=" * 90)
        print("BENCHMARK 1: ANALYTICAL ORACLES (REWEIGHTING ACROSS INTEGRATION STEPS)")
        print("=" * 90)
        for steps in steps_list:
            step_key = f"steps_{steps}"
            results["benchmark1_oracles"][step_key] = {}
            print(f"\nEvaluating Analytical Oracles with M={steps} steps...")

            # 1. Oracle ODE
            oracle_ode_bijector = make_oracle_bijector(
                d,
                a,
                solver=dfx.Tsit5(),
                augmented_solver=dfx.Tsit5(),
                stepsize_controller=dfx.ConstantStepSize(),
                augmented_stepsize_controller=dfx.ConstantStepSize(),
                dt0=1.0 / steps,
            )
            oracle_ode_sampler = NeuralSampler(transport=oracle_ode_bijector, base_distribution=base_dist)
            key, subkey = jax.random.split(key)
            m_ode = evaluate_sampler_metrics(
                oracle_ode_sampler, target_dist, d, key=subkey, num_samples=num_samples, batch_size=batch_size
            )
            results["benchmark1_oracles"][step_key]["oracle_ode"] = m_ode

            # 2. Oracle SDE with eps = 1.0 (constant noise schedule)
            oracle_sde_trans = make_oracle_sde_transport(
                d,
                a,
                noise_schedule=lambda t: 1.0,
                num_steps=steps,
            )
            oracle_sde_sampler = NeuralSampler(transport=oracle_sde_trans, base_distribution=base_dist)
            key, subkey = jax.random.split(key)
            m_sde = evaluate_sampler_metrics(
                oracle_sde_sampler, target_dist, d, key=subkey, num_samples=num_samples, batch_size=batch_size
            )
            results["benchmark1_oracles"][step_key]["oracle_sde_eps1"] = m_sde

    # ── Benchmark 2: Training Error (Fields & Singularities Across Time) ──────
    print("\n" + "=" * 90)
    print("BENCHMARK 2: TRAINING ERROR (FIELDS & SINGULARITIES ACROSS TIME)")
    print("=" * 90)
    models_dict = {}
    for name, c_path in [
        ("Run 1", c1),
        ("Run 2", c2),
        ("Run 3", c3),
        ("Run 4", c4),
        ("Run 5 (TSM)", c5),
        ("Run 6 (TSM)", c6),
    ]:
        if c_path is not None:
            print(f"Evaluating field accuracy for {name} from {c_path}...")
            cfg = json.load(open(c_path / "config.json"))
            model = restore_trained_model(c_path, cfg, d)
            vf, sf, d_raw, d_type, s_fn, g_fn = extract_model_fields(model, cfg)
            key, kf = jax.random.split(key)
            fm = evaluate_field_accuracy(vf, sf, d_raw, d_type, d, a, s_fn, g_fn, target_dist, base_dist, key=kf)
            results["benchmark2_training_error"][name] = fm
            models_dict[name] = {
                "model": model,
                "config": cfg,
                "vf": vf,
                "sf": sf,
                "s_fn": s_fn,
                "g_fn": g_fn,
            }
    results["field_accuracy"] = results["benchmark2_training_error"]

    # ── Benchmark 3: Trained Models (The 4 Metrics Across Integration Steps) ──
    print("\n" + "=" * 90)
    print("BENCHMARK 3: TRAINED MODELS (THE 4 METRICS ACROSS INTEGRATION STEPS)")
    print("=" * 90)
    for steps in steps_list:
        step_key = f"steps_{steps}"
        results["benchmark3_trained_models"][step_key] = {}
        step_res = {}
        print(f"\nEvaluating Trained Models with M={steps} steps...")

        # 1. All ODEs
        for name, m_info in models_dict.items():
            vf = m_info["vf"]
            ode_b = ODEBijector(
                vf,
                solver=dfx.Tsit5(),
                augmented_solver=dfx.Tsit5(),
                stepsize_controller=dfx.ConstantStepSize(),
                augmented_stepsize_controller=dfx.ConstantStepSize(),
                dt0=1.0 / steps,
            )
            s_ode = NeuralSampler(transport=ode_b, base_distribution=base_dist)
            key, ks = jax.random.split(key)
            m_res = evaluate_sampler_metrics(
                s_ode, target_dist, d, key=ks, num_samples=num_samples, batch_size=batch_size
            )
            step_res[f"{name}_ode"] = m_res

        # 2. All SDEs with eps = 1.0 (constant noise, safe boundary score)
        for name in ["Run 3", "Run 4", "Run 5 (TSM)", "Run 6 (TSM)"]:
            if name in models_dict and models_dict[name]["sf"] is not None:
                vf = models_dict[name]["vf"]
                raw_sf = models_dict[name]["sf"]
                safe_sf = BoundaryExactScoreField(
                    score_fn=raw_sf,
                    target_score_fn=target_score_fn,
                    base_score_fn=base_score_fn,
                    boundary_threshold=boundary_threshold,
                )
                sde_trans = SDETransport(
                    velocity_field=vf,
                    score_field=safe_sf,
                    noise_schedule=lambda t: 1.0,
                    num_steps=steps,
                )
                s_sde = NeuralSampler(transport=sde_trans, base_distribution=base_dist)
                key, ks = jax.random.split(key)
                m_res = evaluate_sampler_metrics(
                    s_sde, target_dist, d, key=ks, num_samples=num_samples, batch_size=batch_size
                )
                step_res[f"{name}_sde_eps1"] = m_res

        results["benchmark3_trained_models"][step_key] = step_res

        # SDE scaling compatibility
        results["sde_scaling"][step_key] = {}
        if include_oracles and "oracle_sde_eps1" in results["benchmark1_oracles"].get(step_key, {}):
            results["sde_scaling"][step_key]["oracle_sde"] = results["benchmark1_oracles"][step_key]["oracle_sde_eps1"]
        if "Run 3_sde_eps1" in step_res:
            results["sde_scaling"][step_key]["trained_run3_sde"] = step_res["Run 3_sde_eps1"]
        if "Run 4_sde_eps1" in step_res:
            results["sde_scaling"][step_key]["trained_run4_sde"] = step_res["Run 4_sde_eps1"]
        if "Run 5 (TSM)_sde_eps1" in step_res:
            results["sde_scaling"][step_key]["trained_run5_sde"] = step_res["Run 5 (TSM)_sde_eps1"]
        if "Run 6 (TSM)_sde_eps1" in step_res:
            results["sde_scaling"][step_key]["trained_run6_sde"] = step_res["Run 6 (TSM)_sde_eps1"]

    # Summary table populated from representative step count (100 steps or last)
    rep_step = 100 if 100 in steps_list else steps_list[-1]
    rep_key = f"steps_{rep_step}"

    if include_oracles and rep_key in results["benchmark1_oracles"]:
        orc_dict = results["benchmark1_oracles"][rep_key]
        if "oracle_ode" in orc_dict:
            results["summary_table"]["Oracle: ODE (Exact)"] = {
                "transport": f"Analytical ODE ({rep_step} steps)",
                "metrics": orc_dict["oracle_ode"],
            }
        if "oracle_sde_eps1" in orc_dict:
            results["summary_table"]["Oracle: SDE (Exact)"] = {
                "transport": f"Analytical SDE ({rep_step} steps, eps=1)",
                "metrics": orc_dict["oracle_sde_eps1"],
            }

    if rep_key in results["benchmark3_trained_models"]:
        t_dict = results["benchmark3_trained_models"][rep_key]
        if "Run 1_ode" in t_dict:
            results["summary_table"]["Run 1: ODE (Velocity-only, γ=0)"] = {
                "transport": f"Continuous ODE ({rep_step} steps)",
                "metrics": t_dict["Run 1_ode"],
            }
        if "Run 2_ode" in t_dict:
            results["summary_table"]["Run 2: ODE (Velocity-only, γ>0)"] = {
                "transport": f"Continuous ODE ({rep_step} steps)",
                "metrics": t_dict["Run 2_ode"],
            }
        if "Run 3_ode" in t_dict:
            results["summary_table"]["Run 3: Probability Flow ODE"] = {
                "transport": f"Continuous ODE ({rep_step} steps)",
                "metrics": t_dict["Run 3_ode"],
            }
        if "Run 3_sde_eps1" in t_dict:
            results["summary_table"]["Run 3: SDE (Score-based, eps=1)"] = {
                "transport": f"SDE ({rep_step} steps, eps=1)",
                "metrics": t_dict["Run 3_sde_eps1"],
            }
        if "Run 4_ode" in t_dict:
            results["summary_table"]["Run 4: Probability Flow ODE"] = {
                "transport": f"Continuous ODE ({rep_step} steps)",
                "metrics": t_dict["Run 4_ode"],
            }
        if "Run 4_sde_eps1" in t_dict:
            results["summary_table"]["Run 4: SDE (Score-based, eps=1)"] = {
                "transport": f"SDE ({rep_step} steps, eps=1)",
                "metrics": t_dict["Run 4_sde_eps1"],
            }
        if "Run 5 (TSM)_ode" in t_dict:
            results["summary_table"]["Run 5: ODE (One-Sided TSM)"] = {
                "transport": f"Continuous ODE ({rep_step} steps)",
                "metrics": t_dict["Run 5 (TSM)_ode"],
            }
        if "Run 5 (TSM)_sde_eps1" in t_dict:
            results["summary_table"]["Run 5: SDE (One-Sided TSM, eps=1)"] = {
                "transport": f"SDE ({rep_step} steps, eps=1)",
                "metrics": t_dict["Run 5 (TSM)_sde_eps1"],
            }
        if "Run 6 (TSM)_ode" in t_dict:
            results["summary_table"]["Run 6: ODE (Full SI TSM)"] = {
                "transport": f"Continuous ODE ({rep_step} steps)",
                "metrics": t_dict["Run 6 (TSM)_ode"],
            }
        if "Run 6 (TSM)_sde_eps1" in t_dict:
            results["summary_table"]["Run 6: SDE (Full SI TSM, eps=1)"] = {
                "transport": f"SDE ({rep_step} steps, eps=1)",
                "metrics": t_dict["Run 6 (TSM)_sde_eps1"],
            }

    return results


def print_comparison_tables(results: Dict[str, Any]):
    """Format and print Markdown comparison tables matching the 3-step benchmark."""
    print("\n" + "=" * 90)
    print("                      SAMPLING METHODS BENCHMARK RESULTS")
    print("=" * 90)

    # 1. Main Summary Table (at representative step count)
    print("\n### Summary Overview Table")
    headers = [
        "Model Paradigm",
        "Transport",
        "\\lvert\\log Ẑ\\rvert",
        "ESS (%)",
        "W₂ Distance",
        "\\lvert P(M₁) - 2/3\\rvert",
    ]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-|-'.join([':' + '-' * (len(h) - 2) + ':' for h in headers])}|")
    for name, data in results.get("summary_table", {}).items():
        m = data["metrics"]
        row = [
            f"{name}",
            f"{data['transport'][:28]}",
            f"{m['abs_log_z']:.4f}",
            f"{m['ess']:.1f}%",
            f"{m['w2']:.4f}",
            f"{m['mode_err']:.4f}",
        ]
        print(f"| {' | '.join(row)} |")

    # 2. Benchmark 1 Table: Oracles Across Steps
    b1 = results.get("benchmark1_oracles", {})
    if b1:
        print("\n### Benchmark 1: Analytical Oracles Across Integration Steps")
        b1_headers = [
            "Steps M",
            "Oracle ODE ESS",
            "Oracle ODE \\lvert\\log Ẑ\\rvert",
            "Oracle ODE W₂",
            "Oracle SDE (ε=1) ESS",
            "Oracle SDE (ε=1) \\lvert\\log Ẑ\\rvert",
            "Oracle SDE (ε=1) W₂",
        ]
        print(f"| {' | '.join(b1_headers)} |")
        print(f"|{'-|-'.join([':' + '-' * (len(h) - 2) + ':' for h in b1_headers])}|")
        for step_key, step_data in b1.items():
            steps_num = step_key.replace("steps_", "")
            ode_m = step_data.get("oracle_ode", {})
            sde_m = step_data.get("oracle_sde_eps1", {})
            row = [
                f"{steps_num}",
                f"{ode_m.get('ess', 0.0):.1f}%",
                f"{ode_m.get('abs_log_z', 0.0):.4f}",
                f"{ode_m.get('w2', 0.0):.4f}",
                f"{sde_m.get('ess', 0.0):.1f}%",
                f"{sde_m.get('abs_log_z', 0.0):.4f}",
                f"{sde_m.get('w2', 0.0):.4f}",
            ]
            print(f"| {' | '.join(row)} |")

    # 3. Benchmark 2 Table: Training Error across Time Slices
    b2 = results.get("benchmark2_training_error", {})
    if b2:
        print("\n### Benchmark 2: Training Error of Learned Fields Across Time Grid")
        t_headers = [f"t={t:.2f}" for t in T_EVAL_GRID]
        print(f"| Model & Evaluated Field | {' | '.join(t_headers)} | Mean |")
        print(f"|:---|{'-|-'.join([':' + '-' * len(h) + ':' for h in t_headers])}|:---:|")
        for model_name, fm in b2.items():
            if any(e is not None for e in fm.get("error_b", [])):
                vals = [f"{e:.4f}" if e is not None else "N/A" for e in fm["error_b"]]
                valid_e = [e for e in fm["error_b"] if e is not None]
                mean_val = f"{np.mean(valid_e):.4f}" if valid_e else "N/A"
                print(f"| {model_name} (Velocity b̂) | {' | '.join(vals)} | {mean_val} |")
            if any(e is not None for e in fm.get("error_eta", [])):
                vals = [f"{e:.4f}" if e is not None else "N/A" for e in fm["error_eta"]]
                valid_e = [e for e in fm["error_eta"] if e is not None]
                mean_val = f"{np.mean(valid_e):.4f}" if valid_e else "N/A"
                print(f"| {model_name} (Denoiser η̂) | {' | '.join(vals)} | {mean_val} |")
            if any(e is not None for e in fm.get("error_s", [])):
                vals = [f"{e:.4f}" if e is not None else "N/A" for e in fm["error_s"]]
                valid_e = [e for e in fm["error_s"] if e is not None]
                mean_val = f"{np.mean(valid_e):.4f}" if valid_e else "N/A"
                print(f"| {model_name} (Score ŝ) | {' | '.join(vals)} | {mean_val} |")

    # 4. Benchmark 3 Table: Trained Models Generative Metrics Across Steps
    b3 = results.get("benchmark3_trained_models", {})
    if b3:
        print("\n### Benchmark 3: Trained Models Generative Metrics Across Integration Steps")
        b3_headers = [
            "Steps M",
            "Model Paradigm",
            "Transport",
            "\\lvert\\log Ẑ\\rvert",
            "ESS (%)",
            "W₂ Distance",
            "\\lvert P(M₁) - 2/3\\rvert",
        ]
        print(f"| {' | '.join(b3_headers)} |")
        print(f"|{'-|-'.join([':' + '-' * (len(h) - 2) + ':' for h in b3_headers])}|")
        for step_key, step_data in b3.items():
            steps_num = step_key.replace("steps_", "")
            for m_key, m_val in step_data.items():
                parts = m_key.split("_", 1)
                m_name = parts[0]
                t_kind = parts[1].upper() if len(parts) > 1 else ""
                row = [
                    f"{steps_num}",
                    f"{m_name}",
                    f"{t_kind}",
                    f"{m_val['abs_log_z']:.4f}",
                    f"{m_val['ess']:.1f}%",
                    f"{m_val['w2']:.4f}",
                    f"{m_val['mode_err']:.4f}",
                ]
                print(f"| {' | '.join(row)} |")

    print("\n" + "=" * 90 + "\n")


@app.command()
def main(
    ckpt_run1: Optional[Path] = typer.Option(None, help="Path to Run 1 checkpoint dir (ODE, gamma=0)"),
    ckpt_run2: Optional[Path] = typer.Option(None, help="Path to Run 2 checkpoint dir (ODE, gamma>0)"),
    ckpt_run3: Optional[Path] = typer.Option(None, help="Path to Run 3 checkpoint dir (Full SDE, gamma>0)"),
    ckpt_run4: Optional[Path] = typer.Option(None, help="Path to Run 4 checkpoint dir (One-Sided SDE, gamma=0)"),
    ckpt_run5: Optional[Path] = typer.Option(None, help="Path to Run 5 checkpoint dir (One-Sided TSM)"),
    ckpt_run6: Optional[Path] = typer.Option(None, help="Path to Run 6 checkpoint dir (Full SI TSM)"),
    benchmark_dir: Optional[Path] = typer.Option(
        Path("tmp/ckpt_benchmark"), help="Root directory containing benchmark checkpoints."
    ),
    steps_list: str = typer.Option("50,100,200", help="Comma-separated list of integration steps for SDE scaling."),
    num_samples: int = typer.Option(1000, help="Number of samples to generate per model."),
    batch_size: int = typer.Option(500, help="Batch size for sample generation."),
    seed: int = typer.Option(42, help="Evaluation random seed."),
    output_json: Optional[Path] = typer.Option(None, help="Optional path to save JSON results."),
    include_oracles: bool = typer.Option(
        True, "--include-oracles/--no-include-oracles", help="Include analytical oracles."
    ),
    boundary_threshold: float = typer.Option(
        1e-4, help="Time threshold for exact boundary score clamping (t <= threshold or t >= 1 - threshold)."
    ),
):
    """Run systematic comparison of generative sampling methods."""
    parsed_steps = [int(s.strip()) for s in steps_list.split(",") if s.strip()]

    results = run_comparison_suite(
        ckpt_run1=ckpt_run1,
        ckpt_run2=ckpt_run2,
        ckpt_run3=ckpt_run3,
        ckpt_run4=ckpt_run4,
        ckpt_run5=ckpt_run5,
        ckpt_run6=ckpt_run6,
        benchmark_dir=benchmark_dir,
        steps_list=parsed_steps,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        include_oracles=include_oracles,
        boundary_threshold=boundary_threshold,
    )
    print_comparison_tables(results)

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
        print(f"Results saved to {output_json}")


if __name__ == "__main__":
    app()
