"""Sample from a trained Gaussian mixture model using unified NeuralSampler.

Supports:
- Discrete normalising flows (RealNVP)
- Continuous ODE normalising flows (Flow / ODEBijector)
- Continuous SDE stochastic transports (NeuralSampler / SDETransport)
- Exact analytical oracles (oracle_ode / oracle_sde)
"""

import json
import time
from pathlib import Path
from typing import Optional

import diffrax as dfx
import distreqx.distributions as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import typer

from scripts.gaussian_mixture.analytical import make_oracle_bijector, make_oracle_sde_transport
from scripts.gaussian_mixture.train_gaussian_mixture import (
    NoiseScheduleFromRatio,
    OsmoticDriftFromDenoiser,
    OsmoticDriftFromGamma,
    RealNVPBijector,
    ReconstructedScoreFromDenoiser,
    ReconstructedVelocityFromDenoiser,
    ScoreFromGamma,
    VelocityDenoiserPair,
    build_louis_mixture,
    build_model,
    build_schedule_fn,
    build_solver,
    restore_evaluation_model,
)
from superiorflows import NeuralSampler, ODEBijector, SDETransport
from superiorflows.bijector import AbstractBijector

app = typer.Typer(pretty_exceptions_show_locals=False)


def load_trained_sampler(
    ckpt_path: Path,
    transport_type: str = "auto",
    oracle: bool = False,
    sde_steps: Optional[int] = None,
    noise_ratio: Optional[str] = None,
    **solver_overrides,
):
    """Load a trained model or oracle and wrap it into a unified NeuralSampler.

    Args:
        ckpt_path: Path to checkpoint directory containing config.json and weights.
        transport_type: One of 'auto', 'ode', 'sde', 'discrete', 'oracle_ode', 'oracle_sde'.
        oracle: If True, overrides neural model with exact analytical oracle.
        sde_steps: Optional override for Euler–Maruyama SDE integration steps.
        noise_ratio: Optional override for SDE noise ratio lambda(t) = eps(t)/alpha(t).
        **solver_overrides: Additional ODE solver keyword arguments.

    Returns:
        Tuple (sampler, target_dist, d, a, config, mode_name).
    """
    config_file = ckpt_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config.json in {ckpt_path}")

    with open(config_file, "r") as f:
        config = json.load(f)

    tgcfg = config["target"]
    d = tgcfg["d"]
    a = tgcfg["a"]
    sigma2_max = tgcfg.get("sigma2_max", 0.2)
    sigma2_min = tgcfg.get("sigma2_min", 0.01)
    weight = tgcfg.get("weight", 2.0 / 3.0)

    # Base and target distributions
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))
    target_dist = build_louis_mixture(d, a, sigma2_max, sigma2_min, weight)

    # Schedulers
    si_cfg = config.get("stochastic_interpolant", {})
    s_expr = si_cfg.get("interpolant_scheduler", "t")
    gamma_expr = si_cfg.get("noise_scheduler", "None")
    s_fn = build_schedule_fn(s_expr)
    gamma_fn = build_schedule_fn(gamma_expr)

    # Resolve oracle modes
    if oracle or transport_type in ("oracle_ode", "oracle_sde", "oracle"):
        if transport_type == "oracle_sde" or (
            transport_type == "oracle" and config.get("sde", {}).get("steps") is not None
        ):
            sde_cfg = config.get("sde", {})
            ratio_expr = noise_ratio or sde_cfg.get("noise_ratio")
            steps = sde_steps or sde_cfg.get("steps", 100)
            if ratio_expr and str(ratio_expr).lower() not in ("none", "null"):
                ratio_fn = build_schedule_fn(ratio_expr)

                def oracle_scale(t):
                    return 1.0 - (s_fn(t) if s_fn is not None else t)

                noise_sched = NoiseScheduleFromRatio(ratio_fn, oracle_scale)
                transport = make_oracle_sde_transport(
                    d, a, noise_sched, sigma2_max, sigma2_min, weight, s_fn, gamma_fn, num_steps=steps
                )
                mode_name = f"Oracle SDE (ratio={ratio_expr}, steps={steps})"
            else:
                noise_expr = sde_cfg.get("noise_schedule", "0.1 * (1-t) * t")
                noise_sched = build_schedule_fn(noise_expr)
                transport = make_oracle_sde_transport(
                    d, a, noise_sched, sigma2_max, sigma2_min, weight, s_fn, gamma_fn, num_steps=steps
                )
                mode_name = f"Oracle SDE (steps={steps})"
        else:
            base_solver_kwargs = build_solver(config)
            base_solver_kwargs.update(solver_overrides)
            transport = make_oracle_bijector(d, a, sigma2_max, sigma2_min, weight, s_fn, gamma_fn, **base_solver_kwargs)
            mode_name = "Oracle ODE"

        sampler = NeuralSampler(transport=transport, base_distribution=base_dist)
        return sampler, target_dist, d, a, config, mode_name

    # Restore neural network from checkpoint
    key = jax.random.key(0)
    model_template = build_model(config, d, key=key)
    checkpointer = ocp.CheckpointManager(ckpt_path.resolve(), item_names=("model", "optimizer", "metadata"))
    step = checkpointer.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found at {ckpt_path}")

    trained_model = restore_evaluation_model(checkpointer, step, model_template, config)

    # 1. Discrete Bijector (RealNVP)
    if isinstance(trained_model, AbstractBijector) or isinstance(trained_model, RealNVPBijector):
        if transport_type not in ("auto", "discrete", "ode"):
            raise ValueError(
                f"Trained model is a discrete RealNVP bijector; cannot use transport_type='{transport_type}'"
            )
        transport = trained_model
        mode_name = "Discrete RealNVP Flow"

    # 2. Continuous SDE Transport
    elif transport_type == "sde":
        denoiser_only = si_cfg.get("denoiser_only", False)
        learn_denoiser = si_cfg.get("learn_denoiser", False)
        sde_cfg = config.get("sde", {})
        steps = sde_steps or sde_cfg.get("steps", 100)
        ratio_expr = noise_ratio or sde_cfg.get("noise_ratio")

        if ratio_expr and str(ratio_expr).lower() not in ("none", "null"):
            ratio_fn = build_schedule_fn(ratio_expr)

            def default_scale(t):
                return 1.0 - (s_fn(t) if s_fn is not None else t)

            if denoiser_only:
                vf = ReconstructedVelocityFromDenoiser(trained_model, s_fn)
                noise_sched = NoiseScheduleFromRatio(ratio_fn, default_scale)
                osmotic_u = OsmoticDriftFromDenoiser(trained_model, ratio_fn)
            elif learn_denoiser and isinstance(trained_model, VelocityDenoiserPair):
                vf = trained_model.velocity_field
                scale_fn = gamma_fn if gamma_fn is not None else default_scale
                noise_sched = NoiseScheduleFromRatio(ratio_fn, scale_fn)
                osmotic_u = OsmoticDriftFromGamma(trained_model.denoiser, ratio_fn)
            else:
                raise ValueError("noise_ratio requires a denoiser model (denoiser_only=True or learn_denoiser=True).")

            transport = SDETransport(
                velocity_field=vf,
                noise_schedule=noise_sched,
                osmotic_drift=osmotic_u,
                num_steps=steps,
            )
            mode_name = f"Continuous SDE (ratio={ratio_expr}, steps={steps})"
        else:
            if denoiser_only:
                vf = ReconstructedVelocityFromDenoiser(trained_model, s_fn)
                sf = ReconstructedScoreFromDenoiser(trained_model, s_fn)
            elif learn_denoiser and isinstance(trained_model, VelocityDenoiserPair):
                vf = trained_model.velocity_field
                if gamma_fn is not None:
                    sf = ScoreFromGamma(trained_model.denoiser, gamma_fn)
                else:
                    sf = ReconstructedScoreFromDenoiser(trained_model.denoiser, s_fn)
            else:
                raise ValueError(
                    "SDE sampling requires a score field or noise_ratio. The model was trained with velocity-only. "
                    "Train with denoiser_only=True or learn_denoiser=True, or use --transport ode."
                )

            noise_expr = sde_cfg.get("noise_schedule", "0.1 * (1-t) * t")
            noise_sched = build_schedule_fn(noise_expr)

            transport = SDETransport(
                velocity_field=vf,
                score_field=sf,
                noise_schedule=noise_sched,
                num_steps=steps,
            )
            mode_name = f"Continuous SDE (steps={steps})"

    # 3. Continuous ODE Bijector (Default for continuous models)
    else:
        denoiser_only = si_cfg.get("denoiser_only", False)
        if denoiser_only:
            vf = ReconstructedVelocityFromDenoiser(trained_model, s_fn)
        elif isinstance(trained_model, VelocityDenoiserPair):
            vf = trained_model.velocity_field
        elif hasattr(trained_model, "velocity_field"):
            vf = trained_model.velocity_field
        else:
            vf = trained_model

        base_solver_kwargs = build_solver(config)
        base_solver_kwargs.update(solver_overrides)
        transport = ODEBijector(vf, **base_solver_kwargs)
        mode_name = "Continuous ODE Flow"

    sampler = NeuralSampler(transport=transport, base_distribution=base_dist)
    return sampler, target_dist, d, a, config, mode_name


@app.command()
def main(
    ckpt_path: Path = typer.Argument(
        ...,
        help="Path to checkpoint directory (containing config.json & checkpoint)",
        exists=True,
        dir_okay=True,
    ),
    num_samples: int = typer.Option(1000, help="Total number of samples to generate"),
    batch_size: int = typer.Option(1000, help="Number of samples per batch"),
    output_path: Path = typer.Option("tmp/sampled_gaussian_mixture.npz", help="Path to save the .npz output"),
    transport: str = typer.Option("auto", help="Transport mode: auto | ode | sde | discrete | oracle_ode | oracle_sde"),
    oracle: bool = typer.Option(
        False, "--oracle", help="If True, evaluate analytical ground-truth dynamics instead of checkpoint weights."
    ),
    sde_steps: Optional[int] = typer.Option(None, help="Number of steps for SDE Euler–Maruyama integration"),
    noise_ratio: Optional[str] = typer.Option(
        None, help="Override SDE noise ratio lambda(t) = eps(t)/alpha(t) (eliminates boundary division)."
    ),
    seed: int = typer.Option(0, help="Random seed for sample generation"),
    device: Optional[str] = typer.Option(None, help="JAX device: cpu | gpu"),
    solver: Optional[str] = typer.Option(None, help="ODE solver type (euler, tsit5, dopri5)"),
    tolerance: Optional[float] = typer.Option(None, help="ODE solver tolerance (sets both atol and rtol)"),
    solver_steps: Optional[int] = typer.Option(None, help="Number of steps for fixed-step ODE solvers"),
    hutchinson_samples: Optional[int] = typer.Option(
        None, help="Number of Hutchinson samples for ODE divergence estimation."
    ),
    ignore_density: bool = typer.Option(
        False, "--ignore-density", help="If True, only sample configurations, skipping log-weight computation."
    ),
):
    """Generate new samples from a trained Gaussian mixture model using NeuralSampler."""
    if device is not None:
        jax.config.update("jax_platform_name", device)

    print(f"\n{'=' * 60}")
    print("Sampling from Gaussian Mixture model (NeuralSampler)")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Total samples  : {num_samples}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Transport mode : {transport} {'(oracle)' if oracle else ''}")
    print(f"  Output path    : {output_path}")
    print(f"  Seed           : {seed}")
    if solver:
        print(f"  ODE Solver     : {solver}")
    if tolerance:
        print(f"  ODE Tolerance  : {tolerance}")
    if solver_steps:
        print(f"  ODE Steps      : {solver_steps}")
    if sde_steps:
        print(f"  SDE Steps      : {sde_steps}")
    if hutchinson_samples:
        print(f"  Hutchinson     : {hutchinson_samples}")
    print(f"  Ignore density : {ignore_density}")
    print(f"  JAX process    : {jax.process_index()}/{jax.process_count()}")
    print(f"  JAX devices    : {jax.devices()}")
    print(f"{'=' * 60}\n")

    print("Loading model and building NeuralSampler...")
    t0 = time.time()

    solver_overrides = {}
    if solver:
        solvers = {"euler": dfx.Euler, "tsit5": dfx.Tsit5, "dopri5": dfx.Dopri5}
        if solver.lower() not in solvers:
            raise ValueError(f"Unknown solver '{solver}'. Available: {list(solvers)}")
        slv = solvers[solver.lower()]()
        solver_overrides["solver"] = slv
        solver_overrides["augmented_solver"] = slv
    if solver_steps is not None:
        solver_overrides["stepsize_controller"] = dfx.ConstantStepSize()
        solver_overrides["augmented_stepsize_controller"] = dfx.ConstantStepSize()
        solver_overrides["dt0"] = 1.0 / solver_steps
    elif tolerance is not None:
        solver_overrides["stepsize_controller"] = dfx.PIDController(rtol=tolerance, atol=tolerance)
        solver_overrides["augmented_stepsize_controller"] = dfx.PIDController(rtol=tolerance, atol=tolerance)
    if hutchinson_samples is not None:
        solver_overrides["hutchinson_samples"] = hutchinson_samples

    sampler, target_dist, d, a, config, mode_name = load_trained_sampler(
        ckpt_path=ckpt_path,
        transport_type=transport.lower(),
        oracle=oracle,
        sde_steps=sde_steps,
        noise_ratio=noise_ratio,
        **solver_overrides,
    )
    t1 = time.time()
    print(f"Ready in {t1 - t0:.1f}s. Mode: {mode_name} | Dimension: d={d}")

    key = jax.random.key(seed)

    print("Precompiling JAX graph...")
    t_comp = time.time()

    @eqx.filter_jit
    def sample_batch(rng, current_batch_size):
        sample_keys = jax.random.split(rng, current_batch_size)
        x0 = jax.vmap(sampler.base_distribution.sample)(sample_keys)

        if ignore_density:
            x1 = jax.vmap(sampler.transport.forward)(x0, key=sample_keys)
            return x1, None, None, None

        x1, log_weight = jax.vmap(lambda x, k: sampler.push_forward_and_log_weight(x, key=k))(x0, sample_keys)
        target_log_prob = jax.vmap(target_dist.log_prob)(x1)
        log_iw = target_log_prob - log_weight
        return x1, log_weight, target_log_prob, log_iw

    print(f"Compiled in {time.time() - t_comp:.1f}s")
    print("Starting generation...")

    total_time = 0.0
    all_samples = []
    all_log_weights = []
    all_target_log_probs = []
    all_log_iws = []

    num_batches = (num_samples + batch_size - 1) // batch_size
    samples_generated = 0

    for i in range(num_batches):
        key, subkey = jax.random.split(key)
        current_batch_size = min(batch_size, num_samples - samples_generated)

        print(f"  [{i + 1}/{num_batches}] Generating {current_batch_size} samples... ", end="", flush=True)
        t_start = time.time()

        samples, log_weights, target_log_probs, log_iws = sample_batch(subkey, current_batch_size)
        samples.block_until_ready()
        if log_weights is not None:
            log_weights.block_until_ready()

        t_batch = time.time() - t_start
        total_time += t_batch
        print(f"Done in {t_batch:.2f}s")

        all_samples.append(np.asarray(samples))
        if log_weights is not None:
            all_log_weights.append(np.asarray(log_weights))
            all_target_log_probs.append(np.asarray(target_log_probs))
            all_log_iws.append(np.asarray(log_iws))

        samples_generated += current_batch_size

    final_samples = np.concatenate(all_samples, axis=0)

    save_dict = {"samples": final_samples}
    if not ignore_density:
        final_log_weights = np.concatenate(all_log_weights, axis=0)
        final_target_log_probs = np.concatenate(all_target_log_probs, axis=0)
        final_log_iws = np.concatenate(all_log_iws, axis=0)

        save_dict["log_weights"] = final_log_weights
        save_dict["target_log_probs"] = final_target_log_probs
        save_dict["importance_weights"] = final_log_iws

        # Diagnostics
        max_log_w = np.max(final_log_iws)
        weights_unnorm = np.exp(final_log_iws - max_log_w)
        ess = (np.sum(weights_unnorm) ** 2) / (len(weights_unnorm) * np.sum(weights_unnorm**2))
        z_hat = np.mean(np.exp(final_log_iws))

        print("\nImportance Sampling Diagnostics:")
        print(f"  Mean exp(log_w) (Z_hat) : {z_hat:.4f} (ideal: 1.0000)")
        print(f"  Effective Sample Size   : {ess * 100:.2f}%")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, **save_dict)

    print(f"\nAll done! Generated {num_samples} samples in {total_time:.1f}s.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    app()
