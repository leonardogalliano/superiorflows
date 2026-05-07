"""Evaluate log-probabilities of pre-generated CNF samples.

Loads samples produced by ``sampling_particles.py`` (typically with
``--ignore-density``), optionally filters them by an energy quantile,
and evaluates ``Flow.log_prob`` in batches.  Output mirrors the
directory layout of ``sampling_particles.py``.
"""

import datetime
import json
import math
import time
from pathlib import Path

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import typer

from particle_systems.particle_system import (
    BoltzmannDistribution,
    ParticleSystem,
    TrajectoryDataSource,
    batch_to_trajectory,
)
from particle_systems.sampling_particles import load_trained_flow

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    ckpt_path: Path = typer.Argument(
        ...,
        help="Path to checkpoint directory (containing config.json & checkpoint)",
        exists=True,
        dir_okay=True,
    ),
    samples_path: Path = typer.Argument(
        ...,
        help="Directory containing sample trajectories (scanned recursively for samples.xyz)",
        exists=True,
        dir_okay=True,
    ),
    batch_size: int = typer.Option(64, help="Number of samples per log_prob evaluation call"),
    output_path: Path = typer.Option("tmp/evaluated_densities", help="Root output directory"),
    energy_fraction: float = typer.Option(None, help="Fraction of lowest-energy samples to keep (None = keep all)"),
    seed: int = typer.Option(0, help="Random seed"),
    device: str = typer.Option(None, help="JAX device: cpu | gpu"),
    solver: str = typer.Option(None, help="Solver type (euler, tsit5, dopri5)"),
    tolerance: float = typer.Option(None, help="Solver tolerance (sets both atol and rtol)"),
    solver_steps: int = typer.Option(None, help="Number of steps for fixed-step solvers"),
    hutchinson_samples: int = typer.Option(None, help="Number of Hutchinson samples for divergence estimation"),
    temperature: float = typer.Option(None, help="Temperature for energy filtering (overrides config)"),
):
    """Evaluate log-probabilities of pre-generated CNF particle samples."""
    if device is not None:
        jax.config.update("jax_platform_name", device)

    print(f"\n{'='*60}")
    print("Evaluating log-probabilities of pre-generated samples")
    print(f"  Checkpoint      : {ckpt_path}")
    print(f"  Samples path    : {samples_path}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Output path     : {output_path}")
    print(f"  Energy fraction : {energy_fraction}")
    print(f"  Seed            : {seed}")
    if solver:
        print(f"  Solver          : {solver}")
    if tolerance:
        print(f"  Tolerance       : {tolerance}")
    if solver_steps:
        print(f"  Solver steps    : {solver_steps}")
    if hutchinson_samples:
        print(f"  Hutchinson      : {hutchinson_samples}")
    if temperature:
        print(f"  Temperature     : {temperature}")
    print(f"  JAX process     : {jax.process_index()}/{jax.process_count()}")
    print(f"  JAX devices     : {jax.devices()}")
    print(f"{'='*60}\n")

    # ── Load trained flow ─────────────────────────────────────────────
    print("Loading trained model...")
    t0 = time.time()

    flow_kwargs = {}
    if solver:
        solvers = {"euler": dfx.Euler, "tsit5": dfx.Tsit5, "dopri5": dfx.Dopri5}
        if solver.lower() not in solvers:
            raise ValueError(f"Unknown solver '{solver}'. Available: {list(solvers)}")
        slv = solvers[solver.lower()]()
        flow_kwargs["solver"] = slv
        flow_kwargs["augmented_solver"] = slv
    if solver_steps is not None:
        flow_kwargs["stepsize_controller"] = dfx.ConstantStepSize()
        flow_kwargs["augmented_stepsize_controller"] = dfx.ConstantStepSize()
        flow_kwargs["dt0"] = 1.0 / solver_steps
    elif tolerance is not None:
        flow_kwargs["stepsize_controller"] = dfx.PIDController(rtol=tolerance, atol=tolerance)
        flow_kwargs["augmented_stepsize_controller"] = dfx.PIDController(rtol=tolerance, atol=tolerance)
    if hutchinson_samples is not None:
        flow_kwargs["hutchinson_samples"] = hutchinson_samples

    flow, N, d, L, composition = load_trained_flow(ckpt_path, **flow_kwargs)
    t1 = time.time()
    print(f"Loaded successfully in {t1 - t0:.1f}s. Model handles N={N}, d={d}, L={L:.4f}")

    # ── Load all samples ──────────────────────────────────────────────
    print("Loading samples...")
    source = TrajectoryDataSource(samples_path, filename="samples.xyz")
    n_total = len(source)
    print(f"Found {n_total} samples")

    # ── Optional energy-based filtering ───────────────────────────────
    if energy_fraction is not None:
        if not 0.0 < energy_fraction <= 1.0:
            raise ValueError(f"energy_fraction must be in (0, 1], got {energy_fraction}")

        config_file = ckpt_path / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)

        T = temperature if temperature is not None else config["data"].get("temperature")
        if T is None:
            raise ValueError(
                "Temperature is required for energy filtering. "
                "Provide via --temperature or ensure it is set in the checkpoint config."
            )

        model_file = config["data"].get("model_file")
        if model_file is None:
            raise ValueError("data.model_file is required in checkpoint config for energy filtering.")

        with open(model_file) as f:
            potential = json.load(f)

        target_dist = BoltzmannDistribution(N=N, d=d, L=L, temperature=T, model=potential, composition=composition)

        print(f"Computing energies for filtering (T={T})...")
        t_energy = time.time()
        positions = jnp.asarray(source[:].positions)
        species = jnp.asarray(source[:].species)
        energies = jax.vmap(target_dist._energy_fn)(positions, species)

        n_keep = int(n_total * energy_fraction)
        _, ids = jax.lax.top_k(-energies, n_keep)
        ids = jnp.sort(ids)
        filtered = source[np.asarray(ids)]
        print(f"Filtered {n_total} → {n_keep} samples in {time.time() - t_energy:.1f}s")
    else:
        filtered = source[:]
        n_keep = n_total

    # ── Prepare batches ───────────────────────────────────────────────
    n_batches = math.ceil(n_keep / batch_size)
    n_padded = n_batches * batch_size
    n_pad = n_padded - n_keep

    pos = np.asarray(filtered.positions)
    spec = np.asarray(filtered.species)
    box = np.asarray(filtered.box)

    if n_pad > 0:
        pad_pos = np.repeat(pos[:1], n_pad, axis=0)
        pad_spec = np.repeat(spec[:1], n_pad, axis=0)
        pad_box = np.repeat(box[:1], n_pad, axis=0)
        pos = np.concatenate([pos, pad_pos], axis=0)
        spec = np.concatenate([spec, pad_spec], axis=0)
        box = np.concatenate([box, pad_box], axis=0)

    pos_b = pos.reshape(n_batches, batch_size, N, d)
    spec_b = spec.reshape(n_batches, batch_size, N)
    box_b = box.reshape(n_batches, batch_size, d)

    systems = [
        ParticleSystem(jnp.asarray(pos_b[i]), jnp.asarray(spec_b[i]), jnp.asarray(box_b[i])) for i in range(n_batches)
    ]

    # ── JIT-compile log_prob ──────────────────────────────────────────
    key = jax.random.PRNGKey(seed)

    print("Precompiling JAX graph...")
    t_comp = time.time()

    @jax.jit
    def eval_log_prob(batch, rng):
        if hutchinson_samples is not None:
            return flow.log_prob(batch, key=rng)
        return flow.log_prob(batch)

    compiled_eval = eval_log_prob.lower(systems[0], key).compile()
    print(f"Compiled in {time.time() - t_comp:.1f}s")

    # ── Build output directory ────────────────────────────────────────
    s_name = type(flow.solver).__name__.lower()
    if isinstance(flow.stepsize_controller, dfx.PIDController):
        suffix = f"tol{flow.stepsize_controller.atol}"
    elif isinstance(flow.stepsize_controller, dfx.ConstantStepSize):
        dt0_val = getattr(flow, "dt0", None)
        if dt0_val is not None and dt0_val > 0:
            steps = int(round(1.0 / dt0_val))
            suffix = f"steps{steps}"
        else:
            suffix = "unknown"
    else:
        suffix = "custom"

    solver_tag = f"{s_name}_{suffix}"
    hutch_tag = f"_hutch{hutchinson_samples}" if hutchinson_samples is not None else ""
    frac_tag = f"_frac{energy_fraction}" if energy_fraction is not None else ""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = (
        output_path / ckpt_path.name / f"{solver_tag}{hutch_tag}{frac_tag}_b{batch_size}" / f"seed{seed}_{timestamp}"
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate and write on the fly ─────────────────────────────────
    import atooms.trajectory

    pos_out = np.asarray(filtered.positions)
    spec_out = np.asarray(filtered.species)
    box_out = np.asarray(filtered.box)

    print("Starting log-prob evaluation...")
    total_time = 0.0

    for i in range(n_batches):
        key, subkey = jax.random.split(key)
        print(f"[{i + 1}/{n_batches}] Evaluating batch... ", end="", flush=True)
        t_start = time.time()
        lp = compiled_eval(systems[i], subkey)
        lp.block_until_ready()
        t_batch = time.time() - t_start
        total_time += t_batch

        start = i * batch_size
        end = min(start + batch_size, n_keep)
        log_probs = np.asarray(lp)[: end - start]

        trajectory_dir = run_output_dir / str(i + 1)
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        batch_sys = ParticleSystem(
            positions=pos_out[start:end],
            species=spec_out[start:end],
            box=box_out[start:end],
        )
        trj = batch_to_trajectory(batch_sys)
        trj.metadata = {"generated_by": ckpt_path.name}

        out_xyz = trajectory_dir / "samples.xyz"
        with atooms.trajectory.TrajectoryXYZ(str(out_xyz), "w") as out:
            out.metadata = trj.metadata
            for sys in trj:
                out.write(sys)

        np.savetxt(trajectory_dir / "log_probs.dat", log_probs, fmt="%.6e")
        print(f"Done in {t_batch:.2f}s")

    print(f"\nAll done! Evaluated {n_keep} log-probs in {total_time:.1f}s.")
    print(f"Saved to {run_output_dir.resolve()}")


if __name__ == "__main__":
    app()
