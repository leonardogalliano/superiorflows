"""Sample patch updates and evaluate acceptance ratios from a trained CNF model."""

import datetime
import json
import time
from pathlib import Path
from typing import Any

import diffrax as dfx
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import typer

from particle_systems.particle_system import (
    ParticleSystem,
    TrajectoryDataSource,
    batch_to_trajectory,
)
from particle_systems.patch_updater import PatchUpdater
from particle_systems.potentials import build_masked_energy_fn, build_masked_local_energy_fn
from particle_systems.selection import (
    check_patch_remainder_separation,
    random_point_select,
    soft_log_pi_ratio,
    torus_distances_from_point,
)
from particle_systems.training_patches import build_solver_patch, build_velocity_patch
from superiorflows import ODEBijector

app = typer.Typer(pretty_exceptions_show_locals=False)


def load_trained_patch_updater(
    ckpt_path: Path,
    data_path: Path | None = None,
    step: int | None = None,
    solver: str | None = None,
    tolerance: float | None = None,
    solver_steps: int | None = None,
    hutchinson_samples: int | None = None,
    r_patch: float | None = None,
    r_context: float | None = None,
    sigma_r: float | None = None,
    sigma_base: float | None = None,
    max_dataset_length: int | None = None,
):
    """Load a trained PatchUpdater and associated metadata from a checkpoint directory.

    Loads the dataset once as the source material for patch updates, restores
    the velocity field weights from the Orbax checkpoint, configures solvers and
    selection protocols, and sets up masked potential energy callables.
    """
    config_file = ckpt_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config.json in {ckpt_path}")

    with open(config_file, "r") as f:
        config = json.load(f)

    # 1. Dataset loading
    resolved_data_path = Path(data_path) if data_path is not None else Path(config["data"]["data_path"])
    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"Dataset path '{resolved_data_path}' does not exist. "
            "Pass --data-path on the CLI to specify a valid trajectory path."
        )

    source = TrajectoryDataSource(resolved_data_path, max_frames=max_dataset_length)
    N, d = source.N, source.d
    L = float(source.box_size[0])
    ref_species = source[0].species
    n_species = int(jnp.max(jnp.asarray(ref_species)) + 1)
    composition = np.bincount(ref_species, minlength=n_species) / len(ref_species)

    # 2. Patch parameters
    pcfg = config["patch"]
    N_patch = int(pcfg["N_patch"])
    N_context = int(pcfg["N_context"])
    r_patch_val = float(r_patch) if r_patch is not None else float(pcfg["r_patch"])
    r_context_val = float(r_context) if r_context is not None else float(pcfg["r_context"])
    sigma_r_val = float(sigma_r) if sigma_r is not None else float(pcfg.get("sigma_r", 0.0))
    sigma_base_val = float(sigma_base) if sigma_base is not None else float(pcfg.get("sigma_base", 1.0))

    # 3. Model construction and restoration
    key = jax.random.key(0)
    velocity_field = build_velocity_patch(config, N_patch, N_context, d, n_species, key=key)

    model_params = eqx.filter(velocity_field, eqx.is_array)
    static_model = eqx.filter(velocity_field, eqx.is_array, inverse=True)

    checkpointer = ocp.CheckpointManager(ckpt_path.resolve(), item_names=("model", "optimizer", "metadata"))
    if step is None:
        step = checkpointer.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found at {ckpt_path}")

    restore_args = ocp.args.Composite(model=ocp.args.StandardRestore(model_params))
    restored = checkpointer.restore(step, args=restore_args)
    trained_velocity_field = eqx.combine(restored.model, static_model)

    # 4. Solver configuration
    bijector_kwargs = build_solver_patch(config)
    if solver:
        solvers = {"euler": dfx.Euler, "tsit5": dfx.Tsit5, "dopri5": dfx.Dopri5}
        stype = solver.lower()
        if stype not in solvers:
            raise ValueError(f"Unknown solver '{solver}'. Available: {list(solvers)}")
        slv = solvers[stype]()
        bijector_kwargs["solver"] = slv
        bijector_kwargs["augmented_solver"] = slv

    if solver_steps is not None:
        bijector_kwargs["stepsize_controller"] = dfx.ConstantStepSize()
        bijector_kwargs["augmented_stepsize_controller"] = dfx.ConstantStepSize()
        bijector_kwargs["dt0"] = 1.0 / solver_steps
    elif tolerance is not None:
        bijector_kwargs["stepsize_controller"] = dfx.PIDController(rtol=tolerance, atol=tolerance)
        bijector_kwargs["augmented_stepsize_controller"] = dfx.PIDController(rtol=tolerance, atol=tolerance)

    if hutchinson_samples is not None:
        bijector_kwargs["hutchinson_samples"] = hutchinson_samples

    bijector = ODEBijector(trained_velocity_field, **bijector_kwargs)

    # 5. Selection protocol
    selection_fn = random_point_select(
        N_patch=N_patch,
        N_context=N_context,
        r_patch=r_patch_val,
        r_context=r_context_val,
        sigma_r=sigma_r_val,
    )

    patch_updater = PatchUpdater(
        bijector=bijector,
        selection_fn=selection_fn,
        N_patch=N_patch,
        N_context=N_context,
        sigma=sigma_base_val,
        L=L,
        d=d,
    )

    # 6. Potential and energy functions
    model_file = config["data"].get("model_file")
    temperature = config["data"].get("temperature")
    masked_local_energy_fn = None
    masked_energy_fn = None
    max_rcut = None

    if model_file is not None:
        mpath = Path(model_file)
        if not mpath.exists():
            raise FileNotFoundError(f"Model file '{mpath}' not found.")
        with open(mpath) as f:
            boltzmann_model = json.load(f)

        box_arr = jnp.full((d,), L)
        masked_local_energy_fn = build_masked_local_energy_fn(boltzmann_model, box_arr, n_species, N_patch=N_patch)
        masked_energy_fn = build_masked_energy_fn(boltzmann_model, box_arr, n_species, N_patch=N_patch)

        max_rcut = max(
            float(np.max(c["parameters"]["rcut"]))
            if isinstance(c["parameters"]["rcut"], list)
            else float(c["parameters"]["rcut"])
            for c in boltzmann_model["cutoff"]
        )

    metadata = {
        "N": N,
        "d": d,
        "L": L,
        "n_species": n_species,
        "composition": composition,
        "N_patch": N_patch,
        "N_context": N_context,
        "r_patch": r_patch_val,
        "r_context": r_context_val,
        "sigma_r": sigma_r_val,
        "sigma_base": sigma_base_val,
        "temperature": temperature,
        "max_rcut": max_rcut,
        "step": step,
        "config": config,
    }

    return patch_updater, source, masked_local_energy_fn, masked_energy_fn, metadata


# ── JIT-compiled batch evaluations ──────────────────────────────────────────


@eqx.filter_jit
def evaluate_patch_batch_with_density(
    patch_updater: PatchUpdater,
    batch: ParticleSystem,
    subkeys: jax.Array,
    masked_local_energy_fn,
    temperature: float,
    r_mean: float,
    sigma_r: float,
    r_cut: float,
):
    """Propose patch updates, evaluate exact densities, and compute acceptance probabilities.

    Args:
        patch_updater: Instantiated PatchUpdater.
        batch: Batched ParticleSystem of shape (B, N, d).
        subkeys: Array of PRNG keys of shape (B, ...).
        masked_local_energy_fn: Callable computing local interaction energy U_loc.
        temperature: System temperature T.
        r_mean: Mean selection radius.
        sigma_r: Selection radius standard deviation.
        r_cut: Potential interaction cutoff radius.

    Returns:
        (systems_new, sels, metrics_dict)
    """
    # 1. Propose new systems and compute forward flow log-density log q(x'_p | x_c)
    systems_new, log_qs, sels = jax.vmap(patch_updater.update_and_log_prob)(batch, subkeys)

    # 2. Evaluate backward flow log-density log q(x_p | x_c) on initial patch
    log_q_bwds = jax.vmap(patch_updater.log_prob)(batch, sels)
    log_q_ratios = log_q_bwds - log_qs

    # 3. Selection probability ratio log pi(p, c | x') - log pi(p, c | x)
    dists_before = jax.vmap(torus_distances_from_point)(sels.anchor, batch.positions, batch.box)
    dists_after = jax.vmap(torus_distances_from_point)(sels.anchor, systems_new.positions, systems_new.box)

    log_pi_ratios = jax.vmap(soft_log_pi_ratio, in_axes=(0, 0, 0, 0, 0, 0, None, None))(
        dists_before,
        dists_after,
        sels.patch_idx,
        sels.context_idx,
        sels.n_patch,
        sels.n_context,
        r_mean,
        sigma_r,
    )

    # 4. Local energy difference: U_loc(x'_p, x_c) - U_loc(x_p, x_c)
    active_old = jax.vmap(patch_updater._build_subsystem)(batch, sels)
    active_new = jax.vmap(patch_updater._build_subsystem)(systems_new, sels)

    u_loc_old = jax.vmap(masked_local_energy_fn)(active_old.positions, active_old.species, sels.n_patch, sels.n_context)
    u_loc_new = jax.vmap(masked_local_energy_fn)(active_new.positions, active_new.species, sels.n_patch, sels.n_context)
    delta_U = u_loc_new - u_loc_old
    delta_U_T = delta_U / temperature

    # 5. Safety check: patch must remain strictly separated from remainder
    pos_patch_new = jax.vmap(lambda pos, idx: pos[idx])(systems_new.positions, sels.patch_idx)
    is_safe = jax.vmap(check_patch_remainder_separation, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
        batch.positions,
        pos_patch_new,
        sels.patch_idx,
        sels.context_idx,
        sels.n_patch,
        sels.n_context,
        batch.box,
        r_cut,
    )

    # 6. Exact acceptance probability in log domain
    log_alpha = -delta_U_T + log_pi_ratios + log_q_ratios
    log_alpha = jnp.where(is_safe, log_alpha, -jnp.inf)
    accept_probs = jnp.exp(jnp.minimum(log_alpha, 0.0))

    metrics = {
        "accept_probs": accept_probs,
        "is_safe": is_safe,
        "log_alpha": log_alpha,
        "log_pi_ratios": log_pi_ratios,
        "log_q_ratios": log_q_ratios,
        "delta_U_T": delta_U_T,
        "u_loc_old": u_loc_old,
        "u_loc_new": u_loc_new,
        "log_q_forward": log_qs,
        "log_q_backward": log_q_bwds,
    }
    return systems_new, sels, metrics


@eqx.filter_jit
def evaluate_patch_batch_fast(
    patch_updater: PatchUpdater,
    batch: ParticleSystem,
    subkeys: jax.Array,
    masked_local_energy_fn,
    temperature: float,
    r_cut: float,
):
    """Fast patch proposal generation via plain ODE integration and local energy evaluation.

    Bypasses augmented ODE divergence integration (~500x faster), evaluating local energies
    U_loc(x_p, x_c) and U_loc(x'_p, x_c) for extensive statistical sampling.

    Args:
        patch_updater: Instantiated PatchUpdater.
        batch: Batched ParticleSystem of shape (B, N, d).
        subkeys: Array of PRNG keys of shape (B, ...).
        masked_local_energy_fn: Callable computing local interaction energy U_loc.
        temperature: System temperature T.
        r_cut: Potential interaction cutoff radius.

    Returns:
        (systems_new, sels, metrics_dict)
    """
    # 1. Propose new systems via fast forward ODE
    systems_new, sels = jax.vmap(patch_updater.update)(batch, subkeys)

    # 2. Local interaction energies
    active_old = jax.vmap(patch_updater._build_subsystem)(batch, sels)
    active_new = jax.vmap(patch_updater._build_subsystem)(systems_new, sels)

    u_loc_old = jax.vmap(masked_local_energy_fn)(active_old.positions, active_old.species, sels.n_patch, sels.n_context)
    u_loc_new = jax.vmap(masked_local_energy_fn)(active_new.positions, active_new.species, sels.n_patch, sels.n_context)
    delta_U = u_loc_new - u_loc_old
    delta_U_T = delta_U / temperature

    # 3. Safety check
    pos_patch_new = jax.vmap(lambda pos, idx: pos[idx])(systems_new.positions, sels.patch_idx)
    is_safe = jax.vmap(check_patch_remainder_separation, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
        batch.positions,
        pos_patch_new,
        sels.patch_idx,
        sels.context_idx,
        sels.n_patch,
        sels.n_context,
        batch.box,
        r_cut,
    )

    metrics = {
        "u_loc_old": u_loc_old,
        "u_loc_new": u_loc_new,
        "delta_U_T": delta_U_T,
        "is_safe": is_safe,
    }
    return systems_new, sels, metrics


# ── Statistics helpers ───────────────────────────────────────────────────────


def _calc_stats(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "count": int(len(finite)),
    }


def _save_trajectory_xyz(filepath: Path, systems: ParticleSystem, metadata: dict):
    import atooms.trajectory

    trj = batch_to_trajectory(systems)
    trj.metadata = metadata
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with atooms.trajectory.TrajectoryXYZ(str(filepath), "w") as out:
        out.metadata = metadata
        for sys in trj:
            out.write(sys)


# ── Main CLI command ─────────────────────────────────────────────────────────


@app.command()
def main(
    ckpt_path: Path = typer.Argument(
        ...,
        help="Path to checkpoint directory (containing config.json and checkpoint)",
        exists=True,
        dir_okay=True,
    ),
    batch_size: int = typer.Option(64, help="Number of samples per batch (trajectory) for fast no-density mode"),
    num_trajectories: int = typer.Option(10, help="Number of trajectories (batches) for fast no-density mode"),
    density_batch_size: int = typer.Option(8, help="Number of samples per batch for density & acceptance mode"),
    density_trajectories: int = typer.Option(10, help="Number of trajectories for density & acceptance mode"),
    output_path: Path = typer.Option("tmp/sampled_patches", help="Directory where outputs will be stored"),
    data_path: Path = typer.Option(None, help="Optional override path to dataset trajectory file"),
    step: int = typer.Option(None, help="Optional checkpoint step number to restore (defaults to latest)"),
    seed: int = typer.Option(0, help="Random seed for generation"),
    device: str = typer.Option(None, help="JAX device: cpu | gpu"),
    solver: str = typer.Option(None, help="Solver type: euler | tsit5 | dopri5"),
    tolerance: float = typer.Option(None, help="Solver tolerance (sets atol and rtol)"),
    solver_steps: int = typer.Option(None, help="Number of steps for fixed-step solvers"),
    hutchinson_samples: int = typer.Option(None, help="Number of Hutchinson samples for divergence estimation"),
    ignore_density: bool = typer.Option(False, "--ignore-density", help="If True, skips density & acceptance mode"),
    ignore_fast: bool = typer.Option(False, "--ignore-fast", help="If True, skips fast no-density mode"),
    max_dataset_length: int = typer.Option(None, help="Max frames to load from the dataset trajectory"),
):
    """Generate patch update proposals and evaluate acceptance ratios from a trained CNF model."""
    if device is not None:
        jax.config.update("jax_platform_name", device)

    if ignore_density:
        density_trajectories = 0
    if ignore_fast:
        num_trajectories = 0

    if num_trajectories == 0 and density_trajectories == 0:
        raise ValueError("Both fast mode and density mode are disabled. Nothing to generate.")

    print(f"\n{'=' * 65}")
    print("Sampling Patch Updates and Acceptance Evaluation")
    print(f"  Checkpoint             : {ckpt_path}")
    print(f"  Fast trajectories      : {num_trajectories} (batch size {batch_size})")
    print(f"  Density trajectories   : {density_trajectories} (batch size {density_batch_size})")
    print(f"  Output path            : {output_path}")
    print(f"  Seed                   : {seed}")
    if solver:
        print(f"  Solver                 : {solver}")
    if tolerance:
        print(f"  Tolerance              : {tolerance}")
    if solver_steps:
        print(f"  Solver steps           : {solver_steps}")
    if hutchinson_samples:
        print(f"  Hutchinson samples     : {hutchinson_samples}")
    print(f"  JAX process            : {jax.process_index()}/{jax.process_count()}")
    print(f"  JAX devices            : {jax.devices()}")
    print(f"{'=' * 65}\n")

    print("Loading trained patch updater and dataset in single pass...")
    t0 = time.time()
    patch_updater, source, masked_local_energy_fn, masked_energy_fn, meta = load_trained_patch_updater(
        ckpt_path=ckpt_path,
        data_path=data_path,
        step=step,
        solver=solver,
        tolerance=tolerance,
        solver_steps=solver_steps,
        hutchinson_samples=hutchinson_samples,
        max_dataset_length=max_dataset_length,
    )
    t_load = time.time() - t0
    N, d, L = meta["N"], meta["d"], meta["L"]
    N_patch, N_context = meta["N_patch"], meta["N_context"]
    temperature = meta["temperature"]
    max_rcut = meta["max_rcut"]
    r_patch = meta["r_patch"]
    sigma_r = meta["sigma_r"]
    print(f"Loaded in {t_load:.2f}s.")
    print(f"System: N={N}, d={d}, L={L:.4f} | Patch: N_patch={N_patch}, N_context={N_context}, T={temperature}\n")

    if masked_local_energy_fn is None or temperature is None or max_rcut is None:
        raise ValueError(
            "Energy function, temperature, or r_cut could not be loaded. "
            "Ensure model_file and temperature are specified in config or dataset."
        )

    # Build output directory
    s_name = type(patch_updater.bijector.solver).__name__.lower()
    if isinstance(patch_updater.bijector.stepsize_controller, dfx.PIDController):
        ctrl_tag = f"tol{patch_updater.bijector.stepsize_controller.atol}"
    elif isinstance(patch_updater.bijector.stepsize_controller, dfx.ConstantStepSize):
        dt0 = getattr(patch_updater.bijector, "dt0", None)
        ctrl_tag = f"steps{int(round(1.0 / dt0))}" if dt0 is not None and dt0 > 0 else "const"
    else:
        ctrl_tag = "custom"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = (
        output_path
        / ckpt_path.name
        / f"{s_name}_{ctrl_tag}_fastM{num_trajectories}b{batch_size}_densM{density_trajectories}b{density_batch_size}"
        / f"seed{seed}_{timestamp}"
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.key(seed)
    read_options = grain.ReadOptions(num_threads=1, prefetch_buffer_size=2)
    summary_data: dict[str, Any] = {
        "metadata": {
            "checkpoint": ckpt_path.name,
            "step": meta["step"],
            "N": N,
            "d": d,
            "L": L,
            "N_patch": N_patch,
            "N_context": N_context,
            "r_patch": r_patch,
            "r_context": meta["r_context"],
            "sigma_r": sigma_r,
            "sigma_base": meta["sigma_base"],
            "temperature": temperature,
            "max_rcut": max_rcut,
            "solver": s_name,
            "stepsize_controller": ctrl_tag,
        },
        "timings": {"load_time": t_load},
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Fast Mode (No-Density forward ODE)
    # ─────────────────────────────────────────────────────────────────────────
    if num_trajectories > 0:
        print(f"Starting FAST generation ({num_trajectories} batches of {batch_size})...")
        fast_ds = grain.MapDataset.source(source).repeat().batch(batch_size, drop_remainder=True)
        fast_iter = iter(fast_ds.to_iter_dataset(read_options))

        # JIT Warmup
        warmup_batch = next(fast_iter)
        warmup_subkeys = jax.random.split(jax.random.key(999), batch_size)
        print("  Compiling fast ODE graph...", end="", flush=True)
        t_comp = time.time()
        _ = evaluate_patch_batch_fast(
            patch_updater, warmup_batch, warmup_subkeys, masked_local_energy_fn, temperature, max_rcut
        )
        print(f" compiled in {time.time() - t_comp:.2f}s")

        fast_u_loc_old = []
        fast_u_loc_new = []
        fast_delta_U_T = []
        fast_safe_rates = []
        total_fast_time = 0.0

        for i in range(1, num_trajectories + 1):
            batch = next(fast_iter)
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, batch_size)

            t_start = time.time()
            systems_new, sels, metrics = evaluate_patch_batch_fast(
                patch_updater, batch, subkeys, masked_local_energy_fn, temperature, max_rcut
            )
            # Ensure completion across async device boundaries
            systems_new.positions.block_until_ready()
            t_batch = time.time() - t_start
            total_fast_time += t_batch

            u_old = np.asarray(metrics["u_loc_old"])
            u_new = np.asarray(metrics["u_loc_new"])
            du_t = np.asarray(metrics["delta_U_T"])
            safe = np.asarray(metrics["is_safe"])

            fast_u_loc_old.append(u_old)
            fast_u_loc_new.append(u_new)
            fast_delta_U_T.append(du_t)
            fast_safe_rates.append(float(np.mean(safe)))

            # Save batch files
            batch_dir = run_output_dir / "fast" / str(i)
            batch_dir.mkdir(parents=True, exist_ok=True)

            _save_trajectory_xyz(
                batch_dir / "samples.xyz", systems_new, {"checkpoint": ckpt_path.name, "type": "proposal"}
            )
            _save_trajectory_xyz(
                batch_dir / "original_samples.xyz", batch, {"checkpoint": ckpt_path.name, "type": "original"}
            )

            np.savez_compressed(
                batch_dir / "metrics.npz",
                u_loc_old=u_old,
                u_loc_new=u_new,
                delta_U_T=du_t,
                is_safe=safe,
                anchor=np.asarray(sels.anchor),
                radius=np.asarray(sels.radius),
                patch_idx=np.asarray(sels.patch_idx),
                context_idx=np.asarray(sels.context_idx),
                n_patch=np.asarray(sels.n_patch),
                n_context=np.asarray(sels.n_context),
            )

            print(
                f"  [Fast {i:03d}/{num_trajectories:03d}] {t_batch:.2f}s | "
                f"u_old: {np.mean(u_old):.2f} | u_new (med): {np.median(u_new):.2f} | "
                f"safe: {100.0 * np.mean(safe):.1f}%"
            )

        all_u_old = np.concatenate(fast_u_loc_old)
        all_u_new = np.concatenate(fast_u_loc_new)
        all_du_t = np.concatenate(fast_delta_U_T)

        summary_data["fast"] = {
            "total_samples": len(all_u_old),
            "total_time": total_fast_time,
            "time_per_sample": total_fast_time / max(len(all_u_old), 1),
            "safe_fraction": float(np.mean(fast_safe_rates)),
            "u_loc_target": _calc_stats(all_u_old),
            "u_loc_model": _calc_stats(all_u_new),
            "delta_U_T": _calc_stats(all_du_t),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Density Mode (Augmented ODE + Acceptance Evaluation)
    # ─────────────────────────────────────────────────────────────────────────
    if density_trajectories > 0:
        print(f"\nStarting DENSITY generation ({density_trajectories} batches of {density_batch_size})...")
        dens_ds = grain.MapDataset.source(source).repeat().batch(density_batch_size, drop_remainder=True)
        dens_iter = iter(dens_ds.to_iter_dataset(read_options))

        # JIT Warmup
        warmup_batch = next(dens_iter)
        warmup_subkeys = jax.random.split(jax.random.key(777), density_batch_size)
        print("  Compiling augmented ODE & acceptance graph...", end="", flush=True)
        t_comp = time.time()
        _ = evaluate_patch_batch_with_density(
            patch_updater,
            warmup_batch,
            warmup_subkeys,
            masked_local_energy_fn,
            temperature,
            r_patch,
            sigma_r,
            max_rcut,
        )
        print(f" compiled in {time.time() - t_comp:.2f}s")

        all_accept_probs = []
        all_is_safe = []
        all_log_alpha = []
        all_log_pi_ratios = []
        all_log_q_ratios = []
        all_delta_U_T = []
        all_u_loc_old = []
        all_u_loc_new = []
        total_density_time = 0.0

        for i in range(1, density_trajectories + 1):
            batch = next(dens_iter)
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, density_batch_size)

            t_start = time.time()
            systems_new, sels, metrics = evaluate_patch_batch_with_density(
                patch_updater,
                batch,
                subkeys,
                masked_local_energy_fn,
                temperature,
                r_patch,
                sigma_r,
                max_rcut,
            )
            # Ensure completion across async device boundaries
            systems_new.positions.block_until_ready()
            t_batch = time.time() - t_start
            total_density_time += t_batch

            ap = np.asarray(metrics["accept_probs"])
            safe = np.asarray(metrics["is_safe"])
            la = np.asarray(metrics["log_alpha"])
            pi = np.asarray(metrics["log_pi_ratios"])
            q = np.asarray(metrics["log_q_ratios"])
            du = np.asarray(metrics["delta_U_T"])
            uo = np.asarray(metrics["u_loc_old"])
            un = np.asarray(metrics["u_loc_new"])
            q_fwd = np.asarray(metrics["log_q_forward"])
            q_bwd = np.asarray(metrics["log_q_backward"])

            all_accept_probs.append(ap)
            all_is_safe.append(safe)
            all_log_alpha.append(la)
            all_log_pi_ratios.append(pi)
            all_log_q_ratios.append(q)
            all_delta_U_T.append(du)
            all_u_loc_old.append(uo)
            all_u_loc_new.append(un)

            # Save batch files
            batch_dir = run_output_dir / "density" / str(i)
            batch_dir.mkdir(parents=True, exist_ok=True)

            _save_trajectory_xyz(
                batch_dir / "samples.xyz", systems_new, {"checkpoint": ckpt_path.name, "type": "proposal"}
            )
            _save_trajectory_xyz(
                batch_dir / "original_samples.xyz", batch, {"checkpoint": ckpt_path.name, "type": "original"}
            )

            # Text format for quick inspection
            np.savetxt(batch_dir / "acceptance.dat", ap, fmt="%.6e", header="accept_prob")

            # Full compressed numpy metrics
            np.savez_compressed(
                batch_dir / "metrics.npz",
                accept_probs=ap,
                is_safe=safe,
                log_alpha=la,
                log_pi_ratios=pi,
                log_q_ratios=q,
                delta_U_T=du,
                u_loc_old=uo,
                u_loc_new=un,
                log_q_forward=q_fwd,
                log_q_backward=q_bwd,
                anchor=np.asarray(sels.anchor),
                radius=np.asarray(sels.radius),
                patch_idx=np.asarray(sels.patch_idx),
                context_idx=np.asarray(sels.context_idx),
                n_patch=np.asarray(sels.n_patch),
                n_context=np.asarray(sels.n_context),
            )

            print(
                f"  [Density {i:03d}/{density_trajectories:03d}] {t_batch:.2f}s | "
                f"acc: {100.0 * np.mean(ap):.2f}% | "
                f"dU/T (med): {np.median(du):.2f} | "
                f"log_q: {np.mean(q):.2f} | "
                f"safe: {100.0 * np.mean(safe):.1f}%"
            )

        cat_ap = np.concatenate(all_accept_probs)
        cat_safe = np.concatenate(all_is_safe)
        cat_la = np.concatenate(all_log_alpha)
        cat_pi = np.concatenate(all_log_pi_ratios)
        cat_q = np.concatenate(all_log_q_ratios)
        cat_du = np.concatenate(all_delta_U_T)
        cat_uo = np.concatenate(all_u_loc_old)
        cat_un = np.concatenate(all_u_loc_new)

        summary_data["density"] = {
            "total_samples": len(cat_ap),
            "total_time": total_density_time,
            "time_per_sample": total_density_time / max(len(cat_ap), 1),
            "safe_fraction": float(np.mean(cat_safe)),
            "acceptance": _calc_stats(cat_ap),
            "log_alpha": _calc_stats(cat_la),
            "log_pi_ratio": _calc_stats(cat_pi),
            "log_q_ratio": _calc_stats(cat_q),
            "delta_U_T": _calc_stats(cat_du),
            "u_loc_target": _calc_stats(cat_uo),
            "u_loc_model": _calc_stats(cat_un),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Write summary.json
    # ─────────────────────────────────────────────────────────────────────────
    with open(run_output_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n{'=' * 65}")
    print("All runs completed successfully!")
    if "fast" in summary_data:
        print(
            f"  Fast mode    : {summary_data['fast']['total_samples']} samples in "
            f"{summary_data['fast']['total_time']:.1f}s ({summary_data['fast']['time_per_sample']:.4f}s/sample)"
        )
    if "density" in summary_data:
        print(
            f"  Density mode : {summary_data['density']['total_samples']} samples in "
            f"{summary_data['density']['total_time']:.1f}s ({summary_data['density']['time_per_sample']:.4f}s/sample)"
        )
        print(f"  Mean accept  : {100.0 * summary_data['density']['acceptance']['mean']:.2f}%")
        print(f"  Safe rate    : {100.0 * summary_data['density']['safe_fraction']:.1f}%")
    print(f"  Saved to     : {run_output_dir.resolve()}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    app()
