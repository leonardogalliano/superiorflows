"""Train a CNF on patches of particle systems.

Supports all four loss types (SI, MLE, EBL, hybrid) operating on subsystems
extracted from MD trajectory data.  Subsystem extraction uses the Grain pipeline
(ExtractAndShiftSubsystem → batch → CouplePatchBase → PatchEquivariantOT).

Layered configuration: defaults → JSON config → CLI overrides.
Run with ``--help`` for available CLI arguments.
"""

import copy
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
import typer
from typing_extensions import Annotated

from particle_systems.particle_system import (
    BoltzmannDistribution,
    ParticleSystem,
    TrajectoryDataSource,
    particle_geodesic_interpolant,
)
from particle_systems.patch_updater import (
    CouplePatchBase,
    ExtractAndShiftSubsystem,
    GaussianPatchBase,
    PatchEquivariantOptimalTransport,
)
from particle_systems.training_particles import (
    build_optimizer,
    build_schedule_fn,
    merge_config,
)
from particle_systems.velocities.conditional_egnn import PatchEGNNVelocity
from superiorflows import ODEBijector
from superiorflows.partial import fixed_selection
from superiorflows.train import (
    CheckpointCallback,
    EnergyBasedLoss,
    KullbackLeiblerLoss,
    LoggerCallback,
    LRSchedulerCallback,
    MaximumLikelihoodLoss,
    ProfilingCallback,
    ProgressBarCallback,
    StochasticInterpolantLoss,
    TensorBoardLogger,
    Trainer,
    ValidationCallback,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data": {
        "data_path": None,
        "model_file": None,
        "temperature": None,
        "num_workers": 1,
        "prefetch_buffer_size": 2,
        "max_dataset_length": None,
    },
    "patch": {
        "N_patch": None,
        "N_context": None,
        "r_patch": None,
        "r_context": None,
        "sigma_r": 0.0,
        "sigma_base": 1.0,
        "M_total": None,
    },
    "training": {
        "nsteps": None,
        "batch_size": 32,
        "seed": 0,
        "loss_type": "stochastic_interpolant",
        "log_freq": 100,
        "ckpt_path": "tmp/ckpt_patches",
        "overwrite": True,
        "num_checkpoints": 1,
        "load_from_checkpoint": None,
    },
    "optimizer": {
        "type": "adam",
        "lr_schedule": "1e-3",
        "clip": None,
    },
    "velocity": {
        "type": "egnn",
        "kwargs": {},
    },
    "solver": {
        "type": "tsit5",
        "atol": 1e-5,
        "rtol": 1e-5,
        "solver_steps": None,
    },
    "stochastic_interpolant": {
        "interpolant_scheduler": "t",
        "noise_scheduler": "None",
        "ot": {
            "enabled": True,
        },
    },
    "callbacks": {
        "acceptance": {
            "enabled": False,
            "freq": 250,
            "samples": 64,
            "batch_size": 32,
        },
        "patch_energy": {
            "enabled": False,
            "freq": 250,
            "samples": 256,
            "batch_size": 32,
        },
        "tensorboard": {
            "enabled": False,
            "freq": 10,
            "log_dir": "tmp/tb_logs_patches",
        },
        "profile": {
            "enabled": False,
            "log_dir": "tmp/profiles_patches",
            "warmup": 50,
            "steps": None,
        },
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

VELOCITY_REGISTRY = {
    "egnn": PatchEGNNVelocity,
}


def build_velocity_patch(config: dict, N_patch: int, N_context: int, d: int, n_species: int, *, key):
    """Instantiate a PatchEGNNVelocity for subsystems of N_patch + N_context particles."""
    vtype = config["velocity"]["type"]
    cls = VELOCITY_REGISTRY.get(vtype)
    if cls is None:
        raise ValueError(f"Unknown velocity type '{vtype}'. Available: {list(VELOCITY_REGISTRY)}")
    kwargs = config["velocity"].get("kwargs", {})
    return cls(N_patch=N_patch, N_context=N_context, d=d, n_species=n_species, **kwargs, key=key)


def build_solver_patch(config: dict) -> dict:
    """Build ``bijector_kwargs`` from the ``solver`` config block."""
    scfg = config["solver"]
    stype = scfg["type"].lower()
    solver_steps = scfg.get("solver_steps")

    solvers = {"euler": dfx.Euler, "tsit5": dfx.Tsit5, "dopri5": dfx.Dopri5}

    if stype not in solvers:
        raise ValueError(f"Unknown solver '{stype}'. Available: {list(solvers)}")

    if stype == "euler" and solver_steps is None:
        raise ValueError("solver.solver_steps is required when solver.type='euler'.")

    slv = solvers[stype]()

    extra_args = dict(throw=False)
    augmented_extra_args = dict(throw=False)
    if "extra_args" in scfg:
        extra_args.update(scfg["extra_args"])
    if "augmented_extra_args" in scfg:
        augmented_extra_args.update(scfg["augmented_extra_args"])

    bijector_kwargs: dict[str, Any] = dict(
        dynamic_mask=ParticleSystem(positions=jnp.arange(config["patch"]["N_patch"]), species=False, box=False),
        solver=slv,
        augmented_solver=slv,
        extra_args=extra_args,
        augmented_extra_args=augmented_extra_args,
    )

    if solver_steps is not None:
        bijector_kwargs.update(
            stepsize_controller=dfx.ConstantStepSize(),
            augmented_stepsize_controller=dfx.ConstantStepSize(),
            dt0=1.0 / solver_steps,
        )
    else:
        bijector_kwargs.update(
            stepsize_controller=dfx.PIDController(rtol=scfg["rtol"], atol=scfg["atol"]),
            augmented_stepsize_controller=dfx.PIDController(rtol=scfg["rtol"], atol=scfg["atol"]),
        )

    return bijector_kwargs


# ── Core training logic ──────────────────────────────────────────────────────


def train_single_model(config: dict):
    """Run a single patch-training job from a fully-merged configuration dict."""
    dcfg = config["data"]
    pcfg = config["patch"]
    tcfg = config["training"]

    data_path = Path(dcfg["data_path"])
    model_file = Path(dcfg["model_file"]) if dcfg["model_file"] else None
    temperature = dcfg["temperature"]
    num_workers = dcfg["num_workers"]
    prefetch_buffer_size = dcfg["prefetch_buffer_size"]

    nsteps = tcfg["nsteps"]
    batch_size = tcfg["batch_size"]
    seed = tcfg["seed"]
    loss_type = tcfg["loss_type"]
    log_freq = tcfg["log_freq"]
    ckpt_path = Path(tcfg["ckpt_path"])
    overwrite = tcfg["overwrite"]
    num_checkpoints = tcfg["num_checkpoints"]

    N_patch = pcfg["N_patch"]
    N_context = pcfg["N_context"]
    r_patch = pcfg["r_patch"]
    r_context = pcfg["r_context"]
    sigma_r = pcfg.get("sigma_r", 0.0)
    sigma_base = pcfg.get("sigma_base", 1.0)
    M_total = pcfg.get("M_total")

    s_expr = config["stochastic_interpolant"]["interpolant_scheduler"]
    gamma_expr = config["stochastic_interpolant"]["noise_scheduler"]
    use_ot = config["stochastic_interpolant"]["ot"]["enabled"]

    key = jax.random.key(seed)

    # Data
    source = TrajectoryDataSource(data_path)
    if (maxlen := dcfg.get("max_dataset_length")) is not None:
        source._positions = source._positions[:maxlen]
        source._species = source._species[:maxlen]
        source._box = source._box[:maxlen]
        source._n_frames = len(source._positions)

    N, d = source.N, source.d
    L = float(source.box_size[0])
    ref_species = source[0].species
    n_species = int(jnp.max(jnp.asarray(ref_species)) + 1)
    composition = np.bincount(ref_species, minlength=n_species) / len(ref_species)

    if M_total is None:
        M_total = N_patch + N_context

    # Selection function for subsystem extraction
    extract_transform = ExtractAndShiftSubsystem(
        N_patch=N_patch,
        N_context=N_context,
        r_patch=r_patch,
        r_context=r_context,
        sigma_r=sigma_r,
        seed=seed,
    )

    # Fixed selection protocol for Energy-Based Loss
    mask = ParticleSystem(positions=jnp.arange(N_patch), species=False, box=False)
    selection_protocol = fixed_selection(mask)

    # Solver configuration
    bijector_kwargs = build_solver_patch(config)

    # Target distribution (for EBL/hybrid)
    boltzmann_model = None
    target_dist = None
    if model_file is not None:
        if temperature is None:
            raise ValueError("data.temperature is required when data.model_file is set.")
        with open(model_file) as f:
            boltzmann_model = json.load(f)
        target_dist = BoltzmannDistribution(
            N=M_total,
            d=d,
            L=L,
            temperature=temperature,
            model=boltzmann_model,
            composition=composition,
            N_patch=N_patch,
        )

        max_rcut = max(
            float(np.max(c["parameters"]["rcut"]))
            if isinstance(c["parameters"]["rcut"], list)
            else float(c["parameters"]["rcut"])
            for c in boltzmann_model["cutoff"]
        )
    else:
        max_rcut = None

    def make_bijector(vf):
        return ODEBijector(vf, **bijector_kwargs)

    # Loss and dataset

    if loss_type == "stochastic_interpolant":
        s_fn = build_schedule_fn(s_expr)
        gamma_fn = build_schedule_fn(gamma_expr)

        def interpolant(t, x0, x1):
            return particle_geodesic_interpolant(t, x0, x1, L, s_fn=s_fn)

        si_kwargs = {k: v for k, v in bijector_kwargs.items() if k in ("dynamic_mask",)}
        loss_fn = StochasticInterpolantLoss(interpolant=interpolant, gamma=gamma_fn, **si_kwargs)
        dataset = (
            grain.MapDataset.source(source)
            .shuffle(seed=seed)
            .repeat()
            .map_with_index(extract_transform)
            .batch(batch_size, drop_remainder=True)
            .map_with_index(CouplePatchBase(N_patch=N_patch, d=d, L=L, sigma=sigma_base, seed=seed))
        )
        if use_ot:
            dataset = dataset.map(PatchEquivariantOptimalTransport(N_patch=N_patch))

    elif loss_type == "maximum_likelihood":
        loss_fn = MaximumLikelihoodLoss(
            base_distribution=GaussianPatchBase(
                center=jnp.full((d,), L / 2.0), sigma=sigma_base, N_patch=N_patch, d=d, L=L
            ),
            make_bijector=make_bijector,
            selection_protocol=selection_protocol,
        )
        dataset = (
            grain.MapDataset.source(source)
            .shuffle(seed=seed)
            .repeat()
            .map_with_index(extract_transform)
            .batch(batch_size, drop_remainder=True)
        )

    elif loss_type == "energy_based":
        if target_dist is None:
            raise ValueError("data.model_file is required for energy_based training.")
        base_dist = GaussianPatchBase(center=jnp.full((d,), L / 2.0), sigma=sigma_base, N_patch=N_patch, d=d, L=L)
        loss_fn = EnergyBasedLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            make_bijector=make_bijector,
            selection_protocol=selection_protocol,
        )
        dataset = (
            grain.MapDataset.source(source)
            .shuffle(seed=seed)
            .repeat()
            .map_with_index(extract_transform)
            .batch(batch_size, drop_remainder=True)
        )

    elif loss_type == "hybrid":
        if target_dist is None:
            raise ValueError("data.model_file is required for hybrid training.")
        base_dist = GaussianPatchBase(center=jnp.full((d,), L / 2.0), sigma=sigma_base, N_patch=N_patch, d=d, L=L)
        loss_fn = KullbackLeiblerLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            make_bijector=make_bijector,
            alpha=0.5,
            selection_protocol=selection_protocol,
        )
        dataset = (
            grain.MapDataset.source(source)
            .shuffle(seed=seed)
            .repeat()
            .map_with_index(extract_transform)
            .batch(batch_size, drop_remainder=True)
        )
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            "Choose from: stochastic_interpolant, maximum_likelihood, energy_based, hybrid."
        )

    # Model
    key, model_key = jax.random.split(key)
    model = build_velocity_patch(config, N_patch, N_context, d, n_species, key=model_key)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))

    # Optimizer
    optimizer, lr_schedule = build_optimizer(config)

    # Validation
    read_options_val = grain.ReadOptions(num_threads=1, prefetch_buffer_size=1)
    if loss_type == "stochastic_interpolant":
        val_ds = (
            grain.MapDataset.source(source)
            .map_with_index(ExtractAndShiftSubsystem(N_patch, N_context, r_patch, r_context, sigma_r, seed=seed + 1))
            .batch(batch_size, drop_remainder=True)
            .map_with_index(CouplePatchBase(N_patch=N_patch, d=d, L=L, sigma=sigma_base, seed=seed + 1))
        )
        if use_ot:
            val_ds = val_ds.map(PatchEquivariantOptimalTransport(N_patch=N_patch))
    else:
        val_ds = (
            grain.MapDataset.source(source)
            .map_with_index(ExtractAndShiftSubsystem(N_patch, N_context, r_patch, r_context, sigma_r, seed=seed + 1))
            .batch(batch_size, drop_remainder=True)
        )
    val_batch = next(iter(val_ds.to_iter_dataset(read_options_val)))
    val_data = [val_batch]

    # Run name and paths
    vtype = config["velocity"]["type"]
    lr_schedule_str = config["optimizer"]["lr_schedule"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"patch_{loss_type}_{vtype}_Np{N_patch}_Nc{N_context}_b{batch_size}_s{seed}_{timestamp}"
    chkpt_run_path = ckpt_path / run_name

    chkpt_run_path.mkdir(parents=True, exist_ok=True)
    with open(chkpt_run_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Callbacks
    ccfg = config["callbacks"]
    callbacks = []

    tb_cfg = ccfg["tensorboard"]
    tb_logger = None
    if tb_cfg["enabled"]:
        tb_run_dir = Path(tb_cfg["log_dir"]) / run_name
        v_kwargs = config["velocity"].get("kwargs", {})
        hparams = {
            "loss_type": loss_type,
            "velocity_type": vtype,
            "N_patch": N_patch,
            "N_context": N_context,
            "M_total": M_total,
            "d": d,
            "L": L,
            "r_patch": r_patch,
            "r_context": r_context,
            "sigma_r": sigma_r,
            "sigma_base": sigma_base,
            "optimizer_type": config["optimizer"].get("type", "adam"),
            "lr_schedule": lr_schedule_str,
            "clip": str(config["optimizer"].get("clip", "none")),
            "batch_size": batch_size,
            "seed": seed,
            "nsteps": nsteps,
            "ot": use_ot,
            "interpolant_scheduler": str(config.get("stochastic_interpolant", {}).get("interpolant_scheduler", "t")),
            "noise_scheduler": str(config.get("stochastic_interpolant", {}).get("noise_scheduler", "None")),
            "time_embedding_dim": int(v_kwargs.get("time_embedding_dim", 0)),
            "coords_range": float(v_kwargs.get("coords_range", 15.0)),
            "temp": temperature if temperature is not None else -1.0,
            "v_kwargs": str(v_kwargs),
            "solver": config["solver"]["type"],
        }
        tb_logger = TensorBoardLogger(log_dir=tb_run_dir, log_freq=tb_cfg.get("freq", log_freq), hparams=hparams)
        tb_writer = tb_logger._writer
    else:
        tb_writer = None

    acc_cfg = ccfg.get("acceptance", {})
    pe_cfg = ccfg.get("patch_energy", {})

    if target_dist is not None and (acc_cfg.get("enabled", False) or pe_cfg.get("enabled", False)):
        from particle_systems.patch_updater import PatchUpdater
        from particle_systems.selection import random_point_select

        sel_fn = random_point_select(
            N_patch=N_patch,
            N_context=N_context,
            r_patch=r_patch,
            r_context=r_context,
            sigma_r=sigma_r,
        )

        def make_patch_updater(vf):
            bij = make_bijector(vf)
            return PatchUpdater(
                bijector=bij,
                selection_fn=sel_fn,
                N_patch=N_patch,
                N_context=N_context,
                sigma=sigma_base,
                L=L,
                d=d,
            )

    if target_dist is not None and acc_cfg.get("enabled", False):
        from particle_systems.callbacks_particles import PatchAcceptanceCallback
        from particle_systems.potentials import build_masked_energy_fn

        acc_samples = acc_cfg.get("samples", 64)
        acc_batch_size = acc_cfg.get("batch_size", batch_size)

        val_full_ds = grain.MapDataset.source(source).repeat().batch(acc_batch_size, drop_remainder=True)
        val_full_iter = iter(val_full_ds.to_iter_dataset(read_options_val))
        num_batches = max(1, acc_samples // acc_batch_size)
        val_full_data = [next(val_full_iter) for _ in range(num_batches)]

        masked_energy_fn = build_masked_energy_fn(boltzmann_model, source.box_size, n_species, N_patch=N_patch)

        callbacks.append(
            PatchAcceptanceCallback(
                val_data=val_full_data,
                make_patch_updater=make_patch_updater,
                masked_energy_fn=masked_energy_fn,
                temperature=temperature,
                r_mean=r_patch,
                sigma_r=sigma_r,
                r_cut=max_rcut,
                r_context=r_context,
                eval_freq=acc_cfg.get("freq", log_freq),
                tb_writer=tb_writer,
                species_radii=np.array(acc_cfg.get("sigmas", [1.0, 0.8, 0.9])),
            )
        )

    if target_dist is not None and pe_cfg.get("enabled", False):
        from particle_systems.callbacks_particles import PatchLocalEnergyCallback
        from particle_systems.potentials import build_masked_local_energy_fn

        pe_samples = pe_cfg.get("samples", 256)
        pe_batch_size = pe_cfg.get("batch_size", batch_size)

        val_full_ds_pe = grain.MapDataset.source(source).repeat().batch(pe_batch_size, drop_remainder=True)
        val_full_iter_pe = iter(val_full_ds_pe.to_iter_dataset(read_options_val))
        num_batches_pe = max(1, pe_samples // pe_batch_size)
        val_full_data_pe = [next(val_full_iter_pe) for _ in range(num_batches_pe)]

        masked_local_energy_fn = build_masked_local_energy_fn(
            boltzmann_model, source.box_size, n_species, N_patch=N_patch
        )

        callbacks.append(
            PatchLocalEnergyCallback(
                val_data=val_full_data_pe,
                make_patch_updater=make_patch_updater,
                masked_local_energy_fn=masked_local_energy_fn,
                eval_freq=pe_cfg.get("freq", log_freq),
                tb_writer=tb_writer,
            )
        )

    save_freq = max(1, nsteps // num_checkpoints) if num_checkpoints > 0 else nsteps + 1
    callbacks += [
        LRSchedulerCallback(lr_schedule),
        ValidationCallback(val_data=val_data, loss_module=loss_fn, val_freq=log_freq),
        LoggerCallback(log_freq=log_freq),
        ProgressBarCallback(refresh_rate=max(1, nsteps // 100)),
        CheckpointCallback(ckpt_path=chkpt_run_path, save_freq=save_freq, overwrite=overwrite),
    ]

    if ccfg["profile"]["enabled"]:
        callbacks.append(
            ProfilingCallback(
                log_dir=Path(ccfg["profile"]["log_dir"]),
                warmup_steps=ccfg["profile"]["warmup"],
                profile_steps=ccfg["profile"]["steps"],
            )
        )

    if tb_logger is not None:
        callbacks.append(tb_logger)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_module=loss_fn,
        seed=seed,
        callbacks=callbacks,
    )

    # Load from checkpoint if requested
    load_from_ckpt = tcfg.get("load_from_checkpoint")
    if load_from_ckpt is not None:
        ckpt_load_path = Path(load_from_ckpt)
        if not ckpt_load_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_load_path}")
        success = trainer.load_checkpoint(str(ckpt_load_path))
        if not success:
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_load_path}")
        trainer._restored_data_state = None

    # Banner
    model_label = model_file.name if model_file else "none"
    vkwargs = config["velocity"].get("kwargs", {})
    vkwargs_str = ", ".join(f"{k}={v}" for k, v in vkwargs.items())
    print(f"\n{'=' * 60}")
    print("Training CNF on particle PATCHES")
    print(f"  Data            : {data_path}")
    print(f"  Potential       : {model_label}")
    print(f"  Full system     : N={N}, d={d}, L={L:.4f}")
    print(f"  Patch           : N_patch={N_patch}, N_context={N_context}, M_total={M_total}")
    print(f"  Selection       : r_patch={r_patch}, r_context={r_context}, sigma_r={sigma_r}")
    print(f"  Dataset size    : {len(source)}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Loss            : {loss_type}")
    print(f"  Velocity        : {vtype} ({vkwargs_str})")
    print(f"  Parameters      : {num_params:,}")
    print(f"  Optimizer       : {config['optimizer']['type']} | lr_schedule = {lr_schedule_str}")
    print(f"  Run             : {run_name}")
    print(f"  Ckpts           : {chkpt_run_path}")
    if load_from_ckpt is not None:
        print(f"  Resumed from    : {load_from_ckpt} (step {trainer.step})")
        print(f"  Training        : step {trainer.step} \u2192 {nsteps}")
    print(f"  JAX process     : {jax.process_index()}/{jax.process_count()}")
    print(f"  JAX devices     : {jax.devices()}")
    print(f"{'=' * 60}\n")

    t_start = time.time()
    read_options = grain.ReadOptions(num_threads=num_workers, prefetch_buffer_size=prefetch_buffer_size)
    trainer.train(dataset=dataset, max_steps=nsteps, read_options=read_options)
    t_elapsed = time.time() - t_start

    print(f"\nDone in {t_elapsed:.1f}s ({1000 * t_elapsed / nsteps:.0f}ms/step)\n")
    return trainer


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    config: Annotated[Path | None, typer.Option("--config", help="JSON config file")] = None,
    data_path: Annotated[str | None, typer.Option("--data-path", help="Path to trajectory directory")] = None,
    nsteps: Annotated[int | None, typer.Option("--nsteps", help="Number of training steps")] = None,
    model_file: Annotated[str | None, typer.Option("--model-file", help="JSON potential model file")] = None,
    loss_type: Annotated[str | None, typer.Option("--loss-type", help="Loss function type")] = None,
    lr: Annotated[
        float | None, typer.Option("--lr", help="Constant learning rate (sets optimizer.lr_schedule)")
    ] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", help="Batch size")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    temperature: Annotated[float | None, typer.Option("--temperature", help="Boltzmann temperature")] = None,
    n_patch: Annotated[int | None, typer.Option("--n-patch", help="Number of patch particles")] = None,
    n_context: Annotated[int | None, typer.Option("--n-context", help="Number of context particles")] = None,
    r_patch: Annotated[float | None, typer.Option("--r-patch", help="Patch selection radius")] = None,
    r_context: Annotated[float | None, typer.Option("--r-context", help="Context selection radius")] = None,
    acceptance: Annotated[
        bool | None, typer.Option("--acceptance/--no-acceptance", help="Enable acceptance monitoring")
    ] = None,
    tensorboard: Annotated[
        bool | None, typer.Option("--tensorboard/--no-tensorboard", help="Enable TensorBoard")
    ] = None,
    profile: Annotated[bool | None, typer.Option("--profile/--no-profile", help="Enable JAX profiling")] = None,
    device: Annotated[str | None, typer.Option("--device", help="JAX device: cpu | gpu")] = None,
    load_from_checkpoint: Annotated[
        str | None, typer.Option("--load-from-checkpoint", help="Checkpoint directory to resume from")
    ] = None,
    time_embedding_dim: Annotated[
        int | None, typer.Option("--time-embedding-dim", help="Dimension of sinusoidal time embedding")
    ] = None,
    coords_range: Annotated[float | None, typer.Option("--coords-range", help="EGNN coordinate jump range")] = None,
):
    """Train a CNF on patches of particle systems."""
    if device is not None:
        jax.config.update("jax_platform_name", device)

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if config is not None:
        with open(config) as f:
            cfg = merge_config(cfg, json.load(f))

    if data_path is not None:
        cfg["data"]["data_path"] = data_path
    if model_file is not None:
        cfg["data"]["model_file"] = model_file
    if temperature is not None:
        cfg["data"]["temperature"] = temperature
    if nsteps is not None:
        cfg["training"]["nsteps"] = nsteps
    if lr is not None:
        cfg["optimizer"]["lr_schedule"] = str(lr)
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
    if seed is not None:
        cfg["training"]["seed"] = seed
    if loss_type is not None:
        cfg["training"]["loss_type"] = loss_type
    if n_patch is not None:
        cfg["patch"]["N_patch"] = n_patch
    if n_context is not None:
        cfg["patch"]["N_context"] = n_context
    if r_patch is not None:
        cfg["patch"]["r_patch"] = r_patch
    if r_context is not None:
        cfg["patch"]["r_context"] = r_context
    if acceptance is not None:
        cfg["callbacks"]["acceptance"]["enabled"] = acceptance
    if tensorboard is not None:
        cfg["callbacks"]["tensorboard"]["enabled"] = tensorboard
    if profile is not None:
        cfg["callbacks"]["profile"]["enabled"] = profile
    if load_from_checkpoint is not None:
        cfg["training"]["load_from_checkpoint"] = load_from_checkpoint
    if time_embedding_dim is not None:
        cfg["velocity"]["kwargs"]["time_embedding_dim"] = time_embedding_dim
    if coords_range is not None:
        cfg["velocity"]["kwargs"]["coords_range"] = coords_range

    missing = []
    if cfg["data"]["data_path"] is None:
        missing.append("data.data_path (--data-path)")
    if cfg["training"]["nsteps"] is None:
        missing.append("training.nsteps (--nsteps)")
    if cfg["patch"]["N_patch"] is None:
        missing.append("patch.N_patch (--n-patch)")
    if cfg["patch"]["N_context"] is None:
        missing.append("patch.N_context (--n-context)")
    if cfg["patch"]["r_patch"] is None:
        missing.append("patch.r_patch (--r-patch)")
    if cfg["patch"]["r_context"] is None:
        missing.append("patch.r_context (--r-context)")
    if missing:
        raise typer.BadParameter(
            f"Missing required config: {', '.join(missing)}. Provide via --config JSON or CLI arguments."
        )

    train_single_model(cfg)


if __name__ == "__main__":
    app()
