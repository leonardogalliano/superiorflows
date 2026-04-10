"""Train a CNF on MD particle trajectory data.

Supports layered configuration: defaults → JSON config → CLI overrides.
Run with ``--help`` for available CLI arguments, or pass ``--config config.json``
to specify everything in a single file.
"""

import copy
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict

import diffrax as dfx
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from superiorflows import DistributionDataSource, Flow
from superiorflows.train import (
    Callback,
    CheckpointCallback,
    EnergyBasedLoss,
    ESSCallback,
    KullbackLeiblerLoss,
    LoggerCallback,
    MaximumLikelihoodLoss,
    ProfilingCallback,
    ProgressBarCallback,
    StochasticInterpolantLoss,
    TensorBoardLogger,
    Trainer,
    ValidationCallback,
)
from typing_extensions import Annotated

from particle_systems.particle_system import (
    BoltzmannDistribution,
    CoupleBaseSamples,
    EquivariantOptimalTransport,
    ParticleSystem,
    TrajectoryDataSource,
    UniformParticles,
    particle_geodesic_interpolant,
)
from particle_systems.velocities import ParticlesEGNNVelocity, ParticlesMLPVelocity

app = typer.Typer(pretty_exceptions_show_locals=False)


# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data": {
        "data_path": None,
        "model_file": None,
        "temperature": None,
        "num_workers": 1,
        "prefetch_buffer_size": 2,
    },
    "training": {
        "nsteps": None,
        "lr": 1e-3,
        "batch_size": 32,
        "seed": 0,
        "loss_type": "maximum_likelihood",
        "log_freq": 100,
        "ckpt_path": "tmp/ckpt_particles",
        "overwrite": True,
        "num_checkpoints": 1,
    },
    "velocity": {
        "type": "mlp",
        "kwargs": {},
    },
    "solver": {
        "type": "tsit5",
        "atol": 1e-5,
        "rtol": 1e-5,
        "euler_steps": None,
    },
    "stochastic_interpolant": {
        "use_gamma": False,
    },
    "ot": {
        "enabled": True,
        "box_symmetry": False,
    },
    "callbacks": {
        "ess": {
            "enabled": False,
            "freq": 250,
            "samples": 512,
        },
        "potential_energy": {
            "enabled": False,
            "freq": 100,
        },
        "tensorboard": {
            "enabled": False,
            "freq": 100,
            "log_dir": "tmp/tb_logs",
        },
        "profile": {
            "enabled": False,
            "log_dir": "tmp/profiles",
            "warmup": 50,
            "steps": None,
        },
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def merge_config(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, returning a new dict."""
    result = base.copy()
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


VELOCITY_REGISTRY = {
    "mlp": ParticlesMLPVelocity,
    "egnn": ParticlesEGNNVelocity,
}


def build_velocity(config: dict, N: int, d: int, n_species: int, *, key):
    """Instantiate a velocity field from the ``velocity`` config block."""
    vtype = config["velocity"]["type"]
    cls = VELOCITY_REGISTRY.get(vtype)
    if cls is None:
        raise ValueError(f"Unknown velocity type '{vtype}'. " f"Available: {list(VELOCITY_REGISTRY)}")
    kwargs = config["velocity"].get("kwargs", {})
    return cls(N=N, d=d, n_species=n_species, **kwargs, key=key)


def build_solver(config: dict) -> dict:
    """Build ``flow_kwargs`` from the ``solver`` config block."""
    scfg = config["solver"]
    stype = scfg["type"].lower()

    if stype == "euler":
        euler_steps = scfg.get("euler_steps")
        if euler_steps is None:
            raise ValueError("solver.euler_steps is required when solver.type='euler'.")
        return dict(
            dynamic_mask=ParticleSystem.get_dynamic_mask(),
            solver=dfx.Euler(),
            augmented_solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            augmented_stepsize_controller=dfx.ConstantStepSize(),
            dt0=1.0 / euler_steps,
        )

    solvers = {"tsit5": dfx.Tsit5, "dopri5": dfx.Dopri5}
    if stype not in solvers:
        raise ValueError(f"Unknown solver '{stype}'. Available: {list(solvers)}")
    slv = solvers[stype]()
    return dict(
        dynamic_mask=ParticleSystem.get_dynamic_mask(),
        solver=slv,
        augmented_solver=slv,
        stepsize_controller=dfx.PIDController(rtol=scfg["rtol"], atol=scfg["atol"]),
        augmented_stepsize_controller=dfx.PIDController(rtol=scfg["rtol"], atol=scfg["atol"]),
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────


class PotentialEnergyCallback(Callback):
    """Periodically samples from the current flow and logs mean potential energy.

    Draws ``n_samples`` configurations from the flow's pushforward distribution,
    evaluates the interatomic potential on each, and injects ``energy``
    into the logs dict so that ``LoggerCallback`` and ``TensorBoardLogger`` pick
    it up automatically.
    """

    def __init__(
        self,
        energy_fn,
        base_distribution,
        ref_species,
        flow_kwargs: dict | None = None,
        n_samples: int = 256,
        eval_freq: int = 250,
    ):
        self.energy_fn = energy_fn
        self.base_distribution = base_distribution
        self.ref_species = ref_species
        self.flow_kwargs = flow_kwargs or {}
        self.n_samples = n_samples
        self.eval_freq = eval_freq

    def on_train_start(self, trainer, **kwargs):
        self.total_steps = kwargs.get("total_steps", -1)

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        is_last = hasattr(self, "total_steps") and step == self.total_steps
        if step % self.eval_freq != 0 and step != 1 and not is_last:
            return

        flow = Flow(
            velocity_field=trainer.model,
            base_distribution=self.base_distribution,
            **self.flow_kwargs,
        )

        try:
            key, subkey = jax.random.split(trainer.key)
            samples = flow.sample(seed=subkey, sample_shape=(self.n_samples,))

            energy_fn = self.energy_fn
            ref_species = self.ref_species

            def single_energy(sample: ParticleSystem):
                return energy_fn(sample.positions, ref_species)

            mean_energy = jnp.mean(jax.vmap(single_energy)(samples))
        except Exception:
            mean_energy = float("nan")
        logs["energy"] = mean_energy


# ── Core training logic ──────────────────────────────────────────────────────


def train_single_model(config: dict):
    """Run a single training job from a fully-merged configuration dict."""
    dcfg = config["data"]
    tcfg = config["training"]

    data_path = Path(dcfg["data_path"])
    model_file = Path(dcfg["model_file"]) if dcfg["model_file"] else None
    temperature = dcfg["temperature"]
    num_workers = dcfg["num_workers"]
    prefetch_buffer_size = dcfg["prefetch_buffer_size"]

    nsteps = tcfg["nsteps"]
    lr = tcfg["lr"]
    batch_size = tcfg["batch_size"]
    seed = tcfg["seed"]
    loss_type = tcfg["loss_type"]
    log_freq = tcfg["log_freq"]
    ckpt_path = Path(tcfg["ckpt_path"])
    overwrite = tcfg["overwrite"]
    num_checkpoints = tcfg["num_checkpoints"]

    use_gamma = config["stochastic_interpolant"]["use_gamma"]
    use_ot = config["ot"]["enabled"]
    use_ot_box_symmetry = config["ot"]["box_symmetry"]

    key = jax.random.key(seed)

    # Data
    source = TrajectoryDataSource(data_path)
    N, d = source.N, source.d
    L = float(source.box_size[0])
    ref_species = source[0].species
    composition = np.bincount(ref_species) / len(ref_species)

    # Distributions
    base_dist = UniformParticles(N=N, d=d, L=L, composition=composition)
    flow_kwargs = build_solver(config)

    # Target distribution
    boltzmann_model = None
    target_dist = None
    if model_file is not None:
        if temperature is None:
            raise ValueError("data.temperature is required when data.model_file is set.")
        with open(model_file) as f:
            boltzmann_model = json.load(f)
        target_dist = BoltzmannDistribution(
            N=N,
            d=d,
            L=L,
            temperature=temperature,
            model=boltzmann_model,
            composition=composition,
        )

    # Loss and dataset
    if loss_type == "maximum_likelihood":
        loss_fn = MaximumLikelihoodLoss(base_distribution=base_dist, **flow_kwargs)
        dataset = source.to_dataset(batch_size=batch_size, shuffle=True, seed=seed).repeat()

    elif loss_type == "energy_based":
        if target_dist is None:
            raise ValueError("data.model_file is required for energy_based training.")
        loss_fn = EnergyBasedLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            **flow_kwargs,
        )
        dataset = grain.MapDataset.source(DistributionDataSource(base_dist, batch_size, seed=seed)).repeat()

    elif loss_type == "hybrid":
        if target_dist is None:
            raise ValueError("data.model_file is required for hybrid training.")
        loss_fn = KullbackLeiblerLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            alpha=0.5,
            **flow_kwargs,
        )
        dataset = source.to_dataset(batch_size=batch_size, shuffle=True, seed=seed).repeat()

    elif loss_type == "stochastic_interpolant":

        def interpolant(t, x0, x1):
            return particle_geodesic_interpolant(t, x0, x1, L)

        def gamma_fn(t):
            return jnp.sqrt(2 * t * (1 - t))

        gamma_fn_arg = gamma_fn if use_gamma else None
        loss_fn = StochasticInterpolantLoss(interpolant=interpolant, gamma=gamma_fn_arg, **flow_kwargs)
        dataset = (
            source.to_dataset(batch_size=batch_size, shuffle=True, seed=seed)
            .repeat()
            .map_with_index(CoupleBaseSamples(base_dist, seed=seed))
        )
        if use_ot:
            dataset = dataset.map(EquivariantOptimalTransport(use_box_symmetry=use_ot_box_symmetry))
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            "Choose from: maximum_likelihood, energy_based, hybrid, stochastic_interpolant."
        )

    # Model
    key, model_key = jax.random.split(key)
    n_species = len(jnp.unique(jnp.asarray(source[0].species)))
    model = build_velocity(config, N, d, n_species, key=model_key)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))

    # Optimizer
    optimizer = optax.adam(lr)

    # Validation
    read_options_val = grain.ReadOptions(num_threads=1, prefetch_buffer_size=1)
    if loss_type == "energy_based":
        val_ds = grain.MapDataset.source(DistributionDataSource(base_dist, batch_size, seed=seed + 1))
    elif loss_type == "stochastic_interpolant":
        val_ds = source.to_dataset(batch_size=batch_size, shuffle=False, seed=seed + 1).map_with_index(
            CoupleBaseSamples(base_dist, seed=seed + 1)
        )
        if use_ot:
            val_ds = val_ds.map(EquivariantOptimalTransport(use_box_symmetry=use_ot_box_symmetry))
    else:
        val_ds = source.to_dataset(batch_size=batch_size, shuffle=False, seed=seed + 1)
    val_batch = next(iter(val_ds.to_iter_dataset(read_options_val)))
    val_data = [val_batch]

    # Run name and paths
    vtype = config["velocity"]["type"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{loss_type}_{vtype}_lr{lr}_b{batch_size}_s{seed}_{timestamp}"
    chkpt_run_path = ckpt_path / run_name

    # Archive the resolved config for reproducibility
    chkpt_run_path.mkdir(parents=True, exist_ok=True)
    with open(chkpt_run_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Callbacks
    ccfg = config["callbacks"]
    callbacks = []

    if ccfg["ess"]["enabled"] and target_dist is not None:
        callbacks.append(
            ESSCallback(
                target_log_prob=target_dist.log_prob,
                base_distribution=base_dist,
                flow_kwargs=flow_kwargs,
                n_samples=ccfg["ess"]["samples"],
                eval_freq=ccfg["ess"]["freq"],
            )
        )

    pe_cfg = ccfg.get("potential_energy", {})
    if target_dist is not None and pe_cfg.get("enabled", True):
        callbacks.append(
            PotentialEnergyCallback(
                energy_fn=target_dist._energy_fn,
                base_distribution=base_dist,
                ref_species=jnp.array(source[0].species),
                flow_kwargs=flow_kwargs,
                n_samples=batch_size,
                eval_freq=pe_cfg.get("freq", log_freq),
            )
        )

    save_freq = max(1, nsteps // num_checkpoints) if num_checkpoints > 0 else nsteps + 1
    callbacks += [
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

    tb_cfg = ccfg["tensorboard"]
    if tb_cfg["enabled"]:
        tb_run_dir = Path(tb_cfg["log_dir"]) / run_name
        hparams = {
            "loss_type": loss_type,
            "velocity_type": vtype,
            "N": N,
            "d": d,
            "L": L,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "nsteps": nsteps,
            "num_workers": num_workers,
            "prefetch": prefetch_buffer_size,
            "ot": use_ot,
            "box_sym": use_ot_box_symmetry,
            "temp": temperature if temperature is not None else -1.0,
            "v_kwargs": str(config["velocity"].get("kwargs", {})),
            "solver": config["solver"]["type"],
            "ess_enabled": ccfg["ess"]["enabled"],
        }
        callbacks.append(TensorBoardLogger(log_dir=tb_run_dir, log_freq=tb_cfg.get("freq", log_freq), hparams=hparams))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_module=loss_fn,
        seed=seed,
        callbacks=callbacks,
    )

    # Banner
    model_label = model_file.name if model_file else "none"
    vkwargs = config["velocity"].get("kwargs", {})
    vkwargs_str = ", ".join(f"{k}={v}" for k, v in vkwargs.items())
    print(f"\n{'='*60}")
    print("Training CNF on particle trajectories")
    print(f"  Data       : {data_path}")
    print(f"  Potential  : {model_label}")
    print(f"  N={N}, d={d}, L={L:.4f}")
    print(f"  Dataset    : {len(source)}")
    print(f"  Loss       : {loss_type}")
    print(f"  Velocity   : {vtype} ({vkwargs_str})")
    print(f"  Parameters : {num_params:,}")
    print(f"  Run        : {run_name}")
    print(f"  Ckpts      : {chkpt_run_path}")
    print(f"{'='*60}\n")

    t_start = time.time()
    read_options = grain.ReadOptions(num_threads=num_workers, prefetch_buffer_size=prefetch_buffer_size)
    trainer.train(dataset=dataset, max_steps=nsteps, read_options=read_options)
    t_elapsed = time.time() - t_start

    print(f"\nDone in {t_elapsed:.1f}s ({1000*t_elapsed/nsteps:.0f}ms/step)\n")
    return trainer


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    config: Annotated[Path | None, typer.Option("--config", help="JSON config file")] = None,
    data_path: Annotated[str | None, typer.Option("--data-path", help="Path to trajectory directory")] = None,
    nsteps: Annotated[int | None, typer.Option("--nsteps", help="Number of training steps")] = None,
    model_file: Annotated[str | None, typer.Option("--model-file", help="JSON potential model file")] = None,
    loss_type: Annotated[str | None, typer.Option("--loss-type", help="Loss function type")] = None,
    lr: Annotated[float | None, typer.Option("--lr", help="Learning rate")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", help="Batch size")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    temperature: Annotated[float | None, typer.Option("--temperature", help="Boltzmann temperature")] = None,
    ess: Annotated[bool | None, typer.Option("--ess/--no-ess", help="Enable ESS monitoring")] = None,
    tensorboard: Annotated[
        bool | None, typer.Option("--tensorboard/--no-tensorboard", help="Enable TensorBoard")
    ] = None,
    profile: Annotated[bool | None, typer.Option("--profile/--no-profile", help="Enable JAX profiling")] = None,
    ot: Annotated[bool | None, typer.Option("--ot/--no-ot", help="Enable optimal transport")] = None,
    device: Annotated[str | None, typer.Option("--device", help="JAX device: cpu | gpu")] = None,
):
    """Train a CNF on MD particle trajectory data."""
    if device is not None:
        jax.config.update("jax_platform_name", device)
        print(f"JAX process: {jax.process_index()}/{jax.process_count()}")
        print(f"JAX devices: {jax.devices()}")

    # 1. Start from defaults
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # 2. Merge JSON config if provided
    if config is not None:
        with open(config) as f:
            cfg = merge_config(cfg, json.load(f))

    # 3. Apply CLI overrides (only when explicitly provided)
    if data_path is not None:
        cfg["data"]["data_path"] = data_path
    if model_file is not None:
        cfg["data"]["model_file"] = model_file
    if temperature is not None:
        cfg["data"]["temperature"] = temperature
    if nsteps is not None:
        cfg["training"]["nsteps"] = nsteps
    if lr is not None:
        cfg["training"]["lr"] = lr
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
    if seed is not None:
        cfg["training"]["seed"] = seed
    if loss_type is not None:
        cfg["training"]["loss_type"] = loss_type
    if ess is not None:
        cfg["callbacks"]["ess"]["enabled"] = ess
    if tensorboard is not None:
        cfg["callbacks"]["tensorboard"]["enabled"] = tensorboard
    if profile is not None:
        cfg["callbacks"]["profile"]["enabled"] = profile
    if ot is not None:
        cfg["ot"]["enabled"] = ot

    # 4. Validate mandatory fields
    missing = []
    if cfg["data"]["data_path"] is None:
        missing.append("data.data_path (--data-path)")
    if cfg["training"]["nsteps"] is None:
        missing.append("training.nsteps (--nsteps)")
    if missing:
        raise typer.BadParameter(
            f"Missing required config: {', '.join(missing)}. " "Provide via --config JSON or CLI arguments."
        )

    train_single_model(cfg)


if __name__ == "__main__":
    app()
