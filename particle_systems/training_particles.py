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
    TensorBoardLogger,
    Trainer,
    ValidationCallback,
)
from typing_extensions import Annotated

from particle_systems.particle_system import (
    BoltzmannDistribution,
    ParticleSystem,
    TrajectoryDataSource,
    UniformParticles,
)
from particle_systems.velocities import ParticlesMLPVelocity

app = typer.Typer(pretty_exceptions_show_locals=False)


class PotentialEnergyCallback(Callback):
    """Periodically samples from the current flow and logs mean potential energy.

    Draws `n_samples` configurations from the flow's pushforward distribution,
    evaluates the interatomic potential on each, and injects `energy`
    into the logs dict so that `LoggerCallback` and `TensorBoardLogger` pick
    it up automatically.

    Args:
        energy_fn: Callable ``(positions, species) -> scalar`` for a single frame.
        base_distribution: The base distribution used by the flow.
        ref_species: Integer species array for a single frame, shape ``(N,)``.
        flow_kwargs: Extra kwargs forwarded to ``Flow(...)`` at eval time.
        n_samples: Number of flow samples to average over.
        eval_freq: Compute every N steps.
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

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        if step % self.eval_freq != 0:
            return

        flow = Flow(
            velocity_field=trainer.model,
            base_distribution=self.base_distribution,
            **self.flow_kwargs,
        )

        key, subkey = jax.random.split(trainer.key)
        samples = flow.sample(seed=subkey, sample_shape=(self.n_samples,))

        energy_fn = self.energy_fn
        ref_species = self.ref_species

        def single_energy(sample: ParticleSystem):
            return energy_fn(sample.positions, ref_species)

        mean_energy = jnp.mean(jax.vmap(single_energy)(samples))
        logs["energy"] = mean_energy


def train_single_model(
    data_path: Path,
    model_file: Path | None,
    loss_type: str,
    width: int,
    depth: int,
    lr: float,
    nsteps: int,
    batch_size: int,
    seed: int,
    log_freq: int,
    ckpt_path: Path,
    overwrite: bool,
    num_checkpoints: int = 1,
    temperature: float | None = None,
    ess: bool = False,
    ess_freq: int = 250,
    ess_samples: int = 512,
    profile: bool = False,
    profile_log_dir: Path = Path("tmp/profiles"),
    profile_warmup: int = 50,
    profile_steps: int | None = None,
    tensorboard: bool = False,
    tensorboard_log_dir: Path = Path("tmp/tb_logs"),
    num_workers: int = 1,
    prefetch_buffer_size: int = 2,
    solver: str = "tsit5",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    euler_steps: int | None = None,
):
    key = jax.random.key(seed)

    # Data
    source = TrajectoryDataSource(data_path)
    N, d = source.N, source.d
    L = float(source.box_size[0])

    # Distributions
    base_dist = UniformParticles(N=N, d=d, L=L, composition=(0.5, 0.5))

    if solver.lower() == "euler":
        if euler_steps is None:
            raise ValueError("--euler-steps must be specified when using the 'euler' solver.")
        flow_kwargs = dict(
            dynamic_mask=ParticleSystem.get_dynamic_mask(),
            solver=dfx.Euler(),
            augmented_solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            augmented_stepsize_controller=dfx.ConstantStepSize(),
            dt0=1.0 / euler_steps,
        )
    else:
        if solver.lower() == "tsit5":
            slv = dfx.Tsit5()
        elif solver.lower() == "dopri5":
            slv = dfx.Dopri5()
        else:
            raise ValueError(f"Unknown solver '{solver}'.")

        flow_kwargs = dict(
            dynamic_mask=ParticleSystem.get_dynamic_mask(),
            solver=slv,
            augmented_solver=slv,
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            augmented_stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        )

    # Load Boltzmann model from JSON if provided
    boltzmann_model = None
    target_dist = None
    if model_file is not None:
        if temperature is None:
            raise ValueError("--temperature is required when --model-file is provided.")
        with open(model_file) as f:
            boltzmann_model = json.load(f)
        target_dist = BoltzmannDistribution(
            N=N,
            d=d,
            L=L,
            temperature=temperature,
            model=boltzmann_model,
            composition=(0.5, 0.5),
        )

    # Loss and data source
    if loss_type == "maximum_likelihood":
        loss_fn = MaximumLikelihoodLoss(base_distribution=base_dist, **flow_kwargs)
        dataset = source.to_dataset(batch_size=batch_size, shuffle=True, seed=seed).repeat()

    elif loss_type == "energy_based":
        if target_dist is None:
            raise ValueError("--model-file is required for energy_based training.")
        loss_fn = EnergyBasedLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            **flow_kwargs,
        )
        dataset = grain.MapDataset.source(DistributionDataSource(base_dist, batch_size, seed=seed)).repeat()

    elif loss_type == "hybrid":
        if target_dist is None:
            raise ValueError("--model-file is required for hybrid training.")
        loss_fn = KullbackLeiblerLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            alpha=0.5,
            **flow_kwargs,
        )
        dataset = source.to_dataset(batch_size=batch_size, shuffle=True, seed=seed).repeat()

    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. " "Choose from: maximum_likelihood, energy_based, hybrid.")

    # Model
    key, model_key = jax.random.split(key)
    n_species = len(jnp.unique(jnp.asarray(source[0].species)))
    model = ParticlesMLPVelocity(N=N, d=d, n_species=n_species, width=width, depth=depth, key=model_key)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))

    # Optimizer
    optimizer = optax.adam(lr)

    # Validation: one batch from the same distribution as training
    read_options_val = grain.ReadOptions(num_threads=1, prefetch_buffer_size=1)
    if loss_type == "energy_based":
        val_ds = grain.MapDataset.source(DistributionDataSource(base_dist, batch_size, seed=seed + 1))
    else:
        val_ds = source.to_dataset(batch_size=batch_size, shuffle=False, seed=seed + 1)
    val_batch = next(iter(val_ds.to_iter_dataset(read_options_val)))
    val_data = [val_batch]

    # Run name / paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{loss_type}_w{width}d{depth}_lr{lr}_b{batch_size}_s{seed}_{timestamp}"
    chkpt_run_path = ckpt_path / run_name
    tb_run_dir = tensorboard_log_dir / run_name

    hparams = {
        "loss_type": loss_type,
        "N": N,
        "d": d,
        "L": L,
        "width": width,
        "depth": depth,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "nsteps": nsteps,
    }

    # Callbacks — metric producers first, then consumers
    callbacks = []

    if ess and target_dist is not None:
        callbacks.append(
            ESSCallback(
                target_log_prob=target_dist.log_prob,
                base_distribution=base_dist,
                flow_kwargs=flow_kwargs,
                n_samples=ess_samples,
                eval_freq=ess_freq,
            )
        )

    if target_dist is not None:
        ref_species = jnp.array(source[0].species)
        callbacks.append(
            PotentialEnergyCallback(
                energy_fn=target_dist._energy_fn,
                base_distribution=base_dist,
                ref_species=ref_species,
                flow_kwargs=flow_kwargs,
                n_samples=batch_size,
                eval_freq=log_freq,
            )
        )

    save_freq = max(1, nsteps // num_checkpoints) if num_checkpoints > 0 else nsteps + 1
    callbacks += [
        ValidationCallback(val_data=val_data, loss_module=loss_fn, val_freq=log_freq),
        LoggerCallback(log_freq=log_freq),
        ProgressBarCallback(refresh_rate=max(1, nsteps // 100)),
        CheckpointCallback(ckpt_path=chkpt_run_path, save_freq=save_freq, overwrite=overwrite),
    ]

    if profile:
        callbacks.append(
            ProfilingCallback(
                log_dir=profile_log_dir,
                warmup_steps=profile_warmup,
                profile_steps=profile_steps,
            )
        )

    if tensorboard:
        callbacks.append(TensorBoardLogger(log_dir=tb_run_dir, log_freq=log_freq, hparams=hparams))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_module=loss_fn,
        seed=seed,
        callbacks=callbacks,
    )

    model_label = model_file.name if model_file else "none"
    print(f"\n{'='*60}")
    print("Training CNF on particle trajectories")
    print(f"    Data       : {data_path}")
    print(f"  Potential    : {model_label}")
    print(f"  N={N}, d={d}, L={L:.4f}")
    print(f"  Dataset size : {len(source)}")
    print(f"  Loss         : {loss_type}")
    print(f"  MLP          : width={width}, depth={depth}")
    print(f"  Parameters   : {num_params:,}")
    print(f"  Run          : {run_name}")
    print(f"  Ckpts        : {chkpt_run_path}")
    print(f"{'='*60}\n")

    t_start = time.time()
    read_options = grain.ReadOptions(num_threads=num_workers, prefetch_buffer_size=prefetch_buffer_size)
    trainer.train(
        dataset=dataset,
        max_steps=nsteps,
        read_options=read_options,
    )
    t_elapsed = time.time() - t_start

    print(f"\nDone in {t_elapsed:.1f}s ({1000*t_elapsed/nsteps:.0f}ms/step)\n")
    return trainer


@app.command()
def main(
    data_path: Annotated[Path, typer.Option("--data-path", help="Path to trajectory directory or file")] = ...,
    nsteps: Annotated[int, typer.Option("--nsteps", help="Number of training steps")] = ...,
    model_file: Annotated[
        Path | None, typer.Option("--model-file", help="JSON file describing the interatomic potential model")
    ] = None,
    loss_type: Annotated[
        str, typer.Option(help="Loss: maximum_likelihood | energy_based | hybrid")
    ] = "maximum_likelihood",
    width: Annotated[int, typer.Option(help="MLP hidden width")] = 64,
    depth: Annotated[int, typer.Option(help="MLP depth")] = 3,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    log_freq: Annotated[int, typer.Option(help="Logging frequency")] = 100,
    ckpt_path: Annotated[Path, typer.Option(help="Checkpoint root path")] = Path("tmp/ckpt_particles"),
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing checkpoints")] = True,
    num_checkpoints: Annotated[int, typer.Option(help="Number of checkpoints to save during training")] = 1,
    temperature: Annotated[
        float | None, typer.Option(help="Boltzmann temperature — required with --model-file")
    ] = None,
    ess: Annotated[bool, typer.Option(help="Enable ESS monitoring (requires --model-file)")] = False,
    ess_freq: Annotated[int, typer.Option(help="ESS evaluation frequency (steps)")] = 250,
    ess_samples: Annotated[int, typer.Option(help="Number of samples for ESS")] = 512,
    profile: Annotated[bool, typer.Option(help="Enable JAX profiling")] = False,
    profile_log_dir: Annotated[Path, typer.Option(help="Profiling trace directory")] = Path("tmp/profiles"),
    profile_warmup: Annotated[int, typer.Option(help="Warmup steps before profiling")] = 50,
    profile_steps: Annotated[int, typer.Option(help="Steps to profile (0=until end)")] = 0,
    tensorboard: Annotated[bool, typer.Option(help="Enable TensorBoard logging")] = False,
    tensorboard_log_dir: Annotated[Path, typer.Option(help="TensorBoard log directory")] = Path("tmp/tb_logs"),
    device: Annotated[str | None, typer.Option(help="JAX device: cpu | gpu | None")] = None,
    num_workers: Annotated[int, typer.Option(help="Grain data loading threads")] = 1,
    prefetch_buffer_size: Annotated[int, typer.Option(help="Grain prefetch buffer size")] = 2,
    solver: Annotated[str, typer.Option(help="ODE solver (tsit5, dopri5, euler)")] = "tsit5",
    atol: Annotated[float, typer.Option(help="Absolute tolerance for adaptive solvers")] = 1e-5,
    rtol: Annotated[float, typer.Option(help="Relative tolerance for adaptive solvers")] = 1e-5,
    euler_steps: Annotated[
        int | None, typer.Option(help="Number of steps for Euler solver (required if solver=euler)")
    ] = None,
):
    """Train a CNF on MD particle trajectory data."""
    if device is not None:
        jax.config.update("jax_platform_name", device)
        print(f"JAX process: {jax.process_index()}/{jax.process_count()}")
        print(f"JAX devices: {jax.devices()}")

    train_single_model(
        data_path=data_path,
        model_file=model_file,
        loss_type=loss_type,
        width=width,
        depth=depth,
        lr=lr,
        nsteps=nsteps,
        batch_size=batch_size,
        seed=seed,
        log_freq=log_freq,
        ckpt_path=ckpt_path,
        overwrite=overwrite,
        num_checkpoints=num_checkpoints,
        temperature=temperature,
        ess=ess,
        ess_freq=ess_freq,
        ess_samples=ess_samples,
        profile=profile,
        profile_log_dir=profile_log_dir,
        profile_warmup=profile_warmup,
        profile_steps=profile_steps or None,
        tensorboard=tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
        num_workers=num_workers,
        prefetch_buffer_size=prefetch_buffer_size,
        solver=solver,
        atol=atol,
        rtol=rtol,
        euler_steps=euler_steps,
    )


if __name__ == "__main__":
    app()
