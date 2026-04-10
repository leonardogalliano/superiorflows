import time
from pathlib import Path

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import optax
import typer
from superiorflows import CoupledDataSource, DistributionDataSource
from superiorflows.train import (
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

app = typer.Typer(pretty_exceptions_show_locals=False)


class MLPVelocity(eqx.Module):
    """MLP velocity field with time conditioning."""

    mlp: eqx.nn.MLP

    def __init__(self, input_dim: int, width: int, depth: int, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 1,
            out_size=input_dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    @eqx.filter_jit
    def __call__(self, t, x, args):
        t_feat = jnp.broadcast_to(t, x.shape[:-1] + (1,))
        return self.mlp(jnp.concatenate([x, t_feat], axis=-1))


def train_single_model(
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
    profile: bool = False,
    profile_log_dir: Path = Path("tmp/profiles"),
    profile_warmup: int = 50,
    profile_steps: int | None = None,
    tensorboard: bool = False,
    tensorboard_log_dir: Path = Path("tmp/tb_logs"),
    ess: bool = False,
    ess_freq: int = 250,
    ess_samples: int = 1000,
    num_workers: int = 1,
    prefetch_buffer_size: int = 2,
):
    # Setup key
    key = jax.random.key(seed)

    # Target: 8 Gaussians in a circle
    d = 2
    angles = jnp.arange(8) * (jnp.pi / 4)
    locs = 10.0 * jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=1)
    target_dist = dsx.MixtureSameFamily(
        mixture_distribution=dsx.Categorical(probs=jnp.ones(8) / 8),
        components_distribution=dsx.MultivariateNormalDiag(loc=locs, scale_diag=jnp.full((8, 2), 0.7)),
    )
    # Base: Standard Gaussian
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))

    # Loss Setup
    flow_kwargs = dict(stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))

    if loss_type == "maximum_likelihood":
        loss_fn = MaximumLikelihoodLoss(base_distribution=base_dist, **flow_kwargs)
        dataset = grain.MapDataset.source(DistributionDataSource(target_dist, batch_size, seed=seed)).repeat()
    elif loss_type == "energy_based":
        loss_fn = EnergyBasedLoss(base_distribution=base_dist, target_distribution=target_dist, **flow_kwargs)
        dataset = grain.MapDataset.source(DistributionDataSource(base_dist, batch_size, seed=seed)).repeat()
    elif loss_type == "hybrid":
        loss_fn = KullbackLeiblerLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            alpha=0.5,
            **flow_kwargs,
        )
        dataset = grain.MapDataset.source(DistributionDataSource(target_dist, batch_size, seed=seed)).repeat()
    elif loss_type == "stochastic_interpolant":

        def interpolant(t, x0, x1):
            return (1 - t) * x0 + t * x1

        def gamma_fn(t):
            return jnp.sqrt(2 * t * (1 - t))

        loss_fn = StochasticInterpolantLoss(interpolant=interpolant, gamma=gamma_fn)
        dataset = grain.MapDataset.source(
            CoupledDataSource(
                DistributionDataSource(base_dist, batch_size, seed=seed),
                DistributionDataSource(target_dist, batch_size, seed=seed + 1),
            )
        ).repeat()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Model
    key, model_key = jax.random.split(key)
    model = MLPVelocity(input_dim=d, width=width, depth=depth, key=model_key)

    # Optimizer
    optimizer = optax.adam(lr)

    # Validation data (fixed for consistency)
    key, val_key = jax.random.split(key)
    if loss_type == "stochastic_interpolant":
        val_key1, val_key2 = jax.random.split(val_key)
        val_data = [
            (
                base_dist.sample(seed=val_key1, sample_shape=(1000,)),
                target_dist.sample(seed=val_key2, sample_shape=(1000,)),
            )
        ]
    else:
        val_data = [target_dist.sample(seed=val_key, sample_shape=(1000,))]

    # Construct unique run name for TensorBoard
    # Convention: {loss_type}_w{width}d{depth}_lr{lr}_s{seed}_{timestamp}
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{loss_type}_" f"w{width}d{depth}_" f"lr{lr}_" f"b{batch_size}_" f"s{seed}_" f"{timestamp}"
    chkpt_run_path = ckpt_path / run_name
    tb_run_dir = tensorboard_log_dir / run_name

    # HParams dictionary
    hparams = {
        "loss_type": loss_type,
        "width": width,
        "depth": depth,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "nsteps": nsteps,
    }

    # Callbacks — metric producers first, then consumers
    callbacks = []

    if ess:
        callbacks.append(
            ESSCallback(
                target_log_prob=target_dist.log_prob,
                base_distribution=base_dist,
                flow_kwargs=flow_kwargs,
                n_samples=ess_samples,
                eval_freq=ess_freq,
            )
        )

    callbacks += [
        ValidationCallback(val_data=val_data, loss_module=loss_fn, val_freq=500),
        LoggerCallback(log_freq=log_freq),
        ProgressBarCallback(refresh_rate=50),
        CheckpointCallback(ckpt_path=chkpt_run_path, save_freq=500, overwrite=overwrite),
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

    print(f"\n{'='*60}")
    print("Training 8 Gaussians")
    print(f"Run Name: {run_name}")
    print(f"Loss: {loss_type}")
    print(f"Checkpoints: {chkpt_run_path}")
    print(f"TensorBoard: {tb_run_dir}")
    print(f"{'='*60}")

    t_start = time.time()
    read_options = grain.ReadOptions(num_threads=num_workers, prefetch_buffer_size=prefetch_buffer_size)
    trainer.train(dataset=dataset, max_steps=nsteps, read_options=read_options)
    t_elapsed = time.time() - t_start

    print(f"Done in {t_elapsed:.1f}s ({1000*t_elapsed/nsteps:.0f}ms/step)")
    return trainer


@app.command()
def main(
    loss_type: Annotated[
        str,
        typer.Option(
            help="Type of loss function: 'maximum_likelihood', 'energy_based', 'hybrid', 'stochastic_interpolant'"
        ),
    ] = "maximum_likelihood",
    width: Annotated[int, typer.Option(help="Width of the MLP")] = 16,
    depth: Annotated[int, typer.Option(help="Depth of the MLP")] = 3,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    nsteps: Annotated[int, typer.Option(help="Number of training steps")] = 5000,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    log_freq: Annotated[int, typer.Option(help="Logging frequency")] = 500,
    ckpt_path: Annotated[Path, typer.Option(help="Checkpoint path")] = Path("tmp/ckpt_8gaussians"),
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing checkpoints")] = True,
    profile: Annotated[bool, typer.Option(help="Enable JAX profiling")] = False,
    profile_log_dir: Annotated[Path, typer.Option(help="Directory to save profiling traces")] = Path("tmp/tb_logs"),
    profile_warmup: Annotated[int, typer.Option(help="Steps to wait before starting profiling")] = 50,
    profile_steps: Annotated[int, typer.Option(help="Number of steps to profile (0=until end)")] = 0,
    tensorboard: Annotated[bool, typer.Option(help="Enable TensorBoard logging")] = False,
    tensorboard_log_dir: Annotated[Path, typer.Option(help="TensorBoard log directory")] = Path("tmp/tb_logs"),
    ess: Annotated[bool, typer.Option(help="Enable ESS monitoring")] = False,
    ess_freq: Annotated[int, typer.Option(help="ESS evaluation frequency (steps)")] = 250,
    ess_samples: Annotated[int, typer.Option(help="Number of samples for ESS estimation")] = 1000,
    device: Annotated[str | None, typer.Option(help="JAX device: 'cpu', 'gpu', or None (auto)")] = None,
    num_workers: Annotated[int, typer.Option(help="Number of Grain data loading threads (match --cpus-per-task)")] = 1,
    prefetch_buffer_size: Annotated[int, typer.Option(help="Grain prefetch buffer size")] = 2,
):
    """
    Train a flow on the 8 Gaussians problem.
    """
    if device is not None:
        jax.config.update("jax_platform_name", device)
        print(f"JAX process: {jax.process_index()}/{jax.process_count()}")
        print(f"JAX devices: {jax.devices()}")

    train_single_model(
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
        profile=profile,
        profile_log_dir=profile_log_dir,
        profile_warmup=profile_warmup,
        profile_steps=profile_steps or None,
        tensorboard=tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
        ess=ess,
        ess_freq=ess_freq,
        ess_samples=ess_samples,
        num_workers=num_workers,
        prefetch_buffer_size=prefetch_buffer_size,
    )


if __name__ == "__main__":
    app()
