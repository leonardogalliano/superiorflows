import time
from pathlib import Path

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import typer
from superiorflows import DistributionDataSource
from superiorflows.train import (
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
        data_source = DistributionDataSource(target_dist, batch_size, seed=seed)
    elif loss_type == "energy_based":
        loss_fn = EnergyBasedLoss(base_distribution=base_dist, target_distribution=target_dist, **flow_kwargs)
        data_source = DistributionDataSource(base_dist, batch_size, seed=seed)
    elif loss_type == "hybrid":
        loss_fn = KullbackLeiblerLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            alpha=0.5,
            **flow_kwargs,
        )
        data_source = DistributionDataSource(target_dist, batch_size, seed=seed)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Model
    key, model_key = jax.random.split(key)
    model = MLPVelocity(input_dim=d, width=width, depth=depth, key=model_key)

    # Optimizer
    optimizer = optax.adam(lr)

    # Validation loader (fixed for consistency)
    key, val_key = jax.random.split(key)
    val_data = target_dist.sample(seed=val_key, sample_shape=(1000,))
    val_loader = [val_data]

    # Custom callback to collect loss history
    class LossHistoryCallback:
        def __init__(self):
            self.losses = []
            self.val_losses = []

        def on_step_end(self, trainer, step, logs, **kwargs):
            self.losses.append(float(logs["loss"]))
            if "val_loss" in logs:
                self.val_losses.append(float(logs["val_loss"]))

    history_cb = LossHistoryCallback()

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
        LoggerCallback(log_freq=log_freq),
        ProgressBarCallback(refresh_rate=50),
        CheckpointCallback(ckpt_path=ckpt_path, save_freq=500, overwrite=overwrite),
        history_cb,
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
        callbacks.append(TensorBoardLogger(log_dir=tensorboard_log_dir, log_freq=log_freq))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_module=loss_fn,
        seed=seed,
        callbacks=callbacks,
    )

    print(f"\n{'='*60}")
    print("Training 8 Gaussians")
    print(f"Loss: {loss_type}")
    print(f"Checkpoints: {ckpt_path}")
    print(f"{'='*60}")

    t_start = time.time()
    trainer.train(data_source=data_source, val_loader=val_loader, max_steps=nsteps, val_freq=250)
    t_elapsed = time.time() - t_start

    print(f"Done in {t_elapsed:.1f}s ({1000*t_elapsed/nsteps:.0f}ms/step)")
    return trainer, history_cb.losses, history_cb.val_losses


@app.command()
def main(
    loss_type: Annotated[
        str, typer.Option(help="Type of loss function: 'maximum_likelihood', 'energy_based', 'hybrid'")
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
    profile_log_dir: Annotated[Path, typer.Option(help="Directory to save profiling traces")] = Path("tmp/profiles"),
    profile_warmup: Annotated[int, typer.Option(help="Steps to wait before starting profiling")] = 50,
    profile_steps: Annotated[int, typer.Option(help="Number of steps to profile (0=until end)")] = 0,
    tensorboard: Annotated[bool, typer.Option(help="Enable TensorBoard logging")] = False,
    tensorboard_log_dir: Annotated[Path, typer.Option(help="TensorBoard log directory")] = Path("tmp/tb_logs"),
    ess: Annotated[bool, typer.Option(help="Enable ESS monitoring")] = False,
    ess_freq: Annotated[int, typer.Option(help="ESS evaluation frequency (steps)")] = 250,
    ess_samples: Annotated[int, typer.Option(help="Number of samples for ESS estimation")] = 1000,
    device: Annotated[str | None, typer.Option(help="JAX device: 'cpu', 'gpu', or None (auto)")] = None,
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
    )


if __name__ == "__main__":
    app()
