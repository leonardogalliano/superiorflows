"""Train a CNF on a Gaussian mixture (Louis mixture).

Supports layered configuration: defaults -> JSON config -> CLI overrides.
"""

import copy
import datetime
import json
import time
from pathlib import Path
from typing import Any

import diffrax as dfx
import distreqx.distributions as dsx
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import optax
import typer
from typing_extensions import Annotated

from superiorflows import CoupledDataSource, DistributionDataSource, ODEBijector
from superiorflows.train import (
    CheckpointCallback,
    EnergyBasedLoss,
    ESSCallback,
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
    "target": {
        "d": 2,
        "a": 2.0,
        "sigma2_max": 0.2,
        "sigma2_min": 0.01,
        "weight": 2.0 / 3.0,
    },
    "data": {
        "num_workers": 1,
        "prefetch_buffer_size": 2,
    },
    "training": {
        "nsteps": None,
        "batch_size": 128,
        "seed": 0,
        "loss_type": "maximum_likelihood",
        "log_freq": 100,
        "ckpt_path": "tmp/ckpt_gaussian_mixture",
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
        "type": "mlp",
        "width": 64,
        "depth": 3,
        "time_embedding_dim": 0,
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
        "learn_denoiser": False,
        "denoiser_weight": 1.0,
    },
    "callbacks": {
        "ess": {
            "enabled": False,
            "freq": 250,
            "samples": 1000,
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


def build_louis_mixture(d: int, a: float, sigma2_max=0.2, sigma2_min=0.01, weight=2.0 / 3.0):
    i = jnp.arange(1, d + 1)
    var1 = (i / d) * sigma2_max + ((d - i) / d) * sigma2_min
    var2 = var1[::-1]
    std1 = jnp.sqrt(var1)
    std2 = jnp.sqrt(var2)

    locs = jnp.stack([-a * jnp.ones(d), a * jnp.ones(d)])
    scales = jnp.stack([std1, std2])
    components = eqx.filter_vmap(dsx.MultivariateNormalDiag)(locs, scales)

    mixing = dsx.Categorical(probs=jnp.array([weight, 1.0 - weight]))

    target_distribution = dsx.MixtureSameFamily(mixture_distribution=mixing, components_distribution=components)

    return target_distribution


def sinusoidal_time_embedding(t: float, dim: int, max_period: float = 10000.0) -> jnp.ndarray:
    half_dim = dim // 2
    frequencies = jnp.exp(-jnp.log(max_period) * jnp.arange(half_dim) / (half_dim - 1))
    angles = t * frequencies
    embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if dim % 2 == 1:
        embedding = jnp.concatenate([embedding, jnp.zeros((1,))], axis=-1)
    return embedding


class VelocityDenoiserPair(eqx.Module):
    """Container holding a velocity field and a denoiser for joint training."""

    velocity_field: eqx.Module
    denoiser: eqx.Module


class MLPVelocity(eqx.Module):
    """MLP velocity field for unbounded domains, with optional time embedding."""

    mlp: eqx.nn.MLP
    time_embedding_dim: int = eqx.field(static=True)

    def __init__(self, d: int, width: int, depth: int, time_embedding_dim: int = 0, *, key):
        self.time_embedding_dim = time_embedding_dim
        if time_embedding_dim > 0:
            in_features = d + time_embedding_dim
        else:
            in_features = d + 1

        self.mlp = eqx.nn.MLP(
            in_size=in_features,
            out_size=d,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t: float, x: jnp.ndarray, args=None) -> jnp.ndarray:
        if self.time_embedding_dim > 0:
            t_emb = sinusoidal_time_embedding(t, self.time_embedding_dim)
        else:
            t_emb = jnp.array([t])

        # broadcast t_emb if x is batched (handled by vmap externally usually, but just in case)
        t_feat = jnp.broadcast_to(t_emb, x.shape[:-1] + t_emb.shape)

        features = jnp.concatenate([x, t_feat], axis=-1)
        return self.mlp(features)


def build_model(config: dict, d: int, *, key):
    vcfg = config["velocity"]
    if vcfg["type"] != "mlp":
        raise ValueError(f"Only mlp velocity is supported for now, got {vcfg['type']}")

    time_emb_dim = vcfg.get("time_embedding_dim", 0)
    loss_type = config["training"]["loss_type"]
    learn_denoiser = config["stochastic_interpolant"].get("learn_denoiser", False)

    if learn_denoiser and loss_type == "stochastic_interpolant":
        vel_key, den_key = jax.random.split(key)
        vel = MLPVelocity(d=d, width=vcfg["width"], depth=vcfg["depth"], time_embedding_dim=time_emb_dim, key=vel_key)
        den = MLPVelocity(d=d, width=vcfg["width"], depth=vcfg["depth"], time_embedding_dim=time_emb_dim, key=den_key)
        return VelocityDenoiserPair(velocity_field=vel, denoiser=den)
    else:
        if learn_denoiser:
            print("Warning: learn_denoiser=True is only supported for loss_type='stochastic_interpolant'. Ignoring.")
        return MLPVelocity(d=d, width=vcfg["width"], depth=vcfg["depth"], time_embedding_dim=time_emb_dim, key=key)


def build_solver(config: dict) -> dict:
    scfg = config["solver"]
    stype = scfg["type"].lower()
    solver_steps = scfg.get("solver_steps")

    solvers = {"euler": dfx.Euler, "tsit5": dfx.Tsit5, "dopri5": dfx.Dopri5}

    if stype not in solvers:
        raise ValueError(f"Unknown solver '{stype}'. Available: {list(solvers)}")

    if stype == "euler" and solver_steps is None:
        raise ValueError("solver.solver_steps is required when solver.type='euler'.")

    slv = solvers[stype]()

    bijector_kwargs: dict[str, Any] = dict(
        solver=slv,
        augmented_solver=slv,
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


def build_schedule_fn(expr: str):
    if not expr or expr.lower() in ("none", "null"):
        return None
    env = {"jnp": jnp}

    def schedule(t):
        return eval(expr, env, {"t": t})

    return schedule


def build_optimizer(config: dict) -> tuple:
    ocfg = config["optimizer"]
    lr_expr = ocfg["lr_schedule"]
    otype = ocfg["type"]

    env = {"optax": optax}
    try:
        schedule = optax.constant_schedule(float(lr_expr))
    except (ValueError, TypeError):
        schedule = eval(lr_expr, env)

    optimizers = {"adam": optax.adam, "adamw": optax.adamw, "sgd": optax.sgd}
    if otype not in optimizers:
        raise ValueError(f"Unknown optimizer type '{otype}'")

    base_opt = optimizers[otype](learning_rate=schedule)

    clip_cfg = ocfg.get("clip")
    if clip_cfg is not None:
        if isinstance(clip_cfg, (int, float)):
            clip_transform = optax.clip_by_global_norm(float(clip_cfg))
        elif isinstance(clip_cfg, dict):
            clip_type = clip_cfg.get("type", "global_norm").lower()
            clip_value = float(clip_cfg["value"])
            if clip_type == "global_norm":
                clip_transform = optax.clip_by_global_norm(clip_value)
            elif clip_type == "value":
                clip_transform = optax.clip(clip_value)
            elif clip_type == "block_rms":
                clip_transform = optax.clip_by_block_rms(clip_value)
            else:
                raise ValueError(f"Unknown clipping type '{clip_type}'")
        else:
            raise TypeError("Optimizer 'clip' must be None, a number, or a dict.")
        optimizer = optax.chain(clip_transform, base_opt)
    else:
        optimizer = base_opt

    return optimizer, schedule


# ── Core training logic ──────────────────────────────────────────────────────


def train_single_model(config: dict):
    tcfg = config["training"]
    tgcfg = config["target"]
    dcfg = config["data"]

    nsteps = tcfg["nsteps"]
    batch_size = tcfg["batch_size"]
    seed = tcfg["seed"]
    loss_type = tcfg["loss_type"]
    log_freq = tcfg["log_freq"]
    ckpt_path = Path(tcfg["ckpt_path"])
    overwrite = tcfg["overwrite"]
    num_checkpoints = tcfg["num_checkpoints"]

    d = tgcfg["d"]
    a = tgcfg["a"]
    sigma2_max = tgcfg["sigma2_max"]
    sigma2_min = tgcfg["sigma2_min"]
    weight = tgcfg["weight"]

    key = jax.random.key(seed)

    # Distributions
    target_dist = build_louis_mixture(d, a, sigma2_max, sigma2_min, weight)
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))
    bijector_kwargs = build_solver(config)

    def make_bijector(m):
        vf = m.velocity_field if isinstance(m, VelocityDenoiserPair) else m
        return ODEBijector(vf, **bijector_kwargs)

    # Data pipeline uses infinite DistributionDataSource
    # Depending on loss type, we source from base or target or both
    if loss_type == "maximum_likelihood":
        loss_fn = MaximumLikelihoodLoss(base_distribution=base_dist, make_bijector=make_bijector)
        source = DistributionDataSource(target_dist, batch_size, seed=seed)
        dataset = grain.MapDataset.source(source).repeat()

    elif loss_type == "energy_based":
        loss_fn = EnergyBasedLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            make_bijector=make_bijector,
        )
        source = DistributionDataSource(base_dist, batch_size, seed=seed)
        dataset = grain.MapDataset.source(source).repeat()

    elif loss_type == "hybrid":
        loss_fn = KullbackLeiblerLoss(
            base_distribution=base_dist,
            target_distribution=target_dist,
            make_bijector=make_bijector,
            alpha=0.5,
        )
        source = DistributionDataSource(target_dist, batch_size, seed=seed)
        dataset = grain.MapDataset.source(source).repeat()

    elif loss_type == "stochastic_interpolant":
        s_expr = config["stochastic_interpolant"]["interpolant_scheduler"]
        gamma_expr = config["stochastic_interpolant"]["noise_scheduler"]

        s_fn = build_schedule_fn(s_expr)
        gamma_fn = build_schedule_fn(gamma_expr)

        def interpolant(t, x0, x1):
            s_t = s_fn(t) if s_fn is not None else t
            return (1 - s_t) * x0 + s_t * x1

        si_kwargs = {k: v for k, v in bijector_kwargs.items() if k in ("dynamic_mask",)}

        if config["stochastic_interpolant"].get("learn_denoiser", False):
            loss_fn = StochasticInterpolantLoss(
                interpolant=interpolant,
                gamma=gamma_fn,
                get_velocity=lambda m: m.velocity_field,
                get_denoiser=lambda m: m.denoiser,
                denoiser_weight=config["stochastic_interpolant"].get("denoiser_weight", 1.0),
                **si_kwargs,
            )
        else:
            loss_fn = StochasticInterpolantLoss(interpolant=interpolant, gamma=gamma_fn, **si_kwargs)

        source = CoupledDataSource(
            DistributionDataSource(base_dist, batch_size, seed=seed),
            DistributionDataSource(target_dist, batch_size, seed=seed + 1),
        )
        dataset = grain.MapDataset.source(source).repeat()
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")

    # Model
    key, model_key = jax.random.split(key)
    model = build_model(config, d, key=model_key)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))

    # Optimizer
    optimizer, lr_schedule = build_optimizer(config)

    # Validation
    val_key = jax.random.key(seed + 123)
    if loss_type == "stochastic_interpolant":
        val_k1, val_k2 = jax.random.split(val_key)
        val_batch = (
            jax.vmap(base_dist.sample)(jax.random.split(val_k1, batch_size)),
            jax.vmap(target_dist.sample)(jax.random.split(val_k2, batch_size)),
        )
    elif loss_type == "energy_based":
        val_batch = jax.vmap(base_dist.sample)(jax.random.split(val_key, batch_size))
    else:
        val_batch = jax.vmap(target_dist.sample)(jax.random.split(val_key, batch_size))

    val_data = [val_batch]

    # Run name and paths
    vcfg = config["velocity"]
    lr_schedule_str = config["optimizer"]["lr_schedule"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{loss_type}_d{d}_w{vcfg['width']}_b{batch_size}_s{seed}_{timestamp}"
    chkpt_run_path = ckpt_path / run_name

    chkpt_run_path.mkdir(parents=True, exist_ok=True)
    with open(chkpt_run_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Callbacks
    ccfg = config["callbacks"]
    callbacks = []

    if ccfg["ess"]["enabled"]:
        callbacks.append(
            ESSCallback(
                target_log_prob=target_dist.log_prob,
                base_distribution=base_dist,
                make_bijector=make_bijector,
                n_samples=ccfg["ess"]["samples"],
                eval_freq=ccfg["ess"]["freq"],
            )
        )

    # TensorBoard
    tb_cfg = ccfg["tensorboard"]
    tb_logger = None
    if tb_cfg["enabled"]:
        tb_run_dir = Path(tb_cfg["log_dir"]) / run_name
        hparams = {
            "loss_type": loss_type,
            "d": d,
            "a": a,
            "lr_schedule": lr_schedule_str,
            "batch_size": batch_size,
            "seed": seed,
            "nsteps": nsteps,
            "solver": config["solver"]["type"],
            "ess_enabled": ccfg["ess"]["enabled"],
        }
        tb_logger = TensorBoardLogger(log_dir=tb_run_dir, log_freq=tb_cfg.get("freq", log_freq), hparams=hparams)

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

    load_from_ckpt = tcfg.get("load_from_checkpoint")
    if load_from_ckpt is not None:
        ckpt_load_path = Path(load_from_ckpt)
        if not ckpt_load_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_load_path}")
        success = trainer.load_checkpoint(str(ckpt_load_path))
        if not success:
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_load_path}")
        trainer._restored_data_state = None

    print(f"\n{'=' * 60}")
    print("Training CNF on Gaussian Mixture (Louis)")
    print(f"  Target        : d={d}, a={a}, w={weight}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Loss          : {loss_type}")
    t_emb = vcfg.get("time_embedding_dim", 0)
    print(f"  Velocity      : MLP (width={vcfg['width']}, depth={vcfg['depth']}, time_emb_dim={t_emb})")
    print(f"  Parameters    : {num_params:,}")
    print(f"  Optimizer     : {config['optimizer']['type']} | lr_schedule = {lr_schedule_str}")
    print(f"  Run           : {run_name}")
    print(f"  Ckpts         : {chkpt_run_path}")
    if load_from_ckpt is not None:
        print(f"  Resumed from  : {load_from_ckpt} (step {trainer.step})")
        print(f"  Training      : step {trainer.step} -> {nsteps}")
    print(f"  JAX process   : {jax.process_index()}/{jax.process_count()}")
    print(f"  JAX devices   : {jax.devices()}")
    print(f"{'=' * 60}\n")

    t_start = time.time()
    read_options = grain.ReadOptions(num_threads=dcfg["num_workers"], prefetch_buffer_size=dcfg["prefetch_buffer_size"])
    trainer.train(dataset=dataset, max_steps=nsteps, read_options=read_options)
    t_elapsed = time.time() - t_start

    print(f"\nDone in {t_elapsed:.1f}s ({1000 * t_elapsed / nsteps:.0f}ms/step)\n")
    return trainer


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    config: Annotated[Path | None, typer.Option("--config", help="JSON config file")] = None,
    d: Annotated[int | None, typer.Option("--d", help="Dimension of the mixture")] = None,
    a: Annotated[float | None, typer.Option("--a", help="Mode separation")] = None,
    nsteps: Annotated[int | None, typer.Option("--nsteps", help="Number of training steps")] = None,
    loss_type: Annotated[str | None, typer.Option("--loss-type", help="Loss function type")] = None,
    lr: Annotated[float | None, typer.Option("--lr", help="Constant learning rate")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", help="Batch size")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    ess: Annotated[bool | None, typer.Option("--ess/--no-ess", help="Enable ESS monitoring")] = None,
    ess_samples: Annotated[int | None, typer.Option("--ess-samples", help="Samples for ESS estimation")] = None,
    tensorboard: Annotated[
        bool | None, typer.Option("--tensorboard/--no-tensorboard", help="Enable TensorBoard")
    ] = None,
    profile: Annotated[bool | None, typer.Option("--profile/--no-profile", help="Enable JAX profiling")] = None,
    ckpt_path: Annotated[str | None, typer.Option("--ckpt-path", help="Base directory for checkpoints")] = None,
    load_from_checkpoint: Annotated[
        str | None, typer.Option("--load-from-checkpoint", help="Checkpoint directory to resume from")
    ] = None,
):
    """Train a CNF on the Louis Gaussian Mixture target."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if config is not None:
        with open(config) as f:
            cfg = merge_config(cfg, json.load(f))

    if d is not None:
        cfg["target"]["d"] = d
    if a is not None:
        cfg["target"]["a"] = a
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
    if ess is not None:
        cfg["callbacks"]["ess"]["enabled"] = ess
    if ess_samples is not None:
        cfg["callbacks"]["ess"]["samples"] = ess_samples
    if tensorboard is not None:
        cfg["callbacks"]["tensorboard"]["enabled"] = tensorboard
    if profile is not None:
        cfg["callbacks"]["profile"]["enabled"] = profile
    if ckpt_path is not None:
        cfg["training"]["ckpt_path"] = ckpt_path
    if load_from_checkpoint is not None:
        cfg["training"]["load_from_checkpoint"] = load_from_checkpoint

    if cfg["training"]["nsteps"] is None:
        raise typer.BadParameter("Missing required config: training.nsteps (--nsteps)")

    train_single_model(cfg)


if __name__ == "__main__":
    app()
