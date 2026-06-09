"""Sample from a trained Gaussian mixture CNF model."""

import json
import time
from pathlib import Path

import diffrax as dfx
import distreqx.distributions as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import typer

from scripts.gaussian_mixture.train_gaussian_mixture import build_solver, build_velocity
from superiorflows import Flow, ODEBijector

app = typer.Typer(pretty_exceptions_show_locals=False)


def load_trained_flow(ckpt_path: Path, **flow_kwargs):
    config_file = ckpt_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config.json in {ckpt_path}")

    with open(config_file, "r") as f:
        config = json.load(f)

    tgcfg = config["target"]
    d = tgcfg["d"]
    a = tgcfg["a"]

    # Base distribution
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))

    # Model structure and Restore weights
    key = jax.random.key(0)
    velocity_field = build_velocity(config, d, key=key)

    model_params = eqx.filter(velocity_field, eqx.is_array)
    static_model = eqx.filter(velocity_field, eqx.is_array, inverse=True)

    checkpointer = ocp.CheckpointManager(ckpt_path.resolve(), item_names=("model", "optimizer", "metadata"))
    step = checkpointer.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found at {ckpt_path}")

    restore_args = ocp.args.Composite(model=ocp.args.StandardRestore(model_params))
    restored = checkpointer.restore(step, args=restore_args)
    trained_velocity_field = eqx.combine(restored.model, static_model)

    # Bind into Flow
    base_flow_kwargs = build_solver(config)
    base_flow_kwargs.update(flow_kwargs)
    bijector = ODEBijector(trained_velocity_field, **base_flow_kwargs)
    flow = Flow(bijector, base_dist)

    return flow, d, a, config


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
    seed: int = typer.Option(0, help="Random seed for sample generation"),
    device: str = typer.Option(None, help="JAX device: cpu | gpu"),
    solver: str = typer.Option(None, help="Solver type (euler, tsit5, dopri5)"),
    tolerance: float = typer.Option(None, help="Solver tolerance (sets both atol and rtol)"),
    solver_steps: int = typer.Option(None, help="Number of steps for fixed-step solvers"),
    hutchinson_samples: int = typer.Option(None, help="Number of Hutchinson samples for divergence estimation."),
    ignore_density: bool = typer.Option(
        False, "--ignore-density", help="If True, only sample configurations, skipping log-probability computation."
    ),
):
    """Generate new samples from a trained Gaussian mixture CNF model."""
    if device is not None:
        jax.config.update("jax_platform_name", device)

    print(f"\n{'=' * 60}")
    print("Sampling from CNF Gaussian Mixture model")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Total samples  : {num_samples}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Output path    : {output_path}")
    print(f"  Seed           : {seed}")
    if solver:
        print(f"  Solver         : {solver}")
    if tolerance:
        print(f"  Tolerance      : {tolerance}")
    if solver_steps:
        print(f"  Solver steps   : {solver_steps}")
    if hutchinson_samples:
        print(f"  Hutchinson     : {hutchinson_samples}")
    print(f"  Ignore density : {ignore_density}")
    print(f"  JAX process    : {jax.process_index()}/{jax.process_count()}")
    print(f"  JAX devices    : {jax.devices()}")
    print(f"{'=' * 60}\n")

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

    flow, d, a, config = load_trained_flow(ckpt_path, **flow_kwargs)
    t1 = time.time()
    print(f"Loaded successfully in {t1 - t0:.1f}s. Model handles d={d}")

    key = jax.random.key(seed)

    print("Precompiling JAX graph...")
    t_comp = time.time()

    @eqx.filter_jit
    def sample_batch(rng, current_batch_size):
        if ignore_density:
            sample_keys = jax.random.split(rng, current_batch_size)
            x0 = jax.vmap(flow.base_distribution.sample)(sample_keys)
            x1 = jax.vmap(flow.bijector.forward)(x0)
            return x1, None

        if flow.bijector.hutchinson_samples is not None:
            key1, key2 = jax.random.split(rng)
            sample_keys = jax.random.split(key1, current_batch_size)
            x0 = jax.vmap(flow.base_distribution.sample)(sample_keys)
            keys = jax.random.split(key2, current_batch_size)
            x1, log_probs = jax.vmap(lambda x, k: flow.push_forward_and_log_prob(x, key=k))(x0, keys)
        else:
            sample_keys = jax.random.split(rng, current_batch_size)
            x0 = jax.vmap(flow.base_distribution.sample)(sample_keys)
            x1, log_probs = jax.vmap(flow.push_forward_and_log_prob)(x0)
        return x1, log_probs

    print(f"Compiled in {time.time() - t_comp:.1f}s")
    print("Starting generation...")

    total_time = 0.0
    all_samples = []
    all_log_probs = []

    num_batches = (num_samples + batch_size - 1) // batch_size
    samples_generated = 0

    for i in range(num_batches):
        key, subkey = jax.random.split(key)

        current_batch_size = min(batch_size, num_samples - samples_generated)

        print(f"  [{i + 1}/{num_batches}] Generating {current_batch_size} samples... ", end="", flush=True)
        t_start = time.time()

        samples, log_probs = sample_batch(subkey, current_batch_size)
        samples.block_until_ready()
        if log_probs is not None:
            log_probs.block_until_ready()

        t_batch = time.time() - t_start
        total_time += t_batch
        print(f"Done in {t_batch:.2f}s")

        all_samples.append(np.asarray(samples))
        if log_probs is not None:
            all_log_probs.append(np.asarray(log_probs))

        samples_generated += current_batch_size

    final_samples = np.concatenate(all_samples, axis=0)

    save_dict = {"samples": final_samples}
    if not ignore_density:
        final_log_probs = np.concatenate(all_log_probs, axis=0)
        save_dict["log_probs"] = final_log_probs

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(output_path, **save_dict)

    print(f"\nAll done! Generated {num_samples} samples in {total_time:.1f}s.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    app()
