"""Sample from a trained CNF particle model."""

import json
import time
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import orbax.checkpoint as ocp
import typer
from superiorflows import Flow

from particle_systems.particle_system import (
    TrajectoryDataSource,
    UniformParticles,
    batch_to_trajectory,
)
from particle_systems.training_particles import build_solver, build_velocity

app = typer.Typer(pretty_exceptions_show_locals=False)


def load_trained_flow(ckpt_path: Path):
    """Load a Flow and metadata from a given checkpoint directory."""
    config_file = ckpt_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config.json in {ckpt_path}")

    with open(config_file, "r") as f:
        config = json.load(f)

    # Target metadata from TrajectoryDataSource
    data_path = Path(config["data"]["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"data_path {data_path} specified in config.json not found.")

    source = TrajectoryDataSource(data_path)
    N, d = source.N, source.d
    L = float(source.box_size[0])
    ref_species = source[0].species
    composition = np.bincount(ref_species) / len(ref_species)
    n_species = len(composition)

    # Base distribution
    base_dist = UniformParticles(N=N, d=d, L=L, composition=composition)

    # Model structure and Restore weights
    key = jax.random.PRNGKey(0)
    velocity_field = build_velocity(config, N, d, n_species, key=key)

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
    flow_kwargs = build_solver(config)
    flow = Flow(velocity_field=trained_velocity_field, base_distribution=base_dist, **flow_kwargs)

    return flow, N, d, L, composition


@app.command()
def main(
    ckpt_path: Path = typer.Argument(
        ...,
        help="Path to checkpoint directory (containing config.json & checkpoint)",
        exists=True,
        dir_okay=True,
    ),
    batch_size: int = typer.Option(64, help="Number of samples per batch (per trajectory)"),
    num_trajectories: int = typer.Option(10, help="Number of trajectories (batches) to generate"),
    output_path: Path = typer.Option("tmp/sampled_particles", help="Directory where samples will be stored"),
    seed: int = typer.Option(0, help="Random seed for sample generation"),
    device: str = typer.Option(None, help="JAX device: cpu | gpu"),
):
    """Generate new samples from a trained CNF particle model."""
    if device is not None:
        jax.config.update("jax_platform_name", device)
        print(f"JAX process: {jax.process_index()}/{jax.process_count()}")
        print(f"JAX devices: {jax.devices()}")

    print(f"\n{'='*60}")
    print("Sampling from CNF particle model")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Batch size : {batch_size}")
    print(f"  Trajectories: {num_trajectories}")
    print(f"  Output path: {output_path}")
    print(f"  Seed       : {seed}")
    print(f"{'='*60}\n")

    print("Loading trained model...")
    t0 = time.time()
    flow, N, d, L, composition = load_trained_flow(ckpt_path)
    t1 = time.time()
    print(f"Loaded successfully in {t1 - t0:.1f}s. Model handles N={N}, d={d}, L={L:.4f}")

    run_output_dir = output_path / ckpt_path.name / f"seed_{seed}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(seed)

    print("Precompiling JAX graph...")
    t_comp = time.time()

    @jax.jit
    def sample_batch(rng):
        return flow.sample_and_log_prob(seed=rng, sample_shape=(batch_size,))

    compiled_sample = sample_batch.lower(key).compile()
    print(f"Compiled in {time.time() - t_comp:.1f}s")
    print("Starting generation...")

    total_time = 0.0
    for i in range(1, num_trajectories + 1):
        key, subkey = jax.random.split(key)

        print(f"[{i}/{num_trajectories}] Generating batch... ", end="", flush=True)
        t_start = time.time()
        samples, log_probs = compiled_sample(subkey)
        t_batch = time.time() - t_start
        total_time += t_batch
        print(f"Done in {t_batch:.2f}s")

        trajectory_dir = run_output_dir / str(i)
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        trj = batch_to_trajectory(samples)
        trj.metadata = {"generated_by": ckpt_path.name}

        import atooms.trajectory

        out_path = trajectory_dir / "samples.xyz"
        with atooms.trajectory.TrajectoryXYZ(str(out_path), "w") as out:
            out.metadata = trj.metadata
            for sys in trj:
                out.write(sys)

        lps = np.asarray(log_probs)
        np.savetxt(trajectory_dir / "log_probs.dat", lps, fmt="%.6e")

    print(f"\nAll done! Generated {batch_size * num_trajectories} samples in {total_time:.1f}s.")
    print(f"Saved to {run_output_dir.resolve()}")


if __name__ == "__main__":
    app()
