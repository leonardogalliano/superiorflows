"""
Compare different training strategies for CNF on the 8 Gaussians task.

This script trains models using:
1. Maximum Likelihood (forward KL)
2. Energy-based (reverse KL)
3. Hybrid KL (alpha-blended)

And compares their convergence and sample quality.
"""
import importlib.util
import sys
from pathlib import Path

import diffrax as dfx
import distrax as dsx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from superiorflows import Flow

# Import 8_gaussians module dynamically
script_path = Path(__file__).parent / "8_gaussians.py"
spec = importlib.util.spec_from_file_location("eight_gaussians", script_path)
eight_gaussians = importlib.util.module_from_spec(spec)
sys.modules["eight_gaussians"] = eight_gaussians
spec.loader.exec_module(eight_gaussians)

# Expose necessary functions/classes
train_single_model = eight_gaussians.train_single_model
MLPVelocity = eight_gaussians.MLPVelocity


def main():
    matplotlib.use("Agg")
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    # Random seed
    key = jax.random.key(42)

    # Common parameters
    nsteps = 5000
    width = 16
    depth = 3
    batch_size = 32
    log_freq = 500

    # =================================================================
    # Train three models using the imported engine
    # =================================================================

    # 1. MLE
    trainer_mle, losses_mle, val_losses_mle = train_single_model(
        loss_type="maximum_likelihood",
        width=width,
        depth=depth,
        lr=1e-3,
        nsteps=nsteps,
        batch_size=batch_size,
        seed=0,
        log_freq=log_freq,
        ckpt_path=Path("tmp/ckpt_method_mle"),
        overwrite=True,
    )
    model_mle = trainer_mle.model

    # 2. Energy
    trainer_energy, losses_energy, val_losses_energy = train_single_model(
        loss_type="energy_based",
        width=width,
        depth=depth,
        lr=1e-3,
        nsteps=nsteps,
        batch_size=batch_size,
        seed=0,
        log_freq=log_freq,
        ckpt_path=Path("tmp/ckpt_method_energy"),
        overwrite=True,
    )
    model_energy = trainer_energy.model

    # 3. Hybrid
    trainer_hybrid, losses_hybrid, val_losses_hybrid = train_single_model(
        loss_type="hybrid",
        width=width,
        depth=depth,
        lr=1e-3,
        nsteps=nsteps,
        batch_size=batch_size,
        seed=0,
        log_freq=log_freq,
        ckpt_path=Path("tmp/ckpt_method_hybrid"),
        overwrite=True,
    )
    model_hybrid = trainer_hybrid.model

    # =================================================================
    # Reload from checkpoints to verify
    # =================================================================
    print("\nReloading best models from checkpoints...")
    trainer_mle.load_checkpoint("tmp/ckpt_method_mle")
    model_mle = trainer_mle.model

    trainer_energy.load_checkpoint("tmp/ckpt_method_energy")
    model_energy = trainer_energy.model

    trainer_hybrid.load_checkpoint("tmp/ckpt_method_hybrid")
    model_hybrid = trainer_hybrid.model

    # =================================================================
    # Visualizations (Losses, Samples, Animation)
    # =================================================================

    # Setup Target Distribution for plotting (Recreating here as it's needed for plotting)
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

    # --- Plot Losses ---
    fig, ax = plt.subplots(figsize=(10, 6))

    def smooth(x, window=50):
        kernel = jnp.ones(window) / window
        return jnp.convolve(jnp.array(x), kernel, mode="valid")

    steps = range(len(smooth(losses_mle)))
    ax.plot(steps, smooth(losses_mle), label="Maximum Likelihood", linewidth=2)
    ax.plot(steps, smooth(losses_energy), label="Energy-Based", linewidth=2)
    ax.plot(steps, smooth(losses_hybrid), label="Hybrid KL", linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss (smoothed)", fontsize=12)
    ax.set_title("Training Loss Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_comparison_losses.png", dpi=150)
    print(f"\nSaved: {output_dir / 'training_comparison_losses.png'}")

    # --- Plot Samples ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Compute target density on a grid
    grid_size = 100
    x_grid = jnp.linspace(-15, 15, grid_size)
    y_grid = jnp.linspace(-15, 15, grid_size)
    xx, yy = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    log_probs = jax.vmap(target_dist.log_prob)(grid_points)
    density = jnp.exp(log_probs).reshape(grid_size, grid_size)

    # Sample from each trained model
    key, k1, k2, k3 = jax.random.split(key, 4)
    n_samples = 500

    def get_samples(model, key):
        flow = Flow(velocity_field=model, base_distribution=base_dist)
        x0 = base_dist.sample(seed=key, sample_shape=(n_samples,))
        return jax.vmap(flow.apply_map)(x0)

    samples_mle = get_samples(model_mle, k1)
    samples_energy = get_samples(model_energy, k2)
    samples_hybrid = get_samples(model_hybrid, k3)

    # Plot target
    ax = axes[0]
    ax.contourf(xx, yy, density, levels=20, cmap="Blues", alpha=0.6)
    ax.contour(xx, yy, density, levels=10, colors="steelblue", linewidths=0.5, alpha=0.8)
    ax.set_title("Target Distribution", fontsize=12)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")

    # Plot samples
    titles = ["Maximum Likelihood", "Energy-Based", "Hybrid KL"]
    samples_list = [samples_mle, samples_energy, samples_hybrid]

    for i, (ax, title, samples) in enumerate(zip(axes[1:], titles, samples_list)):
        ax.contourf(xx, yy, density, levels=20, cmap="Blues", alpha=0.3)
        ax.scatter(samples[:, 0], samples[:, 1], s=8, c="orangered", alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "training_comparison_samples.png", dpi=150)
    print(f"Saved: {output_dir / 'training_comparison_samples.png'}")

    # --- Generate Animation ---
    print("\nGenerating side-by-side animation...")

    # Simulation settings
    n_particles = 300
    n_frames = 40
    save_times = jnp.linspace(0.0, 1.0, n_frames)

    key, subkey = jax.random.split(key)
    x0 = base_dist.sample(seed=subkey, sample_shape=(n_particles,))

    # Integrate methods
    def get_trajectories(model):
        flow = Flow(velocity_field=model, base_distribution=base_dist)
        # return shape: (n_particles, n_frames, dim)
        return jax.vmap(lambda x: flow.integrate(x, saveat=dfx.SaveAt(ts=save_times)).ys)(x0)

    traj_mle = get_trajectories(model_mle)
    traj_energy = get_trajectories(model_energy)
    traj_hybrid = get_trajectories(model_hybrid)

    traj_list = [traj_mle, traj_energy, traj_hybrid]
    titles = ["Maximum Likelihood", "Energy-Based", "Hybrid KL"]

    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    scatters = []
    for i, ax in enumerate(axes):
        ax.contourf(xx, yy, density, levels=20, cmap="Blues", alpha=0.3)
        ax.contour(xx, yy, density, levels=10, colors="steelblue", linewidths=0.5, alpha=0.6)

        scat = ax.scatter([], [], s=8, c="orangered", alpha=0.7)
        scatters.append(scat)

        ax.set_title(titles[i], fontsize=12)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")

    def animate(frame):
        for i, scat in enumerate(scatters):
            data = traj_list[i][:, frame, :]
            scat.set_offsets(data)
        return scatters

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=80, blit=True)
    ani.save(output_dir / "training_comparison_movie.gif", writer="pillow", fps=15)
    print(f"Saved: {output_dir / 'training_comparison_movie.gif'}")

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
