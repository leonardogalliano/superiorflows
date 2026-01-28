"""
Compare different training strategies for CNF on the 8 Gaussians task.

This script trains models using:
1. Maximum Likelihood (forward KL)
2. Energy-based (reverse KL)
3. Hybrid KL (alpha-blended)

And compares their convergence and sample quality.
"""
import time
from pathlib import Path

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import optax
from superiorflows import Flow
from superiorflows.train import (
    CheckpointCallback,
    EnergyBasedLoss,
    KullbackLeiblerLoss,
    LoggerCallback,
    MaximumLikelihoodLoss,
    ProgressBarCallback,
    Trainer,
)

# =============================================================================
# Models
# =============================================================================


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


# =============================================================================
# Data Loaders
# =============================================================================


def infinite_target_loader(distribution, batch_size, key):
    """Yields samples from target distribution (for MLE training)."""
    while True:
        key, subkey = jax.random.split(key)
        yield distribution.sample(seed=subkey, sample_shape=(batch_size,))


def infinite_base_loader(distribution, batch_size, key):
    """Yields samples from base distribution (for energy-based training)."""
    while True:
        key, subkey = jax.random.split(key)
        yield distribution.sample(seed=subkey, sample_shape=(batch_size,))


# =============================================================================
# Training Functions
# =============================================================================


def train_model(model, loss_fn, train_loader, val_loader, name, ckpt_path, nsteps=2000, seed=0):
    """Train a model and return loss history."""
    optimizer = optax.adam(1e-3)

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

    # Checkpoint callback
    ckpt_cb = CheckpointCallback(ckpt_path=ckpt_path, save_freq=500, overwrite=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_module=loss_fn,
        seed=seed,
        callbacks=[LoggerCallback(log_freq=500), ProgressBarCallback(refresh_rate=50), history_cb, ckpt_cb],
    )

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Checkpoints: {ckpt_path}")
    print(f"{'='*60}")

    t_start = time.time()
    trainer.train(train_loader=train_loader, val_loader=val_loader, max_steps=nsteps, val_freq=250)
    t_elapsed = time.time() - t_start

    print(f"Done in {t_elapsed:.1f}s ({1000*t_elapsed/nsteps:.0f}ms/step)")

    return trainer, history_cb.losses, history_cb.val_losses


# =============================================================================
# Main
# =============================================================================


def main():
    matplotlib.use("Agg")
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    # Random seed
    key = jax.random.key(42)

    # =================================================================
    # 1. Setup distributions
    # =================================================================
    d = 2

    # Target: 8 Gaussians in a circle
    angles = jnp.arange(8) * (jnp.pi / 4)
    locs = 10.0 * jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=1)
    target_dist = dsx.MixtureSameFamily(
        mixture_distribution=dsx.Categorical(probs=jnp.ones(8) / 8),
        components_distribution=dsx.MultivariateNormalDiag(loc=locs, scale_diag=jnp.full((8, 2), 0.7)),
    )

    # Base: Standard Gaussian
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))

    # =================================================================
    # 2. Setup loss functions
    # =================================================================
    flow_kwargs = dict(stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5))

    loss_mle = MaximumLikelihoodLoss(base_distribution=base_dist, **flow_kwargs)
    loss_energy = EnergyBasedLoss(base_distribution=base_dist, target_distribution=target_dist, **flow_kwargs)
    loss_hybrid = KullbackLeiblerLoss(
        base_distribution=base_dist,
        target_distribution=target_dist,
        alpha=0.5,
        **flow_kwargs,
    )

    # =================================================================
    # 3. Train three models
    # =================================================================
    nsteps = 5000
    batch_size = 32

    # Same architecture for all
    width = 16
    depth = 3
    key, k1, k2, k3 = jax.random.split(key, 4)
    model_mle = MLPVelocity(input_dim=d, width=width, depth=depth, key=k1)
    model_energy = MLPVelocity(input_dim=d, width=width, depth=depth, key=k2)
    model_hybrid = MLPVelocity(input_dim=d, width=width, depth=depth, key=k3)

    # Data loaders
    key, k1, k2, k3 = jax.random.split(key, 4)
    target_loader = infinite_target_loader(target_dist, batch_size, k1)
    base_loader = infinite_base_loader(base_dist, batch_size, k2)
    hybrid_loader = infinite_target_loader(target_dist, batch_size, k3)  # KL uses target batches

    # Train!
    # Validation loader (common for all, though energy-based might barely use it correctly if target unknown,
    # but here we know target)
    key, val_key = jax.random.split(key)
    val_data = target_dist.sample(seed=val_key, sample_shape=(1000,))
    val_loader = [val_data]

    # Train!
    # MLE
    trainer_mle, losses_mle, val_losses_mle = train_model(
        model_mle, loss_mle, target_loader, val_loader, "Maximum Likelihood", "tmp/ckpt_method_mle", nsteps
    )
    model_mle = trainer_mle.model

    # Energy
    trainer_energy, losses_energy, val_losses_energy = train_model(
        model_energy,
        loss_energy,
        base_loader,
        val_loader,
        "Energy-Based (Reverse KL)",
        "tmp/ckpt_method_energy",
        nsteps,
    )
    model_energy = trainer_energy.model

    # Hybrid
    trainer_hybrid, losses_hybrid, val_losses_hybrid = train_model(
        model_hybrid, loss_hybrid, hybrid_loader, val_loader, "Hybrid KL (alpha=0.5)", "tmp/ckpt_method_hybrid", nsteps
    )
    model_hybrid = trainer_hybrid.model

    # =================================================================
    # Reload from checkpoints to verify and for plotting
    # =================================================================
    print("\nReloading best models from checkpoints...")
    trainer_mle.load_checkpoint("tmp/ckpt_method_mle")
    model_mle = trainer_mle.model

    trainer_energy.load_checkpoint("tmp/ckpt_method_energy")
    model_energy = trainer_energy.model

    trainer_hybrid.load_checkpoint("tmp/ckpt_method_hybrid")
    model_hybrid = trainer_hybrid.model

    # =================================================================
    # 4. Visualize training curves
    # =================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Smooth losses for visualization
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

    # =================================================================
    # 5. Visualize final samples
    # =================================================================
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

    # =================================================================
    # 6. Generate Side-by-Side Animation
    # =================================================================
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

    # Common background density
    # (already computed in step 5: xx, yy, density)

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

        # Optionally update title with time?
        # fig.suptitle(f"t = {save_times[frame]:.2f}", fontsize=14)
        return scatters

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=80, blit=True)
    ani.save(output_dir / "training_comparison_movie.gif", writer="pillow", fps=15)
    print(f"Saved: {output_dir / 'training_comparison_movie.gif'}")

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
