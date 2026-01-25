"""
8 Gaussians: Train a CNF to morph a single Gaussian to 8 Gaussians on a circle.

Simple, minimal, lightning-fast training script with trajectory visualization.
"""

import time

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

# =============================================================================
# Setup
# =============================================================================

key = jax.random.PRNGKey(42)

# Base distribution: single Gaussian
base_dist = dsx.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2))

# Target distribution: 8 Gaussians on a circle
angles = jnp.arange(8) * (jnp.pi / 4)
locs = 10.0 * jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=1)
target_dist = dsx.MixtureSameFamily(
    mixture_distribution=dsx.Categorical(probs=jnp.ones(8) / 8),
    components_distribution=dsx.MultivariateNormalDiag(loc=locs, scale_diag=jnp.full((8, 2), 0.7)),
)


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

    def __call__(self, t, x, args):
        t_feat = jnp.broadcast_to(t, x.shape[:-1] + (1,))
        return self.mlp(jnp.concatenate([x, t_feat], axis=-1))


key, subkey = jax.random.split(key)
velocity_field = MLPVelocity(input_dim=2, width=16, depth=3, key=subkey)


# =============================================================================
# Training
# =============================================================================

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(velocity_field, eqx.is_array))


def make_step(base_dist, target_dist, optimizer, batch_size=32):
    """Create JIT-compiled train step. Closes over distributions (static)."""

    @eqx.filter_jit
    def train_step(vf, opt_state, key):
        key, subkey = jax.random.split(key)
        X = target_dist.sample(seed=subkey, sample_shape=(batch_size,))

        @eqx.filter_value_and_grad
        def loss_fn(vf):
            flow = Flow(velocity_field=vf, base_distribution=base_dist)
            return -jnp.mean(flow.log_prob(X))

        loss, grads = loss_fn(vf)
        updates, opt_state_new = optimizer.update(grads, opt_state, vf)
        vf_new = eqx.apply_updates(vf, updates)
        return vf_new, opt_state_new, loss, key

    return train_step


train_step = make_step(base_dist, target_dist, optimizer)

# Training loop
print("Training CNF: Gaussian -> 8 Gaussians")
print("=" * 40)

n_steps = 5000
key, subkey = jax.random.split(key)

t0 = time.time()
velocity_field, opt_state, loss, subkey = train_step(velocity_field, opt_state, subkey)
loss.block_until_ready()
print(f"Step 0 (compile): loss={loss:.4f}, time={time.time()-t0:.2f}s")

t_start = time.time()
for step in range(1, n_steps):
    velocity_field, opt_state, loss, subkey = train_step(velocity_field, opt_state, subkey)
    if step % 100 == 0:
        print(f"Step {step}: loss={loss:.4f}")

loss.block_until_ready()
t_train = time.time() - t_start
print(f"\nDone! {n_steps-1} steps in {t_train:.1f}s ({1000*t_train/(n_steps-1):.0f}ms/step)")


# =============================================================================
# Visualization: Animated trajectories
# =============================================================================

flow = Flow(velocity_field=velocity_field, base_distribution=base_dist)

print("\nGenerating trajectory animation...")
matplotlib.use("Agg")

# Sample initial points and integrate with saved trajectory
key, subkey = jax.random.split(key)
n_particles = 500
x0 = base_dist.sample(seed=subkey, sample_shape=(n_particles,))

# Integrate and save at multiple time points
n_frames = 50
save_times = jnp.linspace(0.0, 1.0, n_frames)
trajectories = jax.vmap(lambda x: flow.integrate(x, saveat=dfx.SaveAt(ts=save_times)).ys)(
    x0
)  # Shape: (n_particles, n_frames, 2)

# Compute target density on a grid for contour plot
grid_size = 100
x_grid = jnp.linspace(-15, 15, grid_size)
y_grid = jnp.linspace(-15, 15, grid_size)
xx, yy = jnp.meshgrid(x_grid, y_grid)
grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
log_probs = jax.vmap(target_dist.log_prob)(grid_points)
density = jnp.exp(log_probs).reshape(grid_size, grid_size)

# Create animation
fig, ax = plt.subplots(figsize=(8, 8))

# Contour of target density
ax.contourf(xx, yy, density, levels=20, cmap="Blues", alpha=0.6)
ax.contour(xx, yy, density, levels=10, colors="steelblue", linewidths=0.5, alpha=0.8)

# Initial scatter (will be updated)
scatter = ax.scatter([], [], s=8, c="orangered", alpha=0.7, zorder=5)

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect("equal")
ax.set_title("t = 0.00")
ax.set_xlabel("x")
ax.set_ylabel("y")


def animate(frame):
    positions = trajectories[:, frame, :]
    scatter.set_offsets(positions)
    ax.set_title(f"t = {save_times[frame]:.2f}")
    return (scatter,)


ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=80, blit=True)
ani.save("experiments/8_gaussians_flow.gif", writer="pillow", fps=12)
plt.close()

print("Saved: 8_gaussians_flow.gif")

# Also save final frame as static image
fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(xx, yy, density, levels=20, cmap="Blues", alpha=0.6)
ax.contour(xx, yy, density, levels=10, colors="steelblue", linewidths=0.5, alpha=0.8)
ax.scatter(trajectories[:, -1, 0], trajectories[:, -1, 1], s=8, c="orangered", alpha=0.7)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_aspect("equal")
ax.set_title("Learned Flow: Final samples (t=1)")
plt.tight_layout()
plt.savefig("experiments/8_gaussians_result.png", dpi=150)
print("Saved: 8_gaussians_result.png")
