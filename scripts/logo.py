from pathlib import Path

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import optax
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from superiorflows import CoupledDataSource, DistributionDataSource, Flow
from superiorflows.train import (
    CheckpointCallback,
    LoggerCallback,
    ProgressBarCallback,
    StochasticInterpolantLoss,
    Trainer,
)


def generate_smooth_grains(key, text, n_samples=20000, grid_size=256, sigma=0.05):
    img = Image.new("L", (grid_size, grid_size), color=0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(((grid_size - w) / 2, (grid_size - h) / 2), text, fill=255, font=font)

    pdf = jnp.array(img, dtype=jnp.float32)
    pdf /= pdf.sum()

    k1, k2, k3 = jax.random.split(key, 3)

    flat_indices = jax.random.choice(k1, jnp.arange(grid_size**2), shape=(n_samples,), p=pdf.ravel())
    y = (flat_indices // grid_size).astype(jnp.float32)
    x = (flat_indices % grid_size).astype(jnp.float32)

    x_noise = jax.random.normal(k2, shape=(n_samples,))
    y_noise = jax.random.normal(k3, shape=(n_samples,))

    x = x + sigma * x_noise
    y = y + sigma * y_noise

    coords = jnp.stack([x, y], axis=1)
    coords = (coords / grid_size) * 2 - 1
    coords = coords.at[:, 1].set(-coords[:, 1])

    return coords


# Generate a very dense set of grains
key = jax.random.key(0)
# Generate a much sharper logo with higher resolution
logo_samples = generate_smooth_grains(key, "Superior Flows", n_samples=10_000_000, sigma=0.05)
logo_samples = 14 * (logo_samples - logo_samples.mean(axis=0))


class RandomFourierFeatures(eqx.Module):
    """Random Fourier Features (RFF) for coordinate embedding."""

    weight: jax.Array

    def __init__(self, in_size: int, out_size: int, scale: float, *, key):
        # out_size should be even for sin/cos pairs
        self.weight = jax.random.normal(key, (out_size // 2, in_size)) * scale

    def __call__(self, x):
        h = self.weight @ x
        return jnp.concatenate([jnp.sin(h), jnp.cos(h)])


class ResidualBlock(eqx.Module):
    """Residual block with SiLU activation."""

    linear: eqx.nn.Linear

    def __init__(self, dim: int, *, key):
        self.linear = eqx.nn.Linear(dim, dim, key=key)

    def __call__(self, x):
        return x + jax.nn.silu(self.linear(x))


class MLPVelocity(eqx.Module):
    """ResNet with RFF and Additive Conditioning."""

    time_rff: RandomFourierFeatures
    time_embed: eqx.nn.Linear
    x_rff: RandomFourierFeatures
    x_embed: eqx.nn.Linear
    blocks: list[ResidualBlock]
    last: eqx.nn.Linear

    def __init__(self, input_dim: int, width: int, depth: int, *, key):
        k1, k2, k3, k4, *k_blocks = jax.random.split(key, depth + 5)

        # Time embedding
        self.time_rff = RandomFourierFeatures(1, width, scale=1.0, key=k1)
        self.time_embed = eqx.nn.Linear(width, width, key=k2)

        # Space embedding
        self.x_rff = RandomFourierFeatures(input_dim, width, scale=3.0, key=k3)
        self.x_embed = eqx.nn.Linear(width, width, key=k4)

        self.blocks = [ResidualBlock(width, key=kb) for kb in k_blocks[:-1]]
        self.last = eqx.nn.Linear(width, input_dim, key=k_blocks[-1])

    @eqx.filter_jit
    def __call__(self, t, x, args):
        # Additive embedding: forced interaction in latent space
        t_feat = self.time_embed(self.time_rff(jnp.array([t])))
        x_feat = self.embed_x(self.x_rff(x)) if hasattr(self, "embed_x") else self.x_embed(self.x_rff(x))

        h = t_feat + x_feat
        for block in self.blocks:
            h = block(h)
        return self.last(h)


seed = 42
key = jax.random.key(seed)
d = 2
base_dist = dsx.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))


def interpolant(t, x0, x1):
    return (1 - t) * x0 + t * x1


def gamma_fn(t):
    """Smoother noise schedule to avoid derivative blow-up at t=0, 1."""
    return 0.5 * t * (1 - t)


loss_fn = StochasticInterpolantLoss(interpolant=interpolant, gamma=gamma_fn)
width = 384
depth = 4
key, subkey = jax.random.split(key)
model = MLPVelocity(input_dim=d, width=width, depth=depth, key=subkey)

batch_size = 512


class EmpiricalLogoDistribution:
    def __init__(self, samples):
        self.samples = samples

    @property
    def event_shape(self):
        return (self.samples.shape[-1],)

    def sample(self, seed, sample_shape):
        n = sample_shape[0]
        idx = jax.random.randint(seed, (n,), 0, len(self.samples))
        return self.samples[idx]


target_dist = EmpiricalLogoDistribution(logo_samples)

train_source = CoupledDataSource(
    DistributionDataSource(base_dist, batch_size, seed=seed),
    DistributionDataSource(target_dist, batch_size, seed=seed + 1),
)


class MinibatchOTCoupling:
    """Couples x0 and x1 via Optimal Transport (OT) within each minibatch."""

    def __call__(self, batch):
        x0, x1 = batch
        delta = x0[:, None, :] - x1[None, :, :]
        cost_matrix = np.sum(delta**2, axis=-1)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return x0[row_ind], x1[col_ind]


train_dataset = grain.MapDataset.source(train_source).repeat().map(MinibatchOTCoupling())

max_steps = 10000
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-3,
    peak_value=1e-3,
    warmup_steps=100,
    decay_steps=max_steps,
    end_value=1e-5,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(lr_schedule, weight_decay=1e-3),
)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_module=loss_fn,
    seed=seed,
    callbacks=[
        LoggerCallback(log_freq=50),
        ProgressBarCallback(refresh_rate=10),
        CheckpointCallback(ckpt_path="tmp/ckpt_logo", save_freq=1000, overwrite=True),
    ],
)

trainer.train(dataset=train_dataset, max_steps=max_steps)


# =================================================================
# Animation
# =================================================================


trainer.load_checkpoint("tmp/ckpt_logo")
model = trainer.model
n_particles = 20000
n_frames = 40
save_times = jnp.linspace(0.0, 1.0, n_frames)

# For animation, we transport from Gaussian to Full Logo
key, subkey = jax.random.split(key)
x0 = base_dist.sample(seed=subkey, sample_shape=(n_particles,))


def get_trajectories(model):
    flow = Flow(velocity_field=model, base_distribution=base_dist)
    return jax.vmap(lambda x: flow.integrate(x, saveat=dfx.SaveAt(ts=save_times)).ys)(x0)


trajectory = get_trajectories(model)

# Color particles based on final x-position using a brighter range of Plasma
final_x = trajectory[:, -1, 0]
norm_x = (final_x - final_x.min()) / (final_x.max() - final_x.min() + 1e-6)
# Use the range [0.2, 1.0] of plasma to avoid the dark purple/black parts
colors = plt.cm.plasma(0.2 + 0.8 * norm_x)

fig, ax = plt.subplots(1, 1, figsize=(12, 6), facecolor="black")
ax.set_facecolor("black")
# Use smaller points for a sharper, high-res feel
scatter = ax.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], s=1.0, c=colors, alpha=0.9, edgecolors="none")
ax.set_xlim(-12, 12)
ax.set_ylim(-6, 6)
ax.axis("off")
fig.tight_layout()


def init():
    scatter.set_offsets(trajectory[:, 0, :])
    return (scatter,)


def animate(frame):
    scatter.set_offsets(trajectory[:, frame, :])
    return (scatter,)


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=80, blit=True)

output_dir = Path("experiments")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "logo.gif"
ani.save(str(output_path), writer="pillow", fps=15)
