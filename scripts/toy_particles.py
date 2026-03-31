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
from superiorflows import DistributionDataSource
from superiorflows.train import (
    CheckpointCallback,
    LoggerCallback,
    MaximumLikelihoodLoss,
    ProgressBarCallback,
    Trainer,
)
from typing_extensions import Annotated


class ToyParticles(eqx.Module):
    positions: jnp.ndarray
    species: jnp.ndarray
    box: jnp.ndarray

    @property
    def d(self):
        return self.positions.shape[-1]

    @property
    def N(self):
        return self.positions.shape[-2]

    @classmethod
    def get_dynamic_mask(cls):
        return cls(positions=True, species=False, box=False)


class ToyParticlesDistribution(eqx.Module, dsx.Distribution):
    L: float = eqx.field(static=True)
    alphas: jnp.ndarray
    betas: jnp.ndarray
    n_species: int = eqx.field(static=True)
    n_particles: int = eqx.field(static=True)

    def __init__(self, n_species, L, alphas=None, betas=None):
        self.n_species = n_species
        self.n_particles = 2 * n_species
        self.L = float(L)
        self.alphas = 2.0 * jnp.ones(n_species) if alphas is None else jnp.asarray(alphas)
        self.betas = 2.0 * jnp.ones(n_species) if betas is None else jnp.asarray(betas)

        if self.alphas.shape[0] != n_species:
            raise ValueError(f"Alphas length mismatch: {self.alphas.shape[0]} vs {n_species}")

    @property
    def event_shape(self):
        # Describes the shape of a SINGLE sample (unbatched)
        return ToyParticles(positions=(self.n_particles, 2), species=(self.n_particles,), box=(2,))

    @property
    def _prior_dist(self):
        return dsx.Uniform(low=jnp.zeros(2), high=jnp.full(2, self.L))

    @property
    def _angle_dist(self):
        return dsx.Uniform(low=jnp.zeros(self.n_species), high=2 * jnp.pi * jnp.ones(self.n_species))

    @property
    def _norm_dist(self):
        return dsx.Transformed(
            distribution=dsx.Beta(alpha=self.alphas, beta=self.betas),
            bijector=dsx.ScalarAffine(shift=jnp.zeros(self.n_species), scale=(self.L / 2.0)),
        )

    def _sample_n(self, key, n):
        k1, k2, k3 = jax.random.split(key, 3)

        # 1. Sample Centers (Species 0, 1, 2...)
        # Shape: (n, n_species, 2)
        x_centers = self._prior_dist.sample(seed=k1, sample_shape=(n, self.n_species))

        # 2. Sample Relative Polar Coords
        # Shape: (n, n_species)
        angles = self._angle_dist.sample(seed=k2, sample_shape=(n,))
        radii = self._norm_dist.sample(seed=k3, sample_shape=(n,))

        # 3. Compute Satellites
        dx = radii * jnp.cos(angles)
        dy = radii * jnp.sin(angles)
        delta = jnp.stack([dx, dy], axis=-1)

        # Apply PBC: (Center + Delta) % L
        x_satellites = jnp.remainder(x_centers + delta, self.L)

        # 4. Interleave to Sort: [Center0, Sat0, Center1, Sat1...]
        # Stack: (n, n_species, 2, 2) -> (n, 2*n_species, 2)
        X_pairs = jnp.stack([x_centers, x_satellites], axis=2)
        positions = X_pairs.reshape(n, self.n_particles, 2)

        # 5. Create Species Labels
        # [0, 0, 1, 1, 2, 2...]
        s_single = jnp.repeat(jnp.arange(self.n_species), 2)
        species = jnp.broadcast_to(s_single, (n, self.n_particles))

        # 6. Box
        batched_box = jnp.full((n, 2), self.L)

        return ToyParticles(positions=positions, species=species, box=batched_box)

    def log_prob(self, value: ToyParticles):
        # 1. Parse Inputs (Handle potential batching implicitly via JAX logic)
        X = value.positions  # (..., N, 2)
        a = value.species  # (..., N)

        # 2. Sort X based on species `a` to ensure [C0, S0, C1, S1] ordering
        idx_sort = jnp.argsort(a, axis=-1)
        X_sorted = jnp.take_along_axis(X, idx_sort[..., None], axis=-2)

        # 3. Reshape into Pairs: (..., n_species, 2_particles, 2_coords)
        # axis -2 is the pair index: 0=Center, 1=Satellite
        batch_shape = X.shape[:-2]
        X_pairs = X_sorted.reshape(*batch_shape, self.n_species, 2, 2)

        x_centers = X_pairs[..., 0, :]  # (..., n_species, 2)
        x_satellites = X_pairs[..., 1, :]  # (..., n_species, 2)

        # 4. Compute Log Probs

        # A. Prior (Centers)
        lp_prior = self._prior_dist.log_prob(x_centers).sum(axis=(-1, -2))

        # B. Geometric Relations (PBC difference)
        diff = x_satellites - x_centers
        diff = diff - jnp.round(diff / self.L) * self.L

        norm = jnp.linalg.norm(diff, axis=-1)  # (..., n_species)
        angle = jnp.arctan2(diff[..., 1], diff[..., 0])  # (..., n_species)
        angle = jnp.remainder(angle, 2 * jnp.pi)

        # C. Conditionals
        # Radius probability
        lp_norm = self._norm_dist.log_prob(norm).sum(axis=-1)

        # Angle probability
        lp_angle = self._angle_dist.log_prob(angle).sum(axis=-1)

        # Jacobian adjustment: -log(r)
        lp_jacobian = -jnp.sum(jnp.log(norm), axis=-1)

        # 5. Boundary Checks (Hard constraints)
        valid_norm = jnp.all(norm < (self.L / 2.0), axis=-1)

        total_lp = lp_prior + lp_norm + lp_angle + lp_jacobian

        return jnp.where(valid_norm, total_lp, -jnp.inf)


class UniformToyParticles(eqx.Module, dsx.Distribution):
    L: float = eqx.field(static=True)
    ref_species: jnp.ndarray  # Shape: (N,)

    def __init__(self, L, ref_species):
        self.L = float(L)
        self.ref_species = jnp.asarray(ref_species)

    @property
    def n_particles(self):
        return self.ref_species.shape[0]

    @property
    def event_shape(self):
        return ToyParticles(
            positions=(self.n_particles, 2),
            species=(self.n_particles,),
            box=(2,),  # scalar broadcasted to 2 dims
        )

    def _sample_n(self, key, n):
        k_pos, k_spec = jax.random.split(key)

        # 1. Sample Positions: Uniform(0, L)
        # Shape: (n, N, 2)
        pos = jax.random.uniform(k_pos, shape=(n, self.n_particles, 2), minval=0.0, maxval=self.L)

        # 2. Sample Species: Random permutation of ref_species
        def _permute(k):
            return jax.random.permutation(k, self.ref_species)

        keys_perm = jax.random.split(k_spec, n)
        species = jax.vmap(_permute)(keys_perm)

        # 3. Box: Broadcast scalar L
        batched_box = jnp.full((n, 2), self.L)

        return ToyParticles(positions=pos, species=species, box=batched_box)

    def log_prob(self, value: ToyParticles):
        # 1. Base Log Prob: -N * log(Volume)
        # Volume = L^2
        # log(Volume) = 2 * log(L)
        base_log_prob = -self.n_particles * (2.0 * jnp.log(self.L))

        # Remove this check, otherwise it requires a projection
        # # 2. Check Constraints
        # # A. Positions inside box [0, L]
        # in_box = jnp.all(
        #     (value.positions >= 0.0) & (value.positions <= self.L),
        #     axis=(-1, -2)
        # )
        in_box = True

        # B. Correct Species Composition
        sorted_val_species = jnp.sort(value.species, axis=-1)
        sorted_ref_species = jnp.sort(self.ref_species, axis=-1)

        valid_composition = jnp.all(sorted_val_species == sorted_ref_species, axis=-1)

        # 3. Return
        is_valid = in_box & valid_composition
        return jnp.where(is_valid, base_log_prob, -jnp.inf)


class ParticlesMLPVelocity(eqx.Module):
    mlp: eqx.nn.MLP
    N: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, N: int, d: int, width: int, depth: int, *, key):
        self.N = N
        self.d = d
        self.mlp = eqx.nn.MLP(
            in_size=(2 * d) * N + N + 1,  # positions (sin/cos) + species + time
            out_size=N * d,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    @eqx.filter_jit
    def __call__(self, t, x, ctx):
        # Embed positions: [sin(2pi*x/L), cos(2pi*x/L)]
        L = ctx.box[0]
        k = 2 * jnp.pi / L
        pos_embed = jnp.concatenate([jnp.sin(k * x.positions), jnp.cos(k * x.positions)], axis=-1)

        flatten_embedding = pos_embed.ravel()
        species_embedding = ctx.species.astype(jnp.float32)
        t_feat = jnp.array([t])
        velocity_flat = self.mlp(jnp.concatenate([flatten_embedding, species_embedding, t_feat]))
        velocity = velocity_flat.reshape((self.N, self.d))

        return ToyParticles(positions=velocity, species=None, box=None)


app = typer.Typer(pretty_exceptions_show_locals=False)


def train_model(
    n_species: int,
    L: float,
    width: int,
    depth: int,
    lr: float,
    nsteps: int,
    batch_size: int,
    seed: int,
    log_freq: int,
    ckpt_path: Path,
    overwrite: bool,
):
    key = jax.random.key(seed)

    # Distributions
    alphas = jnp.ones(n_species)
    betas = jnp.ones(n_species)
    target_dist = ToyParticlesDistribution(n_species, L, alphas=alphas, betas=betas)
    ref_species = jnp.arange(n_species).repeat(2)
    uniform_dist = UniformToyParticles(L=L, ref_species=ref_species)

    # Data Source
    dataset = grain.MapDataset.source(DistributionDataSource(target_dist, batch_size)).repeat()

    # Loss & Flow
    flow_kwargs = dict(
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5), dynamic_mask=ToyParticles.get_dynamic_mask()
    )
    loss_fn = MaximumLikelihoodLoss(base_distribution=uniform_dist, **flow_kwargs)

    # Model
    key, subkey = jax.random.split(key)
    velocity_field = ParticlesMLPVelocity(N=2 * n_species, d=2, width=width, depth=depth, key=subkey)

    # Optimizer
    optimizer = optax.adam(lr)

    # Callbacks
    callbacks = [
        LoggerCallback(log_freq=log_freq),
        ProgressBarCallback(refresh_rate=50),
        CheckpointCallback(ckpt_path=ckpt_path, save_freq=500, overwrite=overwrite),
    ]

    trainer = Trainer(
        model=velocity_field,
        optimizer=optimizer,
        loss_module=loss_fn,
        seed=seed,
        callbacks=callbacks,
    )

    print(f"\n{'='*60}")
    print("Training Toy Particles")
    print(f"Species: {n_species}, L: {L}")
    print(f"Checkpoints: {ckpt_path}")
    print(f"{'='*60}")

    t_start = time.time()
    # No validation loader for now to keep it simple as in original script,
    # but could be added easily.
    trainer.train(dataset=dataset, max_steps=nsteps)
    t_elapsed = time.time() - t_start

    print(f"Done in {t_elapsed:.1f}s ({1000*t_elapsed/nsteps:.0f}ms/step)")
    return trainer


@app.command()
def main(
    n_species: Annotated[int, typer.Option(help="Number of species")] = 3,
    L: Annotated[float, typer.Option(help="Box size")] = 2.0,
    width: Annotated[int, typer.Option(help="Width of the MLP")] = 16,
    depth: Annotated[int, typer.Option(help="Depth of the MLP")] = 2,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    nsteps: Annotated[int, typer.Option(help="Number of training steps")] = 5000,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 64,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    log_freq: Annotated[int, typer.Option(help="Logging frequency")] = 500,
    ckpt_path: Annotated[Path, typer.Option(help="Checkpoint path")] = Path("tmp/ckpt_toy_particles"),
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing checkpoints")] = True,
):
    train_model(
        n_species=n_species,
        L=L,
        width=width,
        depth=depth,
        lr=lr,
        nsteps=nsteps,
        batch_size=batch_size,
        seed=seed,
        log_freq=log_freq,
        ckpt_path=ckpt_path,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
