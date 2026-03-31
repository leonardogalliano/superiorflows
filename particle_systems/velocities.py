import equinox as eqx
import jax
import jax.numpy as jnp

__all__ = ["ParticlesMLPVelocity"]


class ParticlesMLPVelocity(eqx.Module):
    """MLP velocity field for PBC particle systems with sinusoidal embedding.

    Encodes particle positions with sinusoidal PBC-aware features,
    concatenates one-hot species labels and time, and passes through a tanh MLP.
    Supports general orthorhombic simulation boxes.
    """

    mlp: eqx.nn.MLP
    N: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)

    def __init__(self, N: int, d: int, n_species: int, width: int, depth: int, *, key):
        self.N = N
        self.d = d
        self.n_species = n_species

        # Calculate the total input dimension
        # Positions: 2 * d * N (sin and cos for each coordinate)
        # Species: N * n_species (one-hot vector per particle)
        # Time: 1
        in_features = 2 * d * N + N * n_species + 1

        self.mlp = eqx.nn.MLP(
            in_size=in_features,
            out_size=N * d,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, x, ctx):
        # Allow ctx.box to be a vector of shape (d,) for general boxes
        k = 2.0 * jnp.pi / ctx.box  # Shape (d,)

        # Broadcasting: (N, d) * (d,) results in (N, d)
        pos_embed = jnp.concatenate([jnp.sin(k * x.positions), jnp.cos(k * x.positions)], axis=-1)  # Shape (N, 2d)

        # One-hot encoding for categorical species labels
        # Output shape: (N, n_species)
        species_embed = jax.nn.one_hot(ctx.species, self.n_species)

        features = jnp.concatenate(
            [
                pos_embed.ravel(),
                species_embed.ravel(),
                jnp.array([t]),
            ]
        )

        velocity = self.mlp(features).reshape(self.N, self.d)

        # Construct and return the new state, maintaining the class type of x
        return type(x)(positions=velocity, species=None, box=None)
