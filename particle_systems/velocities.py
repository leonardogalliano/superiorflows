import equinox as eqx
import jax
import jax.numpy as jnp

from particle_systems.particle_system import ParticleSystem

__all__ = ["ParticlesMLPVelocity"]


class ParticlesMLPVelocity(eqx.Module):
    """MLP velocity field for PBC particle systems with sinusoidal embedding.

    Encodes particle positions with sinusoidal PBC-aware features,
    concatenates species labels and time, and passes through a tanh MLP.

    The velocity field assumes **unbatched** inputs — `Flow` handles batching
    via `vmap`. The dynamic part `x` carries only positions (shape `(N, d)`),
    while context `ctx` carries species and box (static under the flow).

    Args:
        N: Number of particles.
        d: Spatial dimensionality.
        width: Hidden layer width of the MLP.
        depth: Number of hidden layers.
        key: JAX PRNG key for weight initialization.
    """

    mlp: eqx.nn.MLP
    N: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, N: int, d: int, width: int, depth: int, *, key):
        self.N = N
        self.d = d
        self.mlp = eqx.nn.MLP(
            in_size=2 * d * N + N + 1,  # sin/cos embed + species + time
            out_size=N * d,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, x: ParticleSystem, ctx: ParticleSystem) -> ParticleSystem:
        L = ctx.box[0]
        k = 2.0 * jnp.pi / L

        pos_embed = jnp.concatenate([jnp.sin(k * x.positions), jnp.cos(k * x.positions)], axis=-1)  # (N, 2d)

        features = jnp.concatenate(
            [
                pos_embed.ravel(),
                ctx.species.astype(jnp.float32),
                jnp.array([t]),
            ]
        )

        velocity = self.mlp(features).reshape(self.N, self.d)
        return ParticleSystem(positions=velocity, species=None, box=None)
