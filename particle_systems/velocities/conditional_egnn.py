"""EGNN velocity field for patch-based particle system updates.

Processes a sub-system of N_patch + N_context particles through the EGNN
message-passing stack, returning velocities only for the N_patch patch
particles. Context particles participate in message passing (they influence
patch velocities) but their own positions are not returned.

Edge masking handles variable-count sub-systems: when the actual number of
valid particles (n_patch + n_context) is less than the static allocation
(N_patch + N_context), the padded (invalid) particles are excluded from all
message-passing interactions via boolean masks applied at each E_GCL layer.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from particle_systems.particle_system import log_map
from particle_systems.velocities.egnn import EGNN

__all__ = ["PatchEGNNVelocity"]


class PatchEGNNVelocity(eqx.Module):
    """EGNN velocity field for patch-based updates on periodic flat tori.

    Constructs an EGNN identical in architecture to
    :class:`ParticlesEGNNVelocity`, but operates on a sub-system of
    N_patch + N_context particles extracted by the selection protocol.
    Only the first N_patch positions (the patch) receive velocities;
    the remaining N_context (context) participate in message passing
    but are held fixed.

    Edge masking is applied at each E_GCL layer to ensure that padded
    (invalid) particles — arising from the fixed-size padding convention
    of the selection protocol — do not corrupt message passing for valid
    particles.

    Calling convention (via the bijector's args packing)::

        v = velocity(t, x_dyn, args)

    where ``x_dyn.positions`` has shape ``(N_patch, d)`` and
    ``args = (ctx, (index_meta, (n_patch, n_context)))``.

    Attributes:
        egnn: The underlying EGNN stack (shared architecture with
            :class:`ParticlesEGNNVelocity`).
        species_embedding: Frozen one-hot species lookup.
        N_patch: Static patch allocation size.
        N_context: Static context allocation size.
        d: Spatial dimension.
        n_species: Number of distinct particle species.
    """

    egnn: EGNN
    species_embedding: eqx.nn.Embedding
    N_patch: int = eqx.field(static=True)
    N_context: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)

    def __init__(
        self,
        N_patch: int,
        N_context: int,
        d: int,
        n_species: int,
        hidden_nf: int = 64,
        n_layers: int = 4,
        recurrent: bool = True,
        attention: bool = False,
        tanh: bool = False,
        coords_range: float = 15.0,
        *,
        key,
    ):
        self.N_patch = N_patch
        self.N_context = N_context
        self.d = d
        self.n_species = n_species

        in_node_nf = 1 + n_species
        k_egnn, k_spec = jax.random.split(key)

        self.egnn = EGNN(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            activation=jax.nn.silu,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            coords_range=coords_range,
            key=k_egnn,
        )

        self.species_embedding = eqx.nn.Embedding(
            num_embeddings=n_species,
            embedding_size=n_species,
            key=k_spec,
        )
        object.__setattr__(
            self,
            "species_embedding",
            eqx.tree_at(lambda m: m.weight, self.species_embedding, jnp.eye(n_species)),
        )

    def __call__(self, t, x_dyn, args):
        """Compute patch velocities for a sub-system configuration.

        Args:
            t: Time scalar.
            x_dyn: Dynamic partition — ``x_dyn.positions`` has shape
                ``(N_patch, d)``, other fields are ``None``.
            args: Packed by the bijector as
                ``(ctx, (index_meta, (n_patch, n_context)))``.
                ``ctx.positions`` has shape ``(N_context, d)``,
                ``ctx.species`` has shape ``(M,)`` where ``M = N_patch + N_context``,
                ``ctx.box`` has shape ``(d,)``.

        Returns:
            ParticleSystem with ``positions`` containing the velocity
            for each patch particle, shape ``(N_patch, d)``, and
            ``species=None``, ``box=None``.
        """
        ctx, extra = args
        if isinstance(extra, tuple) and len(extra) == 2 and isinstance(extra[1], tuple):
            _, user_args = extra
        else:
            user_args = extra
        n_patch, n_context = user_args

        patch_pos = x_dyn.positions
        context_pos = ctx.positions
        all_species = ctx.species
        box = ctx.box
        L = box[0]

        M = self.N_patch + self.N_context
        all_pos = jnp.concatenate([patch_pos, context_pos])

        t_feat = jnp.broadcast_to(jnp.reshape(t, (1, 1)), (M, 1))
        species_feat = jax.vmap(self.species_embedding)(all_species)
        h0 = jnp.concatenate([t_feat, species_feat], axis=-1)

        n_valid = n_patch + n_context
        node_mask = jnp.arange(M) < n_valid
        edge_mask = node_mask[:, None] & node_mask[None, :]
        patch_mask = jnp.arange(M) < n_patch

        _, x_final = self._masked_forward(h0, all_pos, L, edge_mask, node_mask, patch_mask)

        vel = log_map(all_pos[: self.N_patch], x_final[: self.N_patch], L)

        return type(x_dyn)(positions=vel, species=None, box=None)

    def _masked_forward(self, h0, pos, L, edge_mask, node_mask, patch_mask):
        """EGNN forward pass with edge masking for variable-count sub-systems.

        Reuses the EGNN's embedding and E_GCL sub-methods, applying edge
        and node masking between each layer to suppress contributions from
        padded (invalid) particles. Specifically:

        - Edge features are zeroed for any pair involving a padded particle,
          preventing spurious messages from corrupting valid node updates.
        - Coordinate displacements are zeroed for padded pairs, preventing
          position drift from bias-derived coordinate weights.
        - Node features are zeroed for padded particles after each layer,
          ensuring they remain inert throughout the message-passing stack.
        - Context particles have their positions frozen at each layer, acting
          as fixed boundary anchors for the message passing.

        Args:
            h0: Initial node features, shape (M, in_node_nf).
            pos: Particle positions, shape (M, d).
            L: Box side length (scalar, traced).
            edge_mask: Boolean pairwise validity mask, shape (M, M).
            node_mask: Boolean per-node validity mask, shape (M,).
            patch_mask: Boolean mask for patch particles, shape (M,).

        Returns:
            ``(h, pos)``: updated node features and coordinates.
        """
        h = jax.vmap(self.egnn.embedding)(h0)
        h = h * node_mask[:, None]

        for layer in self.egnn.layers:
            radial, coord_diff = layer._coord2radial(pos, L)
            coord_diff = coord_diff * edge_mask[:, :, None]

            edge_feat = layer._edge_model(h, radial)
            edge_feat = edge_feat * edge_mask[:, :, None]

            pos_new = layer._coord_model(pos, coord_diff, edge_feat, L)
            pos = jnp.where(patch_mask[:, None], pos_new, pos)

            h = layer._node_model(h, edge_feat)
            h = h * node_mask[:, None]

        return h, pos
