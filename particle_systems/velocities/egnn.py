"""Equivariant Graph Neural Network (EGNN) velocity field for PBC particle systems.

Implements the EGNN architecture from:
    Satorras et al. (2021) "E(n) Equivariant Graph Neural Networks"
    https://arxiv.org/abs/2102.09844

Adapted for periodic boundary conditions (flat torus) and JAX/Equinox,
following the PyTorch reference implementation in learndiffeq
(https://github.com/h2o64/learndiffeq).

Key design differences from the PyTorch version:
  - No batching inside the architecture. A single ParticleSystem (N, d) is
    processed; batching across systems is handled externally by jax.vmap.
  - Fully-connected graph is represented implicitly via (N, 1, d) / (1, N, d)
    broadcasting — no sparse (row, col) edge index arrays. This is both simpler
    and faster on JAX-accelerated hardware for small N.
  - L (box side length) is NOT stored in the layers; it is passed as a traced
    argument through the call chain so the same compiled model can run for
    any box size and so it participates correctly in JAX autodiff.
  - The unused final node-feature embedding layer (embedding_out) present in
    the original EGNN.forward is omitted, since EGNN_dynamics only uses the
    updated coordinates.
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from particle_systems.particle_system import log_map

__all__ = ["ParticlesEGNNVelocity"]


# ── Periodic geometry helpers ─────────────────────────────────────────────────
# log_map(x, y, L) is imported from particle_system — single source of truth.


# ── Building-block MLP helpers ────────────────────────────────────────────────
#
# The PyTorch EGNN uses two patterns of nn.Sequential:
#
#   edge_mlp:  Linear → act → Linear → act    (activation at output)
#   node_mlp:  Linear → act → Linear           (NO activation at output)
#   coord_mlp: Linear → act → Linear(1)  [+ optional Tanh]
#
# We build a simple 2-layer MLP that supports an optional final activation.


class _MLP(eqx.Module):
    """Generic 2-layer MLP: Linear(in→h) → act → Linear(h→out) [→ final_act]."""

    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)
    final_activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden: int,
        out_features: int,
        activation: Callable,
        final_activation: Callable | None = None,
        *,
        key,
    ):
        k1, k2 = jax.random.split(key)
        self.w1 = eqx.nn.Linear(in_features, hidden, key=k1)
        self.w2 = eqx.nn.Linear(hidden, out_features, key=k2)
        self.activation = activation
        self.final_activation = final_activation if final_activation is not None else (lambda x: x)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.final_activation(self.w2(self.activation(self.w1(x))))


# ── E_GCL: single equivariant graph convolutional layer ──────────────────────


class E_GCL(eqx.Module):
    """One layer of the Equivariant Graph Convolutional Network (EGNN).

    Processes a single particle configuration (no batch dimension).
    L (box side length) is passed as a runtime argument to __call__ so that
    it participates in JAX tracing (autodiff, jit) correctly.

    Args:
        hidden_nf:    Width of all hidden representations.
        activation:   Activation function.
        recurrent:    Whether to use a residual connection in the node update.
        attention:    Whether to gate edge features with a learned scalar.
        tanh:         Whether to apply tanh + coords_range scaling to the
                      coordinate update.
        coords_range: Coordinate update scale when tanh=True.
    """

    edge_mlp: _MLP
    node_mlp: _MLP
    coord_w: eqx.nn.Linear  # final coord update linear (no bias, small init)
    coord_hidden: eqx.nn.Linear  # coord hidden layer
    att_mlp: eqx.nn.Linear | None  # attention gate (None if attention=False)
    recurrent: bool = eqx.field(static=True)
    attention: bool = eqx.field(static=True)
    tanh: bool = eqx.field(static=True)
    coords_range: float = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        hidden_nf: int,
        activation: Callable,
        recurrent: bool,
        attention: bool,
        tanh: bool,
        coords_range: float,
        *,
        key,
    ):
        k_edge, k_node, k_ch, k_cw, k_att = jax.random.split(key, 5)

        # edge_mlp: [h_i || h_j || radial || edge_attr] → hidden_nf → hidden_nf
        # In the original code in_edge_nf=1 (dist_sq as edge_attr) and edge_coords_nf=1
        # (radial from coord2radial) → input dim = 2*hidden_nf + 2.
        self.edge_mlp = _MLP(
            in_features=2 * hidden_nf + 2,
            hidden=hidden_nf,
            out_features=hidden_nf,
            activation=activation,
            final_activation=activation,  # activation at output (matches PyTorch edge_mlp)
            key=k_edge,
        )

        # node_mlp: [h_i || agg_edge_feats] → hidden_nf → hidden_nf, no final act
        self.node_mlp = _MLP(
            in_features=2 * hidden_nf,
            hidden=hidden_nf,
            out_features=hidden_nf,
            activation=activation,
            final_activation=None,
            key=k_node,
        )

        # coord_mlp: coord_hidden (hidden_nf → hidden_nf) → coord_w (hidden_nf → 1)
        self.coord_hidden = eqx.nn.Linear(hidden_nf, hidden_nf, key=k_ch)
        self.coord_w = eqx.nn.Linear(hidden_nf, 1, use_bias=False, key=k_cw)

        # Apply Xavier uniform with gain=0.001, matching the PyTorch reference init:
        #   torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        # bound = gain * sqrt(6 / (fan_in + fan_out))
        bound = 0.001 * jnp.sqrt(6.0 / (hidden_nf + 1))
        new_w = jax.random.uniform(k_cw, shape=self.coord_w.weight.shape, minval=-bound, maxval=bound)
        object.__setattr__(self, "coord_w", eqx.tree_at(lambda m: m.weight, self.coord_w, new_w))

        # Attention gate
        self.att_mlp = eqx.nn.Linear(hidden_nf, 1, key=k_att) if attention else None

        self.recurrent = recurrent
        self.attention = attention
        self.tanh = tanh
        self.coords_range = coords_range
        self.activation = activation

    # ── Private helpers ───────────────────────────────────────────────────────

    def _coord2radial(self, x: jnp.ndarray, L):
        """Pairwise squared torus distances and normalised displacement vectors.

        Args:
            x: coordinates, shape (N, d)
            L: box side length (scalar, traced)

        Returns:
            radial:     shape (N, N, 1)  — squared torus distances
            coord_diff: shape (N, N, d)  — normalised displacement vectors
        """
        # displacement from j to i on the flat torus: shape (N, N, d)
        coord_diff = log_map(x[None, :, :], x[:, None, :], L)  # x_i - x_j

        radial = jnp.sum(coord_diff**2, axis=-1, keepdims=True)  # (N, N, 1)
        norm = jnp.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1.0)  # normalise
        return radial, coord_diff

    def _edge_model(self, h: jnp.ndarray, radial: jnp.ndarray):
        """Compute edge features for all N² pairs.

        Args:
            h:      node features,  shape (N, hidden_nf)
            radial: squared dists,  shape (N, N, 1)

        Returns:
            edge_feat: shape (N, N, hidden_nf)
        """
        N = h.shape[0]
        hi = jnp.broadcast_to(h[:, None, :], (N, N, h.shape[1]))  # (N, N, hid)
        hj = jnp.broadcast_to(h[None, :, :], (N, N, h.shape[1]))  # (N, N, hid)

        # In the original EGNN_dynamics, edge_attr = dist_sq (same as radial).
        # [h_i || h_j || radial || edge_attr] where edge_attr = radial.
        inp = jnp.concatenate([hi, hj, radial, radial], axis=-1)  # (N, N, 2*hid+2)

        inp_flat = inp.reshape(N * N, -1)
        out_flat = jax.vmap(self.edge_mlp)(inp_flat)  # (N*N, hidden_nf)
        edge_feat = out_flat.reshape(N, N, -1)  # (N, N, hidden_nf)

        if self.attention:
            att = jax.nn.sigmoid(jax.vmap(self.att_mlp)(out_flat))  # (N*N, 1)
            edge_feat = edge_feat * att.reshape(N, N, 1)

        return edge_feat

    def _coord_model(self, x: jnp.ndarray, coord_diff: jnp.ndarray, edge_feat: jnp.ndarray, L):
        """Update coordinates by aggregating weighted displacement vectors.

        Args:
            x:          coords,              shape (N, d)
            coord_diff: normalised displ.,   shape (N, N, d)
            edge_feat:  per-edge features,   shape (N, N, hidden_nf)
            L:          box side length (scalar, traced)

        Returns:
            Updated coordinates, shape (N, d), wrapped to [0, L)^d.
        """
        N = x.shape[0]
        edge_feat_flat = edge_feat.reshape(N * N, -1)

        coord_h = jax.vmap(self.coord_hidden)(edge_feat_flat)  # (N*N, hidden_nf)
        coord_h = self.activation(coord_h)
        w = jax.vmap(self.coord_w)(coord_h).reshape(N, N, 1)  # (N, N, 1)

        if self.tanh:
            w = jax.nn.tanh(w) * self.coords_range

        trans = coord_diff * w  # (N, N, d)
        delta = jnp.sum(trans, axis=1)  # sum over j → (N, d)

        return jnp.remainder(x + delta, L)

    def _node_model(self, h: jnp.ndarray, edge_feat: jnp.ndarray):
        """Update node features by aggregating edge features.

        Args:
            h:         node features,  shape (N, hidden_nf)
            edge_feat: edge features,  shape (N, N, hidden_nf)

        Returns:
            Updated node features, shape (N, hidden_nf).
        """
        agg = jnp.sum(edge_feat, axis=1)  # (N, hidden_nf)
        inp = jnp.concatenate([h, agg], axis=-1)  # (N, 2*hidden_nf)
        out = jax.vmap(self.node_mlp)(inp)  # (N, hidden_nf)
        if self.recurrent:
            out = h + out
        return out

    # ── Forward ───────────────────────────────────────────────────────────────

    def __call__(self, h: jnp.ndarray, x: jnp.ndarray, L):
        """One EGNN layer forward pass.

        Args:
            h: node features, shape (N, hidden_nf)
            x: coordinates,   shape (N, d)
            L: box side length (scalar, traced)

        Returns:
            h_new: updated node features, shape (N, hidden_nf)
            x_new: updated coordinates,   shape (N, d)
        """
        radial, coord_diff = self._coord2radial(x, L)
        edge_feat = self._edge_model(h, radial)
        x_new = self._coord_model(x, coord_diff, edge_feat, L)
        h_new = self._node_model(h, edge_feat)
        return h_new, x_new


# ── EGNN stack ────────────────────────────────────────────────────────────────


class EGNN(eqx.Module):
    """Stack of E_GCL layers with input node embedding.

    Processes a single configuration (no batch dimension).

    Args:
        in_node_nf:   Input node feature dimension.
        hidden_nf:    Hidden feature dimension.
        n_layers:     Number of E_GCL layers.
        activation:   Activation function.
        recurrent:    Residual connections in node update.
        attention:    Attention-gated edge features.
        tanh:         Tanh + scale for coordinate updates.
        coords_range: Total coordinate update range (split across layers).
    """

    embedding: eqx.nn.Linear
    layers: list  # list[E_GCL]
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        n_layers: int,
        activation: Callable,
        recurrent: bool,
        attention: bool,
        tanh: bool,
        coords_range: float,
        *,
        key,
    ):
        k_emb, *k_layers = jax.random.split(key, n_layers + 1)

        self.embedding = eqx.nn.Linear(in_node_nf, hidden_nf, key=k_emb)
        self.n_layers = n_layers

        # Distribute coords_range evenly across layers (matches original code)
        layer_coords_range = float(coords_range) / n_layers

        self.layers = [
            E_GCL(
                hidden_nf=hidden_nf,
                activation=activation,
                recurrent=recurrent,
                attention=attention,
                tanh=tanh,
                coords_range=layer_coords_range,
                key=k_layers[i],
            )
            for i in range(n_layers)
        ]

    def __call__(self, h: jnp.ndarray, x: jnp.ndarray, L):
        """Forward pass through the EGNN stack.

        Args:
            h: initial node features, shape (N, in_node_nf)
            x: coordinates,           shape (N, d)
            L: box side length (scalar, traced)

        Returns:
            h_out: updated node features, shape (N, hidden_nf)
            x_out: updated coordinates,   shape (N, d)
        """
        h = jax.vmap(self.embedding)(h)
        for layer in self.layers:
            h, x = layer(h, x, L)
        # Note: the original EGNN has embedding_out(h) here, but EGNN_dynamics
        # only uses x_final and discards h entirely — so we skip it.
        return h, x


# ── Top-level velocity field ──────────────────────────────────────────────────


class ParticlesEGNNVelocity(eqx.Module):
    """EGNN velocity field for PBC particle systems.

    Implements the equivariant velocity field using an EGNN that respects
    E(n)-symmetry of the underlying physics. Designed for systems on a
    periodic flat torus of side length L.

    The velocity at each particle position is computed as the shortest-path
    (log-map) displacement from the original position to the EGNN-predicted
    updated position, preserving PBC correctness.

    Calling convention (matches ``ParticlesMLPVelocity`` and ``flow.py``)::

        velocity = ParticlesEGNNVelocity(...)
        v = velocity(t, x_dynamic, ctx_static)

    where ``x_dynamic`` is the dynamic partition of a ``ParticleSystem``
    (only ``positions`` is non-None) and ``ctx_static`` carries ``species``
    and ``box``.

    Args:
        N:            Number of particles.
        d:            Spatial dimension.
        n_species:    Number of distinct particle species.
        hidden_nf:    Width of all hidden representations in the EGNN.
        n_layers:     Number of E_GCL message-passing layers.
        recurrent:    Residual connection in node update (default True).
        attention:    Attention-gated edge features (default False).
        tanh:         Tanh + scale for coordinate updates (default False).
        coords_range: Total coordinate displacement scale when tanh=True
                      (default 15.0, split evenly across layers).
        key:          JAX PRNGKey for parameter initialisation.
    """

    egnn: EGNN
    species_embedding: eqx.nn.Embedding
    N: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)

    def __init__(
        self,
        N: int,
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
        self.N = N
        self.d = d
        self.n_species = n_species

        # Node feature dimension: time (1 scalar) + one-hot species (n_species)
        # Mirrors: in_node_nf = 1 + n_species in the original EGNN_dynamics
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

        # One-hot species embedding (frozen identity matrix).
        # Using eqx.nn.Embedding for consistent pytree handling; we override
        # the weight with the identity so it acts as a pure one-hot lookup.
        self.species_embedding = eqx.nn.Embedding(
            num_embeddings=n_species,
            embedding_size=n_species,
            key=k_spec,
        )
        id_weights = jnp.eye(n_species)
        object.__setattr__(
            self,
            "species_embedding",
            eqx.tree_at(lambda m: m.weight, self.species_embedding, id_weights),
        )

    def __call__(self, t: float, x, ctx):
        """Compute the velocity for a single particle configuration.

        Args:
            t:   Time scalar (float or 0-d array).
            x:   Dynamic partition of a ParticleSystem — only ``x.positions``
                 is non-None, shape ``(N, d)``.
            ctx: Static context partition — carries ``ctx.species`` (shape (N,))
                 and ``ctx.box`` (shape (d,)).

        Returns:
            A ParticleSystem-like object with ``positions`` containing the
            velocity for each particle (shape (N, d)), and ``species=None``,
            ``box=None``.
        """
        pos = x.positions  # (N, d)
        species = ctx.species  # (N,)  integer labels
        box = ctx.box  # (d,)

        # Assume cubic box — take the first element as L.
        # Consistent with training_particles.py:  L = float(source.box_size[0])
        L = box[0]

        # ── Build initial node features: [t_broadcast || one_hot(species)] ──
        # shape: (N, 1 + n_species)
        t_feat = jnp.broadcast_to(jnp.reshape(t, (1, 1)), (self.N, 1))  # (N, 1)
        species_feat = jax.vmap(self.species_embedding)(species)  # (N, n_species)
        h0 = jnp.concatenate([t_feat, species_feat], axis=-1)  # (N, 1+n_species)

        # ── EGNN forward (L passed as traced argument through the stack) ──────
        _, x_final = self.egnn(h0, pos, L)

        # ── Velocity = torus-geodesic displacement pos → x_final ──────────────
        # log_map: shortest displacement on the flat torus
        vel = log_map(pos, x_final, L)

        return type(x)(positions=vel, species=None, box=None)
