"""Tests for the state_context_partition mechanism and bijector integration.

Validates that the extended partition supports scalar-bool masks
(backwards compatibility with eqx.partition), integer index-array masks
(element-level selection), and their integration with the AbstractBijector
partition/merge layer and the ODEBijector backend.
"""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from superiorflows import Flow, ODEBijector
from superiorflows.bijector import AbstractBijector
from superiorflows.partition import (
    _compute_complement,
    merge_state,
    state_context_partition,
)

# ── Test state (generic PyTree for abstract ℝ^d testing) ──────────────


class VectorState(eqx.Module):
    """Minimal PyTree state for testing partitions on ℝ^d."""

    values: jnp.ndarray
    labels: jnp.ndarray


# ── Test state (particle-like PyTree) ─────────────────────────────────


class MockParticleSystem(eqx.Module):
    """Mimics the structure of ParticleSystem for testing."""

    positions: jnp.ndarray  # (N, d)
    species: jnp.ndarray  # (N,)
    box: jnp.ndarray  # (d,)


# ── Helper fixtures ───────────────────────────────────────────────────


@pytest.fixture
def particle_state():
    N, d = 8, 2
    return MockParticleSystem(
        positions=jnp.arange(N * d, dtype=float).reshape(N, d),
        species=jnp.array([0, 0, 1, 1, 0, 0, 1, 1]),
        box=jnp.array([10.0, 10.0]),
    )


@pytest.fixture
def vector_state():
    return VectorState(
        values=jnp.arange(10, dtype=float),
        labels=jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )


# ======================================================================
# _compute_complement
# ======================================================================


class TestComputeComplement:
    def test_basic(self):
        indices = jnp.array([1, 3, 5, 7])
        complement = _compute_complement(indices, 8)
        assert jnp.array_equal(complement, jnp.array([0, 2, 4, 6]))

    def test_first_elements(self):
        indices = jnp.array([0, 1, 2])
        complement = _compute_complement(indices, 5)
        assert jnp.array_equal(complement, jnp.array([3, 4]))

    def test_single_element(self):
        indices = jnp.array([3])
        complement = _compute_complement(indices, 5)
        assert jnp.array_equal(complement, jnp.array([0, 1, 2, 4]))

    def test_all_elements(self):
        indices = jnp.arange(5)
        complement = _compute_complement(indices, 5)
        assert complement.shape == (0,)

    def test_jit_compatible(self):
        @jax.jit
        def f(idx):
            return _compute_complement(idx, 8)

        result = f(jnp.array([0, 3, 5]))
        assert jnp.array_equal(result, jnp.array([1, 2, 4, 6, 7]))


# ======================================================================
# state_context_partition — scalar-bool masks (eqx.partition equivalence)
# ======================================================================


class TestScalarBoolPartition:
    def test_equivalence_with_eqx_partition(self, particle_state):
        """Standard mask must produce identical output to eqx.partition."""
        mask = MockParticleSystem(positions=True, species=False, box=False)
        dyn_ext, ctx_ext, _ = state_context_partition(particle_state, mask)
        dyn_eqx, ctx_eqx = eqx.partition(particle_state, mask)

        assert jnp.array_equal(dyn_ext.positions, dyn_eqx.positions)
        assert dyn_ext.species is None and dyn_eqx.species is None
        assert ctx_ext.positions is None and ctx_eqx.positions is None
        assert jnp.array_equal(ctx_ext.species, ctx_eqx.species)
        assert jnp.array_equal(ctx_ext.box, ctx_eqx.box)

    def test_all_dynamic(self, particle_state):
        mask = MockParticleSystem(positions=True, species=True, box=True)
        dyn, ctx, _ = state_context_partition(particle_state, mask)
        assert jnp.array_equal(dyn.positions, particle_state.positions)
        assert jnp.array_equal(dyn.species, particle_state.species)
        assert ctx.positions is None
        assert ctx.species is None

    def test_callable_mask(self, particle_state):
        """Callable mask (eqx.is_inexact_array) must work."""
        dyn, ctx, _ = state_context_partition(particle_state, eqx.is_inexact_array)
        # Float arrays → dynamic
        assert jnp.array_equal(dyn.positions, particle_state.positions)
        assert jnp.array_equal(dyn.box, particle_state.box)
        # Integer array → not inexact → static
        assert dyn.species is None
        assert jnp.array_equal(ctx.species, particle_state.species)

    def test_callable_mask_lambda(self, particle_state):
        """Lambda wrapper (the actual default in ODEBijector) must work."""

        def mask_fn(x):
            return jax.tree.map(eqx.is_inexact_array, x)

        dyn, ctx, _ = state_context_partition(particle_state, mask_fn)
        assert jnp.array_equal(dyn.positions, particle_state.positions)
        assert jnp.array_equal(dyn.box, particle_state.box)


# ======================================================================
# state_context_partition — index-array masks (new behaviour)
# ======================================================================


class TestIndexArrayPartition:
    def test_basic_partition(self, particle_state):
        patch_indices = jnp.array([1, 3, 5, 7])
        mask = MockParticleSystem(positions=patch_indices, species=False, box=False)
        dyn, ctx, spec = state_context_partition(particle_state, mask)

        assert dyn.positions.shape == (4, 2)
        assert jnp.array_equal(dyn.positions, particle_state.positions[patch_indices])
        assert dyn.species is None
        assert dyn.box is None

        # Context contains the complement, not the full array
        assert ctx.positions.shape == (4, 2)
        complement = jnp.array([0, 2, 4, 6])
        assert jnp.array_equal(ctx.positions, particle_state.positions[complement])
        assert jnp.array_equal(ctx.species, particle_state.species)
        assert jnp.array_equal(ctx.box, particle_state.box)

    def test_single_element_patch(self, particle_state):
        patch_indices = jnp.array([3])
        mask = MockParticleSystem(positions=patch_indices, species=False, box=False)
        dyn, ctx, _ = state_context_partition(particle_state, mask)
        assert dyn.positions.shape == (1, 2)
        assert ctx.positions.shape == (7, 2)

    def test_mixed_mask(self, particle_state):
        """Some leaves scalar-bool, some index arrays."""
        patch_indices = jnp.array([0, 2, 4])
        mask = MockParticleSystem(positions=patch_indices, species=False, box=True)
        dyn, ctx, _ = state_context_partition(particle_state, mask)

        assert dyn.positions.shape == (3, 2)
        assert dyn.species is None
        assert jnp.array_equal(dyn.box, particle_state.box)
        assert ctx.positions.shape == (5, 2)
        assert ctx.box is None

    def test_multiple_indexed_leaves(self):
        """Index masks on multiple leaves simultaneously."""
        state = VectorState(
            values=jnp.arange(10, dtype=float),
            labels=jnp.arange(10),
        )
        indices = jnp.array([2, 5, 8])
        mask = VectorState(values=indices, labels=indices)
        dyn, ctx, _ = state_context_partition(state, mask)

        assert dyn.values.shape == (3,)
        assert dyn.labels.shape == (3,)
        assert jnp.array_equal(dyn.values, state.values[indices])
        assert jnp.array_equal(dyn.labels, state.labels[indices])
        assert ctx.values.shape == (7,)
        assert ctx.labels.shape == (7,)


# ======================================================================
# merge_state
# ======================================================================


class TestMergeState:
    def test_scalar_bool_merge(self, particle_state):
        mask = MockParticleSystem(positions=True, species=False, box=False)
        dyn, ctx, spec = state_context_partition(particle_state, mask)

        new_dyn = MockParticleSystem(positions=dyn.positions + 100.0, species=None, box=None)
        merged = merge_state(new_dyn, particle_state, spec)
        assert jnp.array_equal(merged.positions, particle_state.positions + 100.0)
        assert jnp.array_equal(merged.species, particle_state.species)
        assert jnp.array_equal(merged.box, particle_state.box)

    def test_index_mask_merge(self, particle_state):
        patch_indices = jnp.array([1, 3, 5, 7])
        mask = MockParticleSystem(positions=patch_indices, species=False, box=False)
        dyn, ctx, spec = state_context_partition(particle_state, mask)

        new_dyn = MockParticleSystem(positions=dyn.positions + 100.0, species=None, box=None)
        merged = merge_state(new_dyn, particle_state, spec)

        assert jnp.all(merged.positions[patch_indices] == particle_state.positions[patch_indices] + 100.0)
        complement = jnp.array([0, 2, 4, 6])
        assert jnp.array_equal(merged.positions[complement], particle_state.positions[complement])
        assert jnp.array_equal(merged.species, particle_state.species)

    def test_merge_is_jit_compatible(self, particle_state):
        patch_indices = jnp.array([1, 3, 5, 7])

        @jax.jit
        def f(positions, idx):
            state = MockParticleSystem(
                positions=positions,
                species=jnp.array([0, 0, 1, 1, 0, 0, 1, 1]),
                box=jnp.array([10.0, 10.0]),
            )
            mask_ = MockParticleSystem(positions=idx, species=False, box=False)
            dyn, ctx, spec = state_context_partition(state, mask_)
            new_dyn = MockParticleSystem(positions=dyn.positions * 2.0, species=None, box=None)
            return merge_state(new_dyn, state, spec)

        result = f(particle_state.positions, patch_indices)
        assert result.positions is not None

        # Different indices, no recompilation
        result2 = f(particle_state.positions, jnp.array([0, 2, 4, 6]))
        assert result2.positions is not None


# ======================================================================
# AbstractBijector — partition/merge at the bijector level
# ======================================================================


class ScaleBijector(AbstractBijector):
    """Trivial bijector for testing: y = scale * x, log|det| = d * log|scale|."""

    scale: float

    def _forward_and_log_det(self, x, *, args=None, **kwargs):
        y = jax.tree.map(lambda leaf: self.scale * leaf, x)
        d = sum(leaf.size for leaf in jax.tree.leaves(x))
        return y, d * jnp.log(jnp.abs(self.scale))

    def _inverse_and_log_det(self, y, *, args=None, **kwargs):
        x = jax.tree.map(lambda leaf: leaf / self.scale, y)
        d = sum(leaf.size for leaf in jax.tree.leaves(y))
        return x, -d * jnp.log(jnp.abs(self.scale))


class TestAbstractBijectorPartition:
    """Test that partition/merge works at the AbstractBijector level,
    independently of any ODE mechanism."""

    def test_plain_array_forward_inverse(self):
        """Plain array, default mask: everything is dynamic."""
        b = ScaleBijector(scale=2.0)
        x = jnp.array([1.0, 2.0, 3.0])
        y = b.forward(x)
        assert jnp.allclose(y, 2.0 * x)
        x_recovered = b.inverse(y)
        assert jnp.allclose(x_recovered, x, atol=1e-6)

    def test_plain_array_log_det(self):
        b = ScaleBijector(scale=3.0)
        x = jnp.array([1.0, 2.0, 3.0])
        y, logdet = b.forward_and_log_det(x)
        assert jnp.allclose(y, 3.0 * x)
        assert jnp.allclose(logdet, 3.0 * jnp.log(3.0))

    def test_pytree_with_scalar_bool_mask(self, particle_state):
        """PyTree state with standard bool mask — positions scaled, rest static."""
        b = ScaleBijector(
            scale=2.0,
            dynamic_mask=MockParticleSystem(positions=True, species=False, box=False),
        )
        y = b.forward(particle_state)
        assert jnp.allclose(y.positions, 2.0 * particle_state.positions)
        assert jnp.array_equal(y.species, particle_state.species)
        assert jnp.array_equal(y.box, particle_state.box)

    def test_pytree_with_index_mask(self, particle_state):
        """Index mask: only selected elements are transformed."""
        patch_indices = jnp.array([1, 3, 5])
        complement = jnp.array([0, 2, 4, 6, 7])
        b = ScaleBijector(
            scale=2.0,
            dynamic_mask=MockParticleSystem(positions=patch_indices, species=False, box=False),
        )
        y = b.forward(particle_state)

        assert jnp.allclose(y.positions[patch_indices], 2.0 * particle_state.positions[patch_indices])
        assert jnp.array_equal(y.positions[complement], particle_state.positions[complement])
        assert jnp.array_equal(y.species, particle_state.species)

    def test_per_call_mask_override(self, particle_state):
        """dynamic_mask kwarg overrides the default on each call."""
        b = ScaleBijector(scale=3.0)
        mask_a = MockParticleSystem(positions=jnp.array([0, 1]), species=False, box=False)
        mask_b = MockParticleSystem(positions=jnp.array([6, 7]), species=False, box=False)

        y_a = b.forward(particle_state, dynamic_mask=mask_a)
        y_b = b.forward(particle_state, dynamic_mask=mask_b)

        assert jnp.allclose(y_a.positions[:2], 3.0 * particle_state.positions[:2])
        assert jnp.array_equal(y_a.positions[2:], particle_state.positions[2:])

        assert jnp.array_equal(y_b.positions[:6], particle_state.positions[:6])
        assert jnp.allclose(y_b.positions[6:], 3.0 * particle_state.positions[6:])


# ======================================================================
# ODEBijector integration — standard mask (regression tests)
# ======================================================================


class SimpleVelocity(eqx.Module):
    """v(t, x, args) = -t * x — simple contracting field on ℝ^d."""

    def __call__(self, t, x, args):
        return -t * x


@pytest.fixture
def simple_flow():
    import distreqx.distributions as dsx

    dim = 6
    base = dsx.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim))
    bijector = ODEBijector(
        SimpleVelocity(),
        stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-7),
    )
    return Flow(bijector, base)


def test_standard_mask_forward_inverse(simple_flow):
    """Verify ODEBijector still works correctly with standard masks."""
    key = jax.random.key(42)
    x0 = simple_flow.base_distribution.sample(key=key)
    x1 = simple_flow.bijector.forward(x0)
    x0_recovered = simple_flow.bijector.inverse(x1)
    assert jnp.allclose(x0, x0_recovered, atol=1e-5)


def test_standard_mask_log_prob(simple_flow):
    """Verify log_prob is consistent with push_forward_and_log_prob."""
    key = jax.random.key(42)
    x0 = simple_flow.base_distribution.sample(key=key)
    x1, logq1 = simple_flow.push_forward_and_log_prob(x0)
    logq_inv = simple_flow.log_prob(x1)
    assert jnp.allclose(logq1, logq_inv, atol=1e-4)


# ======================================================================
# ODEBijector integration — index mask (conditional updates)
# ======================================================================


class ConditionalVelocity(eqx.Module):
    """Velocity field that conditions on context.

    For a MockParticleSystem state:
    - x_dynamic.positions: (n, d) — patch
    - args.positions: (N-n, d) — context complement
    - args.species: (N,)
    - args.box: (d,)

    Returns sin(patch) + 0.1 * mean(context).
    """

    def __call__(self, t, x_dynamic, args):
        patch_pos = x_dynamic.positions
        ctx_pos = args.positions
        if ctx_pos is not None:
            ctx_effect = 0.1 * jnp.mean(ctx_pos, axis=0, keepdims=True)
        else:
            ctx_effect = 0.0
        vel = -t * jnp.sin(patch_pos) + ctx_effect
        return MockParticleSystem(positions=vel, species=None, box=None)


@pytest.fixture
def patch_bijector():
    return ODEBijector(
        ConditionalVelocity(),
        dt0=0.1,
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        augmented_solver=dfx.Euler(),
        augmented_stepsize_controller=dfx.ConstantStepSize(),
        extra_args={"max_steps": 1000},
        augmented_extra_args={"max_steps": 1000},
    )


@pytest.fixture
def full_state():
    N, d = 8, 2
    return MockParticleSystem(
        positions=jax.random.normal(jax.random.key(0), (N, d)),
        species=jnp.array([0, 0, 1, 1, 0, 0, 1, 1]),
        box=jnp.array([10.0, 10.0]),
    )


def test_patch_forward(patch_bijector, full_state):
    """Patch forward integration only modifies patch particles."""
    patch_indices = jnp.array([1, 3, 5, 7])
    complement = jnp.array([0, 2, 4, 6])
    mask = MockParticleSystem(positions=patch_indices, species=False, box=False)

    y = patch_bijector.forward(full_state, dynamic_mask=mask)

    # Context particles unchanged
    assert jnp.array_equal(y.positions[complement], full_state.positions[complement])
    # Patch particles changed
    assert not jnp.array_equal(y.positions[patch_indices], full_state.positions[patch_indices])
    # Species and box unchanged
    assert jnp.array_equal(y.species, full_state.species)
    assert jnp.array_equal(y.box, full_state.box)


def test_patch_forward_inverse(patch_bijector, full_state):
    """Patch forward followed by inverse recovers original state."""
    patch_indices = jnp.array([1, 3, 5, 7])
    mask = MockParticleSystem(positions=patch_indices, species=False, box=False)

    y = patch_bijector.forward(full_state, dynamic_mask=mask)
    x_recovered = patch_bijector.inverse(y, dynamic_mask=mask)

    assert jnp.allclose(x_recovered.positions, full_state.positions, atol=0.2)


def test_patch_forward_and_log_det(patch_bijector, full_state):
    """forward_and_log_det produces finite log-det for patch updates."""
    patch_indices = jnp.array([1, 3, 5, 7])
    mask = MockParticleSystem(positions=patch_indices, species=False, box=False)

    y, log_det = patch_bijector.forward_and_log_det(full_state, dynamic_mask=mask)

    assert jnp.isfinite(log_det)
    assert y.positions.shape == full_state.positions.shape


def test_patch_per_call_mask_override(patch_bijector, full_state):
    """dynamic_mask can be overridden per call via kwargs."""
    mask_a = MockParticleSystem(positions=jnp.array([0, 1, 2, 3]), species=False, box=False)
    mask_b = MockParticleSystem(positions=jnp.array([4, 5, 6, 7]), species=False, box=False)

    y_a = patch_bijector.forward(full_state, dynamic_mask=mask_a)
    y_b = patch_bijector.forward(full_state, dynamic_mask=mask_b)

    # Mask A: particles 0-3 changed, 4-7 unchanged
    assert jnp.array_equal(y_a.positions[4:], full_state.positions[4:])
    assert not jnp.array_equal(y_a.positions[:4], full_state.positions[:4])

    # Mask B: particles 4-7 changed, 0-3 unchanged
    assert jnp.array_equal(y_b.positions[:4], full_state.positions[:4])
    assert not jnp.array_equal(y_b.positions[4:], full_state.positions[4:])


def test_patch_log_det_dimension(patch_bijector, full_state):
    """Divergence computation scales with patch size, not full system."""
    patch_indices = jnp.array([1, 3])
    mask = MockParticleSystem(positions=patch_indices, species=False, box=False)

    # Verify the Jacobian dimension is nd (patch_size * d), not Nd
    dyn, ctx, spec = state_context_partition(full_state, mask)
    dyn_flat, unravel = jax.flatten_util.ravel_pytree(dyn)
    assert dyn_flat.size == 2 * 2  # n=2 particles, d=2 dimensions

    # Also verify forward_and_log_det works with this mask
    y, log_det = patch_bijector.forward_and_log_det(full_state, dynamic_mask=mask)
    assert jnp.isfinite(log_det)


def test_patch_vmap_over_batched_states(patch_bijector, full_state):
    """vmap over a batch of states with the same patch mask."""
    patch_indices = jnp.array([1, 3, 5, 7])
    mask = MockParticleSystem(positions=patch_indices, species=False, box=False)

    B = 4
    keys = jax.random.split(jax.random.key(0), B)
    batch = MockParticleSystem(
        positions=jax.vmap(lambda k: jax.random.normal(k, (8, 2)))(keys),
        species=jnp.tile(full_state.species, (B, 1)),
        box=jnp.tile(full_state.box, (B, 1)),
    )

    ys = jax.vmap(lambda x: patch_bijector.forward(x, dynamic_mask=mask))(batch)
    assert ys.positions.shape == (B, 8, 2)


def test_flat_array_state_with_patch():
    """Patch on a plain array state (ℝ^d case, no PyTree wrapping)."""
    x = jnp.arange(10, dtype=float)
    indices = jnp.array([2, 5, 8])
    complement = _compute_complement(indices, 10)

    dyn, ctx, spec = state_context_partition(x, indices)

    assert dyn.shape == (3,)
    assert jnp.array_equal(dyn, x[indices])
    assert ctx.shape == (7,)
    assert jnp.array_equal(ctx, x[complement])

    new_dyn = dyn * 10.0
    merged = merge_state(new_dyn, x, spec)
    assert jnp.array_equal(merged[indices], x[indices] * 10.0)
    assert jnp.array_equal(merged[complement], x[complement])
