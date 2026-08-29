"""Tests for AbstractTransport, partitioning helpers, and transport polymorphism."""

import jax
import jax.numpy as jnp

from superiorflows.bijector import AbstractBijector, _pack_args
from superiorflows.transport import AbstractTransport


class MockDeterministicTransport(AbstractTransport):
    """Simple linear transport y = 2x."""

    def _forward_and_log_weight(self, x, *, args=None, **kwargs):
        return jax.tree.map(lambda leaf: leaf * 2.0, x), -jnp.log(2.0) * sum(leaf.size for leaf in jax.tree.leaves(x))


class MockBijector(AbstractBijector):
    """Simple scaling bijector."""

    def _forward_and_log_det(self, x, *, args=None, **kwargs):
        return x * 3.0, x.shape[0] * jnp.log(3.0)

    def _inverse_and_log_det(self, y, *, args=None, **kwargs):
        return y / 3.0, -y.shape[0] * jnp.log(3.0)


def test_abstract_transport_interface():
    transport = MockDeterministicTransport()
    x = jnp.array([1.0, 2.0, 3.0])

    y = transport.forward(x)
    assert jnp.allclose(y, jnp.array([2.0, 4.0, 6.0]))

    y, log_w = transport.forward_and_log_weight(x)
    assert jnp.allclose(y, jnp.array([2.0, 4.0, 6.0]))
    assert jnp.allclose(log_w, -3.0 * jnp.log(2.0))


def test_bijector_log_weight_bridge():
    """Verify that bijector _forward_and_log_weight bridges to -log_det."""
    bijector = MockBijector()
    x = jnp.array([1.0, 2.0])

    y, log_det = bijector.forward_and_log_det(x)
    assert jnp.allclose(log_det, 2.0 * jnp.log(3.0))

    y_lw, log_w = bijector.forward_and_log_weight(x)
    assert jnp.allclose(y_lw, y)
    assert jnp.allclose(log_w, -log_det)


def test_transport_dynamic_mask():
    transport = MockDeterministicTransport()
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.array([0, 2])

    y = transport.forward(x, dynamic_mask=mask)
    # Dynamic indices 0 and 2 are doubled; static 1 and 3 are preserved
    assert jnp.allclose(y, jnp.array([2.0, 2.0, 6.0, 4.0]))

    y, log_w = transport.forward_and_log_weight(x, dynamic_mask=mask)
    assert jnp.allclose(y, jnp.array([2.0, 2.0, 6.0, 4.0]))
    assert jnp.allclose(log_w, -2.0 * jnp.log(2.0))


def test_pack_args_helper():
    ctx = {"static": 1}
    # No index meta, no user args
    assert _pack_args(ctx, None, None) == ctx
    # No index meta, with user args
    assert _pack_args(ctx, None, "user") == (ctx, "user")
    # With index meta, no user args
    assert _pack_args(ctx, "meta", None) == (ctx, "meta")
    # With index meta and user args
    assert _pack_args(ctx, "meta", "user") == (ctx, ("meta", "user"))
