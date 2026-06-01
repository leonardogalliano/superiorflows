"""Tests for AbstractBijector and DistreqxBijectorWrapper."""

import equinox as eqx
import jax.numpy as jnp

from superiorflows.bijector import AbstractBijector, DistreqxBijectorWrapper


class ScaleBijector(AbstractBijector):
    """Simple scaling bijector for testing AbstractBijector defaults.

    y = scale * x
    """

    scale: float

    def forward_and_log_det(self, x, **kwargs):
        y = self.scale * x
        # log|det J_f(x)| = log|scale| * dimension
        dimension = x.size
        logdet = jnp.sum(jnp.log(jnp.abs(self.scale))) * dimension
        return y, logdet

    def inverse_and_log_det(self, y, **kwargs):
        x = y / self.scale
        dimension = y.size
        logdet = -jnp.sum(jnp.log(jnp.abs(self.scale))) * dimension
        return x, logdet


class MockDistreqxBijector(eqx.Module):
    """Mock class mimicking a distreqx bijector."""

    scale: float

    def forward(self, x):
        return self.scale * x

    def inverse(self, y):
        return y / self.scale

    def forward_and_log_det(self, x):
        y = self.scale * x
        logdet = jnp.log(jnp.abs(self.scale)) * x.size
        return y, logdet

    def inverse_and_log_det(self, y):
        x = y / self.scale
        logdet = -jnp.log(jnp.abs(self.scale)) * y.size
        return x, logdet


def test_abstract_bijector_defaults():
    """Verify that AbstractBijector default methods delegate correctly."""
    bijector = ScaleBijector(scale=2.0)
    x = jnp.array([1.0, 2.0, 3.0])

    y = bijector.forward(x)
    assert jnp.allclose(y, jnp.array([2.0, 4.0, 6.0]))

    x_rec = bijector.inverse(y)
    assert jnp.allclose(x_rec, x)

    fwd_logdet = bijector.forward_log_det_jacobian(x)
    assert jnp.allclose(fwd_logdet, jnp.log(2.0) * 3)

    inv_logdet = bijector.inverse_log_det_jacobian(y)
    assert jnp.allclose(inv_logdet, -jnp.log(2.0) * 3)


def test_distreqx_bijector_wrapper():
    """Verify that DistreqxBijectorWrapper forwards calls correctly."""
    mock_bijector = MockDistreqxBijector(scale=3.0)
    wrapper = DistreqxBijectorWrapper(mock_bijector)
    x = jnp.array([1.0, 2.0, 3.0])

    y = wrapper.forward(x)
    assert jnp.allclose(y, jnp.array([3.0, 6.0, 9.0]))

    x_rec = wrapper.inverse(y)
    assert jnp.allclose(x_rec, x)

    y_fwd, fwd_logdet = wrapper.forward_and_log_det(x)
    assert jnp.allclose(y_fwd, y)
    assert jnp.allclose(fwd_logdet, jnp.log(3.0) * 3)

    x_inv, inv_logdet = wrapper.inverse_and_log_det(y)
    assert jnp.allclose(x_inv, x)
    assert jnp.allclose(inv_logdet, -jnp.log(3.0) * 3)
