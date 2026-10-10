"""Regression test: TSM gradients stay finite when a sampled time is exactly t = 0.

`jax.random.uniform` draws t = 0.0 exactly with probability ~1e-7 per float32 sample, i.e. about
once every 8000 steps at batch 1024. The TSM target for the t >= 0.5 half divides by β(t) = t;
that branch is discarded by `jnp.where` at t = 0, but its 0/0 NaN gradient was multiplied by 0,
giving NaN gradients and permanently NaN parameters.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from superiorflows.train import StochasticInterpolantLoss


class Field(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, d, key):
        self.mlp = eqx.nn.MLP(d + 1, d, 16, 2, key=key)

    def __call__(self, t, x, args=None):
        return self.mlp(jnp.concatenate([x, jnp.atleast_1d(t)]))


class Pair(eqx.Module):
    velocity_field: Field
    denoiser: Field


def forced_times(key, shape, *args, **kwargs):
    # Includes both endpoints' worst cases for the TSM ratios: t = 0 exactly, and t just below 1.
    base = jnp.array([0.0, 0.25, 0.5, 1.0 - 2.0**-24])
    return jnp.resize(base, shape)


@pytest.mark.parametrize("one_sided", [False, True])
def test_tsm_gradients_finite_at_t_zero(monkeypatch, one_sided):
    d, n = 3, 8
    kv, kd, k0, k1, kl = jax.random.split(jax.random.key(0), 5)
    model = Pair(Field(d, kv), Field(d, kd))
    common = dict(
        interpolant=lambda t, x0, x1: (1 - t) * x0 + t * x1,
        get_denoiser=lambda m: m.denoiser,
        base_score_fn=lambda x: -x,
        target_score_fn=lambda x: -(x - 1.0),
        tsm_weight=1.0,
    )
    if one_sided:
        loss = StochasticInterpolantLoss(compute_velocity_loss=False, **common)
    else:
        loss = StochasticInterpolantLoss(
            gamma=lambda t: 0.5 * t * (1 - t), get_velocity=lambda m: m.velocity_field, **common
        )
    batch = (jax.random.normal(k0, (n, d)), 1.0 + jax.random.normal(k1, (n, d)))

    monkeypatch.setattr(jax.random, "uniform", forced_times)
    (value, aux), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, batch, kl)

    assert jnp.isfinite(value)
    assert all(jnp.isfinite(v) for v in aux.values())
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
