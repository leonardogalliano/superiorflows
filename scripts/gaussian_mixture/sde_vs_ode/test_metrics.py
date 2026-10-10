"""Checks of the importance-sampling metrics against cases with known answers.

Run with ``uv run pytest scripts/gaussian_mixture/sde_vs_ode/test_metrics.py``.
"""

import numpy as np
import pytest

from scripts.gaussian_mixture.sde_vs_ode.metrics import (
    ess_fraction,
    log_z_hat,
    mode1_indicator,
    psis_khat,
    replicate_metrics,
)


def test_exact_sampler_has_unit_weights():
    log_w = np.zeros(1000)
    assert log_z_hat(log_w) == pytest.approx(0.0, abs=1e-12)
    assert ess_fraction(log_w) == pytest.approx(1.0, abs=1e-12)


def test_log_z_of_scaled_weights():
    log_w = np.log(np.full(500, 3.0))
    assert log_z_hat(log_w) == pytest.approx(np.log(3.0), abs=1e-12)


def test_ess_of_two_valued_weights():
    # Half the weights equal 1, half are negligible: ESS/n = (n/2)^2 / (n * n/2) = 1/2.
    log_w = np.concatenate([np.zeros(100), np.full(100, -1e3)])
    assert ess_fraction(log_w) == pytest.approx(0.5, abs=1e-12)


def test_ess_of_lognormal_weights():
    # For log w ~ N(0, s^2): E[w]^2 / E[w^2] = exp(-s^2).
    rng = np.random.default_rng(0)
    s = 0.5
    log_w = rng.normal(0.0, s, 400_000)
    assert ess_fraction(log_w) == pytest.approx(np.exp(-(s**2)), rel=0.01)


@pytest.mark.parametrize("alpha", [1.25, 2.0, 4.0])
def test_khat_recovers_pareto_tail_index(alpha):
    # Pareto(alpha) weights have generalised-Pareto tail shape k = 1/alpha.
    rng = np.random.default_rng(1)
    estimates = [psis_khat(np.log(rng.pareto(alpha, 20_000) + 1.0)) for _ in range(20)]
    assert np.mean(estimates) == pytest.approx(1.0 / alpha, abs=0.06)


def test_khat_small_for_light_tails():
    rng = np.random.default_rng(2)
    assert psis_khat(rng.normal(0.0, 0.3, 4000)) < 0.3


def test_mode_indicator_and_snis():
    a, d = 5.25, 8
    rng = np.random.default_rng(3)
    x1 = -a + 0.3 * rng.normal(size=(200, d))
    x2 = a + 0.3 * rng.normal(size=(100, d))
    x = np.concatenate([x1, x2])
    assert mode1_indicator(x).sum() == 200
    # Up-weighting mode 2 by 2x gives SNIS mode-1 probability 200 / (200 + 2 * 100) = 1/2.
    log_w = np.concatenate([np.zeros(200), np.full(100, np.log(2.0))])
    row = replicate_metrics(x, log_w)
    assert row["status"] == "OK"
    assert row["mode1_unweighted"] == pytest.approx(2.0 / 3.0)
    assert row["mode1_snis"] == pytest.approx(0.5)


def test_nonfinite_replicate_is_flagged_not_dropped():
    x = np.zeros((10, 2))
    log_w = np.zeros(10)
    log_w[3] = np.nan
    row = replicate_metrics(x, log_w)
    assert row["status"] == "FAILED_NONFINITE"
    assert row["n_nonfinite_logw"] == 1
    assert np.isnan(row["log_z"]) and np.isnan(row["ess"])


def test_khat_infinite_when_one_sample_dominates():
    log_w = np.zeros(2000)
    log_w[0] = 1e4
    assert psis_khat(log_w) == np.inf
