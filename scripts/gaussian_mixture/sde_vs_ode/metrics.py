"""Importance-sampling metrics for one replicate of n samples.

All functions take log-weights ``log_w = log ρ★(y) − log_weight(y)`` (ρ★ normalised, so Z = 1).
Non-finite inputs are never silently dropped: :func:`replicate_metrics` reports them and
returns NaN estimates for a replicate that contains any.
"""

import math

import numpy as np
from scipy.special import logsumexp


def log_z_hat(log_w: np.ndarray) -> float:
    """log of the importance-sampling estimate of Z, ``log mean(w)``."""
    return float(logsumexp(log_w) - math.log(len(log_w)))


def ess_fraction(log_w: np.ndarray) -> float:
    """Kish effective sample size divided by n, ``(Σw)² / (n Σw²)``."""
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    return float(w.sum() ** 2 / (len(w) * np.sum(w**2)))


def _gpd_fit(exceedances: np.ndarray) -> float:
    """Generalised-Pareto shape k by Zhang & Stephens (2009), with the PSIS weakly informative prior.

    ``exceedances`` must be sorted ascending and positive. Follows Vehtari et al. (2024),
    *Pareto smoothed importance sampling*, as implemented in ArviZ/loo.
    """
    n = len(exceedances)
    prior_bs, prior_k = 3.0, 10.0
    m_est = 30 + int(math.sqrt(n))
    b = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b /= prior_bs * exceedances[int(n / 4 + 0.5) - 1]
    b += 1.0 / exceedances[-1]
    k = np.log1p(-b[:, None] * exceedances).mean(axis=1)
    len_scale = n * (np.log(-(b / k)) - k - 1.0)
    weights = 1.0 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    keep = weights >= 10 * np.finfo(float).eps
    weights, b = weights[keep], b[keep]
    weights /= weights.sum()
    b_post = np.sum(b * weights)
    k_post = np.log1p(-b_post * exceedances).mean()
    return float((n * k_post + prior_k * 0.5) / (n + prior_k))


def psis_khat(log_w: np.ndarray) -> float:
    """Pareto-k̂ diagnostic of the weight tail; above 0.7 IS estimates (and the ESS) are unreliable.

    Returns +inf when at most 4 samples carry all the tail weight (a degenerate tail with no
    usable fit): that is the most unreliable case, not a missing value.
    """
    n = len(log_w)
    tail_len = int(math.ceil(min(0.2 * n, 3.0 * math.sqrt(n))))
    lw = np.sort(log_w - np.max(log_w))
    cutoff = max(lw[-tail_len - 1], math.log(np.finfo(float).tiny))
    tail = np.exp(lw[-tail_len:]) - math.exp(cutoff)
    tail = tail[tail > 0]
    if len(tail) <= 4:
        return float("inf")
    return _gpd_fit(np.sort(tail))


def mode1_indicator(x: np.ndarray) -> np.ndarray:
    """Membership of the weight-2/3 mode centred at −a·1.

    The modes sit at ∓a·1. Within a mode the coordinate sum has mean ∓a·d and standard
    deviation at most √(0.2·d), so the sign of the sum is wrong only beyond a·√(5d) ≥ 16
    standard deviations (a = 5.25, d ≥ 2): misclassification is negligible.
    """
    return x.sum(axis=-1) < 0.0


def replicate_metrics(x: np.ndarray, log_w: np.ndarray) -> dict:
    """All metrics for one replicate. ``x`` has shape (n, d); ``log_w`` shape (n,)."""
    x = np.asarray(x, dtype=np.float64)
    log_w = np.asarray(log_w, dtype=np.float64)
    finite_x = np.all(np.isfinite(x), axis=-1)
    finite_w = np.isfinite(log_w)
    row = {
        "n": int(len(log_w)),
        "n_nonfinite_x": int(np.sum(~finite_x)),
        "n_nonfinite_logw": int(np.sum(~finite_w)),
        "mode1_unweighted": float(np.mean(mode1_indicator(x[finite_x]))) if finite_x.any() else float("nan"),
        "logw_max": float(np.max(log_w[finite_w])) if finite_w.any() else float("nan"),
        "logw_median": float(np.median(log_w[finite_w])) if finite_w.any() else float("nan"),
    }
    if row["n_nonfinite_x"] or row["n_nonfinite_logw"]:
        row.update(status="FAILED_NONFINITE", log_z=float("nan"), ess=float("nan"), khat=float("nan"))
        row["mode1_snis"] = float("nan")
        return row
    lw = log_w - np.max(log_w)
    w = np.exp(lw)
    row.update(
        status="OK",
        log_z=log_z_hat(log_w),
        ess=ess_fraction(log_w),
        khat=psis_khat(log_w),
        mode1_snis=float(np.sum(w * mode1_indicator(x)) / np.sum(w)),
    )
    return row
