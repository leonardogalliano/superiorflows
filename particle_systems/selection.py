"""Random-point selection protocol for patch-based MCMC.

Implements a selection protocol π(p, c | x) for particle systems on periodic
flat tori (PBC). The protocol:
  1. Draws a random anchor point y ~ Uniform([0, box)^d).
  2. Optionally draws a stochastic radius r ~ max(N(r_mean, σ_r), 0.1).
  3. Selects patch = particles within distance r of y, capped at N_patch.
  4. Selects context = particles with r ≤ d(y, ·) < r_context, capped at N_context.

Provides the soft log-ratio log π(p,c|y,x') / π(p,c|y,x) for the
Metropolis–Hastings acceptance computation.

Padding convention: both patch_idx and context_idx have fixed shapes (N_patch,)
and (N_context,) respectively. When fewer particles fall within the shell,
invalid slots duplicate the last valid index. After sorting, the layout is:
    [valid_0, valid_1, ..., valid_{n-1}, last_valid, last_valid, ...]
where valid_i are unique particle indices in ascending order.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["SelectionResult", "torus_distances_from_point", "random_point_select", "soft_log_pi_ratio"]


class SelectionResult(NamedTuple):
    """Result of a random-point selection protocol.

    Attributes:
        anchor: Random point in the box, shape (d,).
        radius: Drawn (or fixed) selection radius, scalar.
        patch_idx: Particle indices in the patch, shape (N_patch,). First n_patch
            entries are valid; remaining duplicate the last valid index.
        context_idx: Particle indices in the context shell, shape (N_context,).
            Same padding convention as patch_idx.
        n_patch: Actual number of valid patch particles, scalar.
        n_context: Actual number of valid context particles, scalar.
    """

    anchor: jnp.ndarray
    radius: jnp.ndarray
    patch_idx: jnp.ndarray
    context_idx: jnp.ndarray
    n_patch: jnp.ndarray
    n_context: jnp.ndarray


def torus_distances_from_point(point, positions, box):
    """Minimum-image distances from a point to all particles on a periodic flat torus.

    Args:
        point: Anchor point, shape (d,).
        positions: Particle positions, shape (N, d).
        box: Box side lengths, shape (d,).

    Returns:
        Distances from point to each particle, shape (N,).
    """
    diff = positions - point[None, :]
    diff = diff - jnp.round(diff / box) * box
    return jnp.sqrt(jnp.sum(diff**2, axis=-1))


def random_point_select(N_patch, N_context, r_patch, r_context, sigma_r=0.0):
    """Build a random-point selection protocol.

    When sigma_r > 0, the selection radius is drawn from
    r ~ max(N(r_patch, sigma_r), 0.1). When sigma_r = 0, the radius is
    fixed at r_patch.

    Args:
        N_patch: Maximum patch size (static, determines output shape).
        N_context: Maximum context size (static, determines output shape).
        r_patch: Mean patch radius (or fixed radius when sigma_r=0).
        r_context: Context shell outer radius.
        sigma_r: Standard deviation of the radius distribution.

    Returns:
        A callable ``(key, x) → SelectionResult``.
    """

    def select(key, x):
        k_point, k_radius = jax.random.split(key)
        positions = x.positions
        N = positions.shape[0]

        anchor = jax.random.uniform(k_point, (positions.shape[1],), maxval=x.box)

        radius = jnp.where(
            sigma_r > 0,
            jnp.maximum(r_patch + sigma_r * jax.random.normal(k_radius, ()), 0.1),
            jnp.asarray(r_patch, dtype=positions.dtype),
        )

        dists = torus_distances_from_point(anchor, positions, x.box)

        patch_idx, n_patch = _select_shell(dists, dists < radius, N_patch, N)
        context_idx, n_context = _select_shell(dists, (dists >= radius) & (dists < r_context), N_context, N)

        return SelectionResult(anchor, radius, patch_idx, context_idx, n_patch, n_context)

    return select


def _select_shell(dists, membership, max_count, N):
    """Select up to max_count particles from a distance-based shell.

    Uses sentinel-based padding: invalid entries are set to N (the system
    size) before sorting so they float to the end, then replaced with the
    last valid index for safe array gathering.

    Args:
        dists: Distances from anchor to all particles, shape (N,).
        membership: Boolean mask for shell membership, shape (N,).
        max_count: Maximum number of particles to select (static).
        N: Total system size.

    Returns:
        ``(indices, n_valid)``: indices has shape ``(max_count,)``, n_valid is a scalar.
    """
    n_in_shell = jnp.sum(membership)
    shell_dists = jnp.where(membership, dists, jnp.inf)
    nearest = jnp.argsort(shell_dists)[:max_count]

    valid_mask = jnp.arange(max_count) < n_in_shell
    candidates = jnp.where(valid_mask, nearest, N)
    sorted_candidates = jnp.sort(candidates)

    n_valid = jnp.minimum(n_in_shell, max_count)
    last_valid_idx = jnp.maximum(n_valid - 1, 0)
    last_valid = sorted_candidates[last_valid_idx]
    last_valid = jnp.where(n_valid > 0, last_valid, 0)
    indices = jnp.where(jnp.arange(max_count) < n_valid, sorted_candidates, last_valid)

    return indices, n_valid


def _log_cdf_diff(z_low, z_high):
    """Compute log(Φ(z_high) − Φ(z_low)) in a numerically stable way.

    Uses jax.scipy.special.log_ndtr for the log standard normal CDF and
    log1p-exp for the difference to avoid catastrophic cancellation.

    Args:
        z_low: Lower standardised bound, scalar.
        z_high: Upper standardised bound, scalar.

    Returns:
        log(Φ(z_high) − Φ(z_low)), scalar. Returns −inf when z_low ≥ z_high.
    """
    log_hi = jax.scipy.special.log_ndtr(z_high)
    log_lo = jax.scipy.special.log_ndtr(z_low)
    diff = log_lo - log_hi
    safe_diff = jnp.minimum(diff, 0.0)
    result = log_hi + jnp.log1p(-jnp.exp(safe_diff))
    return jnp.where(diff >= 0.0, -jnp.inf, result)


def soft_log_pi_ratio(dists_before, dists_after, patch_idx, context_idx, n_patch, n_context, r_mean, sigma_r):
    """Marginalised log π(p,c|y,x') / π(p,c|y,x) via the Gaussian CDF ratio.

    Treats the selection radius as drawn from r ~ N(r_mean, σ_r) and
    integrates over all r values consistent with the same partition (p, c).
    The context distances are unchanged (context positions are fixed), so
    the upper CDF bound is shared between forward and backward.

    When sigma_r = 0, falls back to a hard boundary check: returns 0.0 if
    the partition is still valid, −inf otherwise.

    Args:
        dists_before: Distances from anchor to all particles before the move, shape (N,).
        dists_after: Distances from anchor to all particles after the move, shape (N,).
        patch_idx: Patch indices, shape (N_patch,).
        context_idx: Context indices, shape (N_context,).
        n_patch: Number of valid patch particles, scalar.
        n_context: Number of valid context particles, scalar.
        r_mean: Mean radius of the selection protocol.
        sigma_r: Standard deviation of the radius distribution.

    Returns:
        Scalar log-ratio. Finite when the move is plausible, −inf when the
        proposed patch has escaped the selection shell entirely.
    """
    if sigma_r == 0.0:
        return _hard_log_ratio(dists_after, patch_idx, n_patch, r_mean)

    patch_d_before = jnp.where(jnp.arange(patch_idx.shape[0]) < n_patch, dists_before[patch_idx], -jnp.inf)
    r_low = jnp.max(patch_d_before)

    patch_d_after = jnp.where(jnp.arange(patch_idx.shape[0]) < n_patch, dists_after[patch_idx], -jnp.inf)
    r_low_new = jnp.max(patch_d_after)

    ctx_d = jnp.where(jnp.arange(context_idx.shape[0]) < n_context, dists_before[context_idx], jnp.inf)
    r_high = jnp.min(ctx_d)

    z_low = (r_low - r_mean) / sigma_r
    z_low_new = (r_low_new - r_mean) / sigma_r
    z_high = (r_high - r_mean) / sigma_r

    return _log_cdf_diff(z_low_new, z_high) - _log_cdf_diff(z_low, z_high)


def _hard_log_ratio(dists_after, patch_idx, n_patch, r_mean):
    """Hard check: log-ratio is 0 if all valid patch particles remain within r_mean, else −inf."""
    patch_d = jnp.where(jnp.arange(patch_idx.shape[0]) < n_patch, dists_after[patch_idx], -jnp.inf)
    all_in = jnp.max(patch_d) < r_mean
    return jnp.where(all_in, 0.0, -jnp.inf)
