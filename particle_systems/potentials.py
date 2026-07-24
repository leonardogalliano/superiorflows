"""Bridge between berni model dictionaries and jax-md energy functions.

Provides custom pair potentials (inverse_power, yukawa), cutoff wrappers
(cut_shift, linear_cut_shift, quadratic_cut_shift, smooth), and a
``build_energy_fn`` entry point that parses a berni model dict into a
jax-md energy callable.

Typical usage::

    from particle_systems.potentials import build_energy_fn

    model = json.load(open("model.json"))
    energy_fn = build_energy_fn(model, box=jnp.array([L, L]), n_species=2)
    U = energy_fn(positions, species=species)
"""

import warnings
from functools import wraps

import jax
import jax.numpy as jnp
from jax_md import energy, smap, space

# ── Pair energy functions (dr → energy, jax-md convention) ──────────────


def inverse_power(dr, sigma=1.0, epsilon=1.0, exponent=12, **_):
    """Inverse power law: ε(σ/r)^n.  Safe at dr=0."""
    safe_dr = jnp.where(dr > 0, dr, 1.0)  # avoid 0/0
    u = epsilon * (sigma / safe_dr) ** exponent
    return jnp.where(dr > 0, u, 0.0)


def yukawa(dr, sigma=0.0, epsilon=1.0, kappa=1.0, **_):
    """Yukawa / screened Coulomb: ε·exp(-κ(r-σ)) / r.  Safe at dr=0."""
    safe_dr = jnp.where(dr > 0, dr, 1.0)
    u = epsilon * jnp.exp(-kappa * (safe_dr - sigma)) / safe_dr
    return jnp.where(dr > 0, u, 0.0)


# ── Cutoff wrappers ─────────────────────────────────────────────────────
#
# Each wrapper takes a raw pair function and returns a new pair function
# that enforces the cutoff scheme.  Pre-computed correction constants
# (u0, du0, d2u0, or A/B/C) are passed as species-indexed kwargs via
# smap.pair — computed exactly with jax.grad at build time.


def make_cut_shift(fn):
    """Truncate and shift so U(rcut)=0.  Zero for r >= rcut.

    Extra kwargs: ``rcut``, ``u0`` (= U(rcut)).
    """

    @wraps(fn)
    def wrapped(dr, rcut=2.5, u0=0.0, **kwargs):
        u = fn(dr, **kwargs)
        result = jnp.where(dr < rcut, u - u0, 0.0)
        return jnp.where(dr > 0, result, 0.0)

    return wrapped


def make_linear_cut_shift(fn):
    """Shifted-force cutoff: U(rcut) = U'(rcut) = 0.

    Extra kwargs: ``rcut``, ``u0``, ``du0`` (= U'(rcut)).
    """

    @wraps(fn)
    def wrapped(dr, rcut=2.5, u0=0.0, du0=0.0, **kwargs):
        u = fn(dr, **kwargs)
        delta = dr - rcut
        result = jnp.where(dr < rcut, u - u0 - du0 * delta, 0.0)
        return jnp.where(dr > 0, result, 0.0)

    return wrapped


def make_quadratic_cut_shift(fn):
    """Taylor-based cutoff: U, U', U'' all vanish at rcut.

    Correction is a 2nd-order Taylor expansion in (r - rcut).
    Extra kwargs: ``rcut``, ``u0``, ``du0``, ``d2u0`` (= U''(rcut)).
    """

    @wraps(fn)
    def wrapped(dr, rcut=2.5, u0=0.0, du0=0.0, d2u0=0.0, **kwargs):
        u = fn(dr, **kwargs)
        delta = dr - rcut
        result = jnp.where(
            dr < rcut,
            u - u0 - du0 * delta - 0.5 * d2u0 * delta**2,
            0.0,
        )
        return jnp.where(dr > 0, result, 0.0)

    return wrapped


def make_smooth(fn):
    """Polynomial-in-r² cutoff: U, U', U'' all vanish at rcut.

    Adds a correction ``A + B·r² + C·r⁴`` computed to make the
    potential and its first two derivatives vanish at rcut.  This is
    the scheme used in Pedersen et al. (JBB model) and similar.

    Unlike ``quadratic_cut_shift`` (Taylor in r - rcut), this uses
    even-power polynomials in r, which is common in classical MD
    because r² is cheaper to compute than r.

    Extra kwargs: ``rcut``, ``_A``, ``_B``, ``_C``.
    """

    @wraps(fn)
    def wrapped(dr, rcut=2.5, _A=0.0, _B=0.0, _C=0.0, **kwargs):
        u = fn(dr, **kwargs)
        r2 = dr * dr
        result = jnp.where(dr < rcut, u + _A + _B * r2 + _C * r2 * r2, 0.0)
        return jnp.where(dr > 0, result, 0.0)

    return wrapped


# ── Exact derivative helpers (build time only) ──────────────────────────


def _exact_derivatives(raw_fn, rcut, params, n_species):
    """Compute U(rcut), U'(rcut), U''(rcut) per species pair using jax.grad.

    Args:
        raw_fn: Raw pair function (dr → energy).
        rcut: Cutoff matrix, shape ``(n_species, n_species)``.
        params: Dict of parameter matrices, each ``(n_species, n_species)``.
        n_species: Number of species types.

    Returns:
        (u0, du0, d2u0) — each of shape ``(n_species, n_species)``.
    """
    u0 = jnp.zeros_like(rcut)
    du0 = jnp.zeros_like(rcut)
    d2u0 = jnp.zeros_like(rcut)

    for i in range(n_species):
        for j in range(n_species):
            kw = {k: v[i, j] for k, v in params.items()}
            r = rcut[i, j]

            def _u(r_, **kw_):
                return raw_fn(r_, **kw_)

            u0 = u0.at[i, j].set(_u(r, **kw))
            du0 = du0.at[i, j].set(jax.grad(_u)(r, **kw))
            d2u0 = d2u0.at[i, j].set(jax.grad(jax.grad(_u))(r, **kw))

    return u0, du0, d2u0


# ── Registries ──────────────────────────────────────────────────────────

POTENTIAL_REGISTRY = {
    "inverse_power": inverse_power,
    "yukawa": yukawa,
    "lennard_jones": energy.lennard_jones,
}

CUTOFF_REGISTRY = {
    "cut_shift": make_cut_shift,
    "linear_cut_shift": make_linear_cut_shift,
    "quadratic_cut_shift": make_quadratic_cut_shift,
    "smooth": make_smooth,
}


# ── Builder ─────────────────────────────────────────────────────────────


def _build_pair_terms(model, n_species):
    """Parse a model dict and build (pair_fn, params) for each potential term.

    Shared setup used by both :func:`build_energy_fn` and
    :func:`build_masked_energy_fn`.  Does NOT construct displacement or
    metric functions — those depend on the summation strategy.

    Args:
        model: A berni model dict with ``"potential"`` and ``"cutoff"`` keys.
        n_species: Number of distinct species types.

    Returns:
        List of ``(pair_fn, all_params)`` tuples where ``pair_fn`` is the
        cutoff-wrapped pair potential and ``all_params`` is a dict of
        ``(n_species, n_species)`` parameter matrices.
    """
    potentials = model["potential"]
    cutoffs = model["cutoff"]
    if len(potentials) != len(cutoffs):
        raise ValueError(f"Number of potentials ({len(potentials)}) must match number of cutoffs ({len(cutoffs)})")

    terms = []
    for pot_spec, cut_spec in zip(potentials, cutoffs):
        pot_type = pot_spec["type"]
        cut_type = cut_spec["type"]

        if pot_type not in POTENTIAL_REGISTRY:
            raise ValueError(f"Unknown potential '{pot_type}'. Available: {list(POTENTIAL_REGISTRY)}")
        if cut_type not in CUTOFF_REGISTRY:
            raise ValueError(f"Unknown cutoff '{cut_type}'. Available: {list(CUTOFF_REGISTRY)}")

        raw_fn = POTENTIAL_REGISTRY[pot_type]
        cutoff_wrapper = CUTOFF_REGISTRY[cut_type]
        pair_fn = cutoff_wrapper(raw_fn)

        # Collect potential parameters, broadcasting scalars to matrices
        params = {}
        for k, v in pot_spec["parameters"].items():
            v = jnp.asarray(v, dtype=jnp.float32)
            if v.ndim == 0:
                v = jnp.full((n_species, n_species), v)
            params[k] = v

        rcut = jnp.asarray(cut_spec["parameters"]["rcut"], dtype=jnp.float32)
        if rcut.ndim == 0:
            rcut = jnp.full((n_species, n_species), rcut)

        # Pre-compute correction constants using exact derivatives
        correction_params = {"rcut": rcut}

        needs_derivs = cut_type in (
            "linear_cut_shift",
            "quadratic_cut_shift",
            "smooth",
        )
        if cut_type == "cut_shift":
            u0 = raw_fn(rcut, **params)
            correction_params["u0"] = u0
        elif needs_derivs:
            u0, du0, d2u0 = _exact_derivatives(raw_fn, rcut, params, n_species)
            if cut_type == "linear_cut_shift":
                correction_params["u0"] = u0
                correction_params["du0"] = du0
            elif cut_type == "quadratic_cut_shift":
                correction_params["u0"] = u0
                correction_params["du0"] = du0
                correction_params["d2u0"] = d2u0
            elif cut_type == "smooth":
                # Solve for A, B, C of correction polynomial A + B·r² + C·r⁴
                rc2 = rcut * rcut
                rc3 = rc2 * rcut
                rc4 = rc2 * rc2
                C = (du0 - rcut * d2u0) / (8 * rc3)
                B = -(d2u0 + 12 * rc2 * C) / 2
                A = -u0 - B * rc2 - C * rc4
                correction_params["_A"] = A
                correction_params["_B"] = B
                correction_params["_C"] = C

        # Merge all params
        all_params = {**params, **correction_params}
        terms.append((pair_fn, all_params))

    return terms


def build_energy_fn(model, box, n_species):
    """Build a jax-md energy function from a berni model dictionary.

    Args:
        model: A berni model dict with ``"potential"`` and ``"cutoff"`` keys.
            Each potential and cutoff entry is paired by index.
        box: Box side lengths, array of shape ``(d,)`` or scalar.
        n_species: Number of distinct species types.

    Returns:
        A function ``energy_fn(positions, species) -> scalar``
        computing the total potential energy.
    """
    displacement_fn, _ = space.periodic(box)
    metric_fn = space.metric(displacement_fn)
    terms = _build_pair_terms(model, n_species)

    # Build one smap.pair per potential term
    term_fns = []
    for pair_fn, all_params in terms:
        energy_term = smap.pair(
            pair_fn,
            metric_fn,
            species=n_species,
            ignore_unused_parameters=True,
            **all_params,
        )
        term_fns.append(energy_term)

    def energy_fn(positions, species):
        """Total potential energy.

        Args:
            positions: Particle positions, shape ``(N, d)``.
            species: Integer species labels, shape ``(N,)``.

        Returns:
            Scalar total energy.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*float64.*", category=UserWarning)
            return sum(fn(positions, species=species) for fn in term_fns)

    return energy_fn


def build_masked_energy_fn(model, box, n_species):
    """Build a masked energy function for subsystems with padded particles.

    Like :func:`build_energy_fn`, but returns a function with signature
    ``energy_fn(positions, species, n_valid) → scalar`` that excludes
    pairs involving padded particles (index ≥ n_valid) from the sum.

    Uses manual O(M²) pairwise computation with vectorised species-indexed
    parameter lookup, rather than ``smap.pair``.

    Args:
        model: A berni model dict.
        box: Box side lengths, array of shape ``(d,)``.
        n_species: Number of distinct species types.

    Returns:
        A function ``energy_fn(positions, species, n_valid) → scalar``.
    """
    displacement_fn, _ = space.periodic(box)
    metric_fn = space.metric(displacement_fn)
    d_fn = space.map_product(metric_fn)
    terms = _build_pair_terms(model, n_species)

    def energy_fn(positions, species, n_valid):
        M = positions.shape[0]
        dr = d_fn(positions, positions)

        valid = (jnp.arange(M) < n_valid).astype(positions.dtype)
        pair_mask = valid[:, None] * valid[None, :]
        diag_mask = 1.0 - jnp.eye(M)
        mask = pair_mask * diag_mask

        total = jnp.float32(0.0)
        for pair_fn, params in terms:
            kw = {k: v[species[:, None], species[None, :]] for k, v in params.items()}
            u = pair_fn(dr, **kw)
            total = total + jnp.sum(u * mask)
        return 0.5 * total

    return energy_fn


def build_masked_local_energy_fn(model, box, n_species):
    """Build a masked local energy function for subsystems with patch and context.

    Calculates U_loc(xp, xc) = U_p(xp) + U_pc(xp, xc), excluding context-context
    interactions U_c(xc) and padded particles (index >= n_valid).

    Args:
        model: A berni model dict.
        box: Box side lengths, array of shape ``(d,)``.
        n_species: Number of distinct species types.

    Returns:
        A function ``energy_fn(positions, species, n_patch, n_valid) -> scalar``.
    """
    displacement_fn, _ = space.periodic(box)
    metric_fn = space.metric(displacement_fn)
    d_fn = space.map_product(metric_fn)
    terms = _build_pair_terms(model, n_species)

    def energy_fn(positions, species, n_patch, n_valid):
        M = positions.shape[0]
        dr = d_fn(positions, positions)

        valid = (jnp.arange(M) < n_valid).astype(positions.dtype)
        pair_mask = valid[:, None] * valid[None, :]
        diag_mask = 1.0 - jnp.eye(M)

        in_patch = jnp.arange(M) < n_patch
        patch_mask = (in_patch[:, None] | in_patch[None, :]).astype(positions.dtype)

        mask = pair_mask * diag_mask * patch_mask

        total = jnp.float32(0.0)
        for pair_fn, params in terms:
            kw = {k: v[species[:, None], species[None, :]] for k, v in params.items()}
            u = pair_fn(dr, **kw)
            total = total + jnp.sum(u * mask)
        return 0.5 * total

    return energy_fn
