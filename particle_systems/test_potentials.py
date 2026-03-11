"""Tests for the berni→jax-md potentials bridge and BoltzmannDistribution."""

import jax
import jax.numpy as jnp
import pytest

from particle_systems.particle_system import BoltzmannDistribution, ParticleSystem
from particle_systems.potentials import (
    _exact_derivatives,
    build_energy_fn,
    energy,
    inverse_power,
    make_cut_shift,
    make_linear_cut_shift,
    make_quadratic_cut_shift,
    make_smooth,
    yukawa,
)

# ── Pair functions ──────────────────────────────────────────────────────


def test_inverse_power_known_value():
    """ε(σ/r)^n with ε=1, σ=1, n=12, r=2 → (1/2)^12."""
    r = jnp.float32(2.0)
    u = inverse_power(r, sigma=1.0, epsilon=1.0, exponent=12)
    expected = (0.5) ** 12
    assert jnp.allclose(u, expected, atol=1e-6)


def test_inverse_power_scaling():
    """Energy should scale as ε and σ^n."""
    r = jnp.float32(1.5)
    u1 = inverse_power(r, sigma=1.0, epsilon=2.0, exponent=6)
    u2 = inverse_power(r, sigma=1.0, epsilon=1.0, exponent=6)
    assert jnp.allclose(u1, 2.0 * u2, atol=1e-6)


def test_yukawa_known_value():
    """ε·exp(-κ(r-σ))/r with ε=1, κ=1, σ=0, r=1 → exp(-1)."""
    r = jnp.float32(1.0)
    u = yukawa(r, sigma=0.0, epsilon=1.0, kappa=1.0)
    expected = jnp.exp(-1.0)
    assert jnp.allclose(u, expected, atol=1e-6)


# ── Exact derivatives ──────────────────────────────────────────────────


def test_exact_derivatives_lj():
    """Verify jax.grad derivatives match analytical LJ derivatives."""
    rcut = jnp.array([[2.5]], dtype=jnp.float32)
    params = {
        "sigma": jnp.array([[1.0]], dtype=jnp.float32),
        "epsilon": jnp.array([[1.0]], dtype=jnp.float32),
    }
    u0, du0, d2u0 = _exact_derivatives(energy.lennard_jones, rcut, params, 1)

    # Analytical: U''(r) = 4ε[156σ^12/r^14 - 42σ^6/r^8]
    d2u_analytical = 4 * (156 / 2.5**14 - 42 / 2.5**8)
    assert jnp.allclose(d2u0[0, 0], d2u_analytical, atol=1e-4)


# ── Cutoff wrappers ─────────────────────────────────────────────────────


def _get_corrections(raw_fn, rcut_val, pot_kw):
    """Helper: get (u0, du0, d2u0) for scalar args via exact derivs."""
    rcut = jnp.array([[rcut_val]], dtype=jnp.float32)
    params = {k: jnp.array([[v]], dtype=jnp.float32) for k, v in pot_kw.items()}
    u0, du0, d2u0 = _exact_derivatives(raw_fn, rcut, params, 1)
    return float(u0[0, 0]), float(du0[0, 0]), float(d2u0[0, 0])


def test_cut_shift_zero_at_cutoff():
    """U(rcut) should be exactly 0."""
    fn = make_cut_shift(inverse_power)
    rcut = jnp.float32(2.5)
    pot_kw = dict(sigma=1.0, epsilon=1.0, exponent=12)
    u0, _, _ = _get_corrections(inverse_power, 2.5, pot_kw)
    assert jnp.allclose(fn(rcut, rcut=rcut, u0=u0, **pot_kw), 0.0, atol=1e-7)


def test_cut_shift_zero_at_dr0():
    """Self-interaction (dr=0) should give exactly 0."""
    fn = make_cut_shift(inverse_power)
    pot_kw = dict(sigma=1.0, epsilon=1.0, exponent=12)
    u0, _, _ = _get_corrections(inverse_power, 2.5, pot_kw)
    assert fn(jnp.float32(0.0), rcut=2.5, u0=u0, **pot_kw) == 0.0


def test_cut_shift_analytical():
    """Verify cut_shift: U(r) - U(rc)."""
    fn = make_cut_shift(inverse_power)
    rcut, r = jnp.float32(2.5), jnp.float32(1.5)
    kw = dict(sigma=1.0, epsilon=1.0, exponent=12)
    u0, _, _ = _get_corrections(inverse_power, 2.5, kw)
    u_cs = fn(r, rcut=rcut, u0=u0, **kw)
    expected = (1.0 / 1.5) ** 12 - (1.0 / 2.5) ** 12
    assert jnp.allclose(u_cs, expected, atol=1e-5)


def test_linear_cut_shift_smooth():
    """Both U(rcut)=0 and dU/dr(rcut)≈0."""
    fn = make_linear_cut_shift(inverse_power)
    rcut = jnp.float32(2.5)
    kw = dict(sigma=1.0, epsilon=1.0, exponent=12)
    u0, du0, _ = _get_corrections(inverse_power, 2.5, kw)
    full_kw = dict(rcut=rcut, u0=u0, du0=du0, **kw)

    assert jnp.allclose(fn(rcut, **full_kw), 0.0, atol=1e-5)

    h = jnp.float32(1e-4)
    du = (fn(rcut - h, **full_kw) - fn(rcut, **full_kw)) / h
    assert jnp.abs(du) < 1e-2


def test_quadratic_cut_shift_smooth():
    """U, U', U'' all ≈0 at rcut."""
    fn = make_quadratic_cut_shift(inverse_power)
    rcut = jnp.float32(2.5)
    kw = dict(sigma=1.0, epsilon=1.0, exponent=12)
    u0, du0, d2u0 = _get_corrections(inverse_power, 2.5, kw)
    full_kw = dict(rcut=rcut, u0=u0, du0=du0, d2u0=d2u0, **kw)

    assert jnp.allclose(fn(rcut, **full_kw), 0.0, atol=1e-5)

    h = jnp.float32(1e-4)
    du = (fn(rcut - h, **full_kw) - fn(rcut, **full_kw)) / h
    assert jnp.abs(du) < 1e-2


def test_smooth_matches_julia_constants():
    """Smooth cutoff A, B, C should match Julia C0, C2, C4 for LJ at 2.5σ."""
    rcut = jnp.array([[2.5]], dtype=jnp.float32)
    params = {
        "sigma": jnp.array([[1.0]], dtype=jnp.float32),
        "epsilon": jnp.array([[1.0]], dtype=jnp.float32),
    }
    u0, du0, d2u0 = _exact_derivatives(energy.lennard_jones, rcut, params, 1)

    rc2, rc3, rc4 = rcut**2, rcut**3, rcut**4
    C = (du0 - rcut * d2u0) / (8 * rc3)
    B = -(d2u0 + 12 * rc2 * C) / 2
    A = -u0 - B * rc2 - C * rc4

    # Julia constants: C0=A/(4ε), C2=B/(4ε), C4=C/(4ε) for σ=1, ε=1
    eps4 = 4.0
    assert jnp.allclose(A[0, 0] / eps4, 0.04049023795, atol=1e-5)
    assert jnp.allclose(B[0, 0] / eps4, -0.00970155098, atol=1e-5)
    assert jnp.allclose(C[0, 0] / eps4, 0.00062012616, atol=1e-5)


def test_smooth_zero_at_cutoff():
    """Smooth cutoff U(rcut) = 0."""
    fn = make_smooth(energy.lennard_jones)
    rcut = jnp.float32(2.5)

    rcut_m = jnp.array([[rcut]], dtype=jnp.float32)
    params = {
        "sigma": jnp.array([[1.0]], dtype=jnp.float32),
        "epsilon": jnp.array([[1.0]], dtype=jnp.float32),
    }
    u0, du0, d2u0 = _exact_derivatives(energy.lennard_jones, rcut_m, params, 1)

    rc2, rc3, rc4 = rcut_m**2, rcut_m**3, rcut_m**4
    C_val = float(((du0 - rcut_m * d2u0) / (8 * rc3)).item())
    B_val = float((-(d2u0 + 12 * rc2 * C_val) / 2).item())
    A_val = float((-u0 - B_val * rc2 - C_val * rc4).item())

    result = fn(rcut, rcut=rcut, _A=A_val, _B=B_val, _C=C_val, sigma=1.0, epsilon=1.0)
    assert jnp.allclose(result, 0.0, atol=1e-5)


def test_smooth_zero_at_dr0():
    """Smooth cutoff at dr=0 (self-interaction) = 0."""
    fn = make_smooth(energy.lennard_jones)
    result = fn(jnp.float32(0.0), rcut=2.5, _A=0.1, _B=0.01, _C=0.001, sigma=1.0, epsilon=1.0)
    assert result == 0.0


# ── build_energy_fn ─────────────────────────────────────────────────────


HIWATARI_MODEL = {
    "potential": [
        {
            "type": "inverse_power",
            "parameters": {
                "exponent": 12,
                "sigma": [[1.0, 1.2], [1.2, 1.4]],
                "epsilon": [[1.0, 1.0], [1.0, 1.0]],
            },
        }
    ],
    "cutoff": [
        {
            "type": "cut_shift",
            "parameters": {"rcut": [[2.5, 2.5], [2.5, 2.5]]},
        }
    ],
}

ROY_MODEL = {
    "potential": [
        {
            "type": "yukawa",
            "parameters": {
                "kappa": [[0.1093, 0.1093], [0.1093, 0.1093]],
                "epsilon": [[0.16395, -0.1093], [-0.1093, 0.0728667]],
                "sigma": [[0.0, 0.0], [0.0, 0.0]],
            },
        },
        {
            "type": "inverse_power",
            "parameters": {
                "exponent": [[12, 12], [12, 12]],
                "epsilon": [[1.0, 1.0], [1.0, 1.0]],
                "sigma": [[2.250, 1.075], [1.075, 0.900]],
            },
        },
    ],
    "cutoff": [
        {
            "type": "linear_cut_shift",
            "parameters": {"rcut": [[9.1518, 9.1518], [9.1518, 9.1518]]},
        },
        {
            "type": "linear_cut_shift",
            "parameters": {"rcut": [[9.1518, 9.1518], [9.1518, 9.1518]]},
        },
    ],
}

JBB_MODEL = {
    "potential": [
        {
            "type": "lennard_jones",
            "parameters": {
                "sigma": [[1.0, 0.8, 0.9], [0.8, 0.88, 0.8], [0.9, 0.8, 0.94]],
                "epsilon": [[1.0, 1.5, 0.75], [1.5, 0.5, 1.5], [0.75, 1.5, 0.75]],
            },
        }
    ],
    "cutoff": [
        {
            "type": "smooth",
            "parameters": {
                "rcut": [[2.5, 2.0, 2.25], [2.0, 2.2, 2.0], [2.25, 2.0, 2.35]],
            },
        }
    ],
}


def test_build_energy_fn_single_potential():
    """Energy should be finite for a simple 2-species system."""
    L = 5.0
    box = jnp.array([L, L])
    energy_fn = build_energy_fn(HIWATARI_MODEL, box, n_species=2)

    key = jax.random.PRNGKey(42)
    positions = jax.random.uniform(key, (4, 2)) * L
    species = jnp.array([0, 0, 1, 1])

    U = energy_fn(positions, species)
    assert jnp.isfinite(U)
    assert U.shape == ()


def test_build_energy_fn_multi_potential():
    """Roy model (yukawa + inverse_power) should give finite energy."""
    L = 10.0
    box = jnp.array([L, L])
    energy_fn = build_energy_fn(ROY_MODEL, box, n_species=2)

    key = jax.random.PRNGKey(42)
    positions = jax.random.uniform(key, (4, 2)) * L
    species = jnp.array([0, 0, 1, 1])

    U = energy_fn(positions, species)
    assert jnp.isfinite(U)


def test_build_energy_fn_smooth_jbb():
    """JBB model with smooth cutoff should give finite energy."""
    L = 10.0
    box = jnp.array([L, L])
    energy_fn = build_energy_fn(JBB_MODEL, box, n_species=3)

    key = jax.random.PRNGKey(42)
    positions = jax.random.uniform(key, (44, 2)) * L
    species = jnp.array([0] * 20 + [1] * 12 + [2] * 12)

    U = energy_fn(positions, species)
    assert jnp.isfinite(U)


def test_energy_fn_jit_compatible():
    """Energy function should work under jax.jit."""
    L = 5.0
    box = jnp.array([L, L])
    energy_fn = build_energy_fn(HIWATARI_MODEL, box, n_species=2)

    key = jax.random.PRNGKey(42)
    positions = jax.random.uniform(key, (4, 2)) * L
    species = jnp.array([0, 0, 1, 1])

    U_jit = jax.jit(energy_fn)(positions, species)
    U_raw = energy_fn(positions, species)
    assert jnp.allclose(U_jit, U_raw, atol=1e-5)


def test_energy_fn_grad_compatible():
    """Forces via jax.grad should be finite."""
    L = 5.0
    box = jnp.array([L, L])
    energy_fn = build_energy_fn(HIWATARI_MODEL, box, n_species=2)

    key = jax.random.PRNGKey(42)
    positions = jax.random.uniform(key, (4, 2)) * L
    species = jnp.array([0, 0, 1, 1])

    forces = jax.grad(energy_fn)(positions, species)
    assert forces.shape == positions.shape
    assert jnp.all(jnp.isfinite(forces))


def test_unknown_potential_raises():
    """Unknown potential type should raise ValueError."""
    model = {
        "potential": [{"type": "nonexistent", "parameters": {}}],
        "cutoff": [{"type": "cut_shift", "parameters": {"rcut": 2.5}}],
    }
    with pytest.raises(ValueError, match="Unknown potential"):
        build_energy_fn(model, jnp.array([5.0, 5.0]), n_species=1)


def test_unknown_cutoff_raises():
    """Unknown cutoff type should raise ValueError."""
    model = {
        "potential": [
            {
                "type": "inverse_power",
                "parameters": {"exponent": 12, "sigma": 1.0, "epsilon": 1.0},
            }
        ],
        "cutoff": [{"type": "nonexistent", "parameters": {"rcut": 2.5}}],
    }
    with pytest.raises(ValueError, match="Unknown cutoff"):
        build_energy_fn(model, jnp.array([5.0, 5.0]), n_species=1)


# ── BoltzmannDistribution ──────────────────────────────────────────────


@pytest.fixture
def boltzmann_setup():
    """BoltzmannDistribution for the Hiwatari model."""
    return BoltzmannDistribution(
        N=4,
        d=2,
        L=5.0,
        temperature=1.0,
        model=HIWATARI_MODEL,
        composition=(0.5, 0.5),
    )


def test_boltzmann_log_prob(boltzmann_setup):
    """log_prob should return -U/T, finite scalar."""
    dist = boltzmann_setup
    ps = ParticleSystem(
        positions=jax.random.uniform(jax.random.PRNGKey(42), (4, 2)) * 5.0,
        species=jnp.array([0, 0, 1, 1]),
        box=jnp.array([5.0, 5.0]),
    )
    lp = dist.log_prob(ps)
    assert jnp.isfinite(lp)
    assert lp.shape == ()


def test_boltzmann_log_prob_is_minus_energy_over_T():
    """Verify log_prob = -U/T by comparing with direct energy_fn call."""
    T = 2.0
    dist = BoltzmannDistribution(
        N=4,
        d=2,
        L=5.0,
        temperature=T,
        model=HIWATARI_MODEL,
        composition=(0.5, 0.5),
    )
    energy_fn = build_energy_fn(HIWATARI_MODEL, jnp.array([5.0, 5.0]), n_species=2)

    ps = ParticleSystem(
        positions=jax.random.uniform(jax.random.PRNGKey(123), (4, 2)) * 5.0,
        species=jnp.array([0, 0, 1, 1]),
        box=jnp.array([5.0, 5.0]),
    )
    lp = dist.log_prob(ps)
    U = energy_fn(ps.positions, ps.species)
    assert jnp.allclose(lp, -U / T, atol=1e-6)


def test_boltzmann_unsampleable(boltzmann_setup):
    """Sampling should raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        boltzmann_setup.sample(seed=jax.random.PRNGKey(0))


def test_boltzmann_event_shape(boltzmann_setup):
    """event_shape should match ParticleSystem structure."""
    es = boltzmann_setup.event_shape
    assert es.positions == (4, 2)
    assert es.species == (4,)
    assert es.box == (2,)


def test_boltzmann_batched(boltzmann_setup):
    """log_prob should handle batched samples directly (distrax convention)."""
    dist = boltzmann_setup
    M = 8
    positions = jax.random.uniform(jax.random.PRNGKey(42), (M, 4, 2)) * 5.0
    species = jnp.tile(jnp.array([0, 0, 1, 1]), (M, 1))
    boxes = jnp.tile(jnp.array([5.0, 5.0]), (M, 1))

    batch = ParticleSystem(positions=positions, species=species, box=boxes)
    log_probs = dist.log_prob(batch)
    assert log_probs.shape == (M,)
    assert jnp.all(jnp.isfinite(log_probs))

    # Batched should match single calls
    single_lps = jnp.stack(
        [dist.log_prob(ParticleSystem(positions=positions[i], species=species[i], box=boxes[i])) for i in range(M)]
    )
    assert jnp.allclose(log_probs, single_lps, atol=1e-5)
