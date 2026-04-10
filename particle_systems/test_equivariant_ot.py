"""Verification of EquivariantOptimalTransport with hyperoctahedral group symmetry.

Tests:
1. Group generation: correct size, orthogonality, closure
2. Box symmetry: maps [0,L)^d -> [0,L)^d
3. OT cost reduction: box symmetry never increases cost
4. Exhaustive optimality: returned cost matches best over all group elements
5. Backward compatibility: use_box_symmetry=False gives same result as before
"""

import sys

import numpy as np

sys.path.insert(0, "/Users/Leonardo/Documents/Postdoc/Projects/superiorflows")
sys.path.insert(0, "/Users/Leonardo/Documents/Postdoc/Projects/superiorflows/particle_systems")

from particle_systems.particle_system import (  # noqa: E402
    EquivariantOptimalTransport,
    ParticleSystem,
    _solve_ot_single,
    apply_box_symmetry,
    generate_hyperoctahedral_group,
)


def test_group_size():
    for d, expected in [(2, 8), (3, 48)]:
        G = generate_hyperoctahedral_group(d)
        assert G.shape == (expected, d, d), f"d={d}: expected {expected}, got {G.shape[0]}"
    print("  [PASS] Group sizes: 8 (2D), 48 (3D)")


def test_orthogonality():
    for d in [2, 3]:
        G = generate_hyperoctahedral_group(d)
        for i, g in enumerate(G):
            prod = g.T @ g
            assert np.allclose(prod, np.eye(d)), f"d={d}, g[{i}] not orthogonal"
            assert np.isclose(abs(np.linalg.det(g)), 1.0), f"d={d}, g[{i}] det != ±1"
    print("  [PASS] All group elements are orthogonal with det ±1")


def test_closure():
    for d in [2, 3]:
        G = generate_hyperoctahedral_group(d)
        # Represent each matrix as a tuple of tuples for set membership
        G_set = set()
        for g in G:
            G_set.add(tuple(map(tuple, np.round(g).astype(int))))

        for i, gi in enumerate(G):
            for j, gj in enumerate(G):
                prod = gi @ gj
                prod_key = tuple(map(tuple, np.round(prod).astype(int)))
                assert prod_key in G_set, f"d={d}: g[{i}] @ g[{j}] not in group"
    print("  [PASS] Group closure verified for 2D and 3D")


def test_identity_in_group():
    for d in [2, 3]:
        G = generate_hyperoctahedral_group(d)
        found = any(np.allclose(g, np.eye(d)) for g in G)
        assert found, f"d={d}: identity not found in group"
    print("  [PASS] Identity element present in 2D and 3D")


def test_box_symmetry_range():
    rng = np.random.default_rng(42)
    for d in [2, 3]:
        L = np.full(d, 7.3)
        G = generate_hyperoctahedral_group(d)
        positions = rng.uniform(0, L[0], size=(50, d))
        for i, g in enumerate(G):
            transformed = apply_box_symmetry(positions, g, L)
            assert np.all(transformed >= 0) and np.all(transformed < L[0]), f"d={d}, g[{i}]: positions out of [0, L)"
    print("  [PASS] apply_box_symmetry maps into [0, L)^d for all group elements")


def test_ot_cost_reduction():
    """Box symmetry OT cost should be <= permutation-only OT cost."""
    rng = np.random.default_rng(123)
    L = np.array([5.0, 5.0])
    N = 20

    n_tests = 50
    improvements = 0

    for _ in range(n_tests):
        pos_0 = rng.uniform(0, L[0], (N, 2))
        pos_1 = rng.uniform(0, L[0], (N, 2))
        species = np.array([0] * 10 + [1] * 10)

        # Permutation-only
        _, cost_perm = _solve_ot_single(pos_0, pos_1, species, L)

        # Best over hyperoctahedral group
        G = generate_hyperoctahedral_group(2)
        best_cost = np.inf
        for g in G:
            p0_g = apply_box_symmetry(pos_0, g, L)
            _, cost_g = _solve_ot_single(p0_g, pos_1, species, L)
            best_cost = min(best_cost, cost_g)

        assert best_cost <= cost_perm + 1e-10, f"Box symmetry cost {best_cost:.4f} > perm-only cost {cost_perm:.4f}"
        if best_cost < cost_perm - 1e-10:
            improvements += 1

    print(
        f"  [PASS] Box symmetry cost <= perm-only cost in all {n_tests} tests "
        f"({improvements}/{n_tests} strict improvements)"
    )


def test_exhaustive_optimality():
    """The EquivariantOptimalTransport class should find the global minimum."""
    rng = np.random.default_rng(777)
    L = np.array([6.0, 6.0])
    N = 15
    B = 4

    pos_0 = rng.uniform(0, L[0], (B, N, 2)).astype(np.float32)
    pos_1 = rng.uniform(0, L[0], (B, N, 2)).astype(np.float32)
    species = np.tile(np.array([0] * 8 + [1] * 7, dtype=np.int32), (B, 1))
    box = np.tile(L.astype(np.float32), (B, 1))

    x0 = ParticleSystem(positions=pos_0, species=species, box=box)
    x1 = ParticleSystem(positions=pos_1, species=species, box=box)

    # Run with box symmetry
    ot_sym = EquivariantOptimalTransport(use_box_symmetry=True)
    (x0_aligned, _) = ot_sym((x0, x1))

    pos_aligned = np.asarray(x0_aligned.positions)

    # Verify: for each batch element, compute cost and compare to exhaustive search
    G = generate_hyperoctahedral_group(2)
    for i in range(B):
        # Cost from the class
        disp = pos_aligned[i] - pos_1[i]
        disp -= L * np.round(disp / L)
        class_cost = np.sum(disp**2)

        # Exhaustive best
        best_cost = np.inf
        for g in G:
            p0_g = apply_box_symmetry(pos_0[i], g, L)
            _, cost_g = _solve_ot_single(p0_g, pos_1[i], species[i], L)
            best_cost = min(best_cost, cost_g)

        assert np.isclose(
            class_cost, best_cost, rtol=1e-4
        ), f"Batch {i}: class cost {class_cost:.4f} != exhaustive best {best_cost:.4f}"

    print(f"  [PASS] EquivariantOptimalTransport finds exhaustive optimum for B={B}")


def test_backward_compatibility():
    """use_box_symmetry=False should give identical results to the identity-only path."""
    rng = np.random.default_rng(55)
    L = np.array([5.0, 5.0])
    N = 12
    B = 3

    pos_0 = rng.uniform(0, L[0], (B, N, 2)).astype(np.float32)
    pos_1 = rng.uniform(0, L[0], (B, N, 2)).astype(np.float32)
    species = np.tile(np.array([0] * 6 + [1] * 6, dtype=np.int32), (B, 1))
    box = np.tile(L.astype(np.float32), (B, 1))

    x0 = ParticleSystem(positions=pos_0, species=species, box=box)
    x1 = ParticleSystem(positions=pos_1, species=species, box=box)

    ot_no_sym = EquivariantOptimalTransport(use_box_symmetry=False)
    (x0_a, _) = ot_no_sym((x0, x1))

    # Manually run identity-only
    for i in range(B):
        aligned_i, _ = _solve_ot_single(pos_0[i], pos_1[i], species[i], L)
        assert np.allclose(np.asarray(x0_a.positions[i]), aligned_i, atol=1e-6), f"Batch {i}: backward compat mismatch"

    print("  [PASS] use_box_symmetry=False matches manual identity-only OT")


def test_species_preserved():
    """After OT, species of aligned x0 must match species of x1."""
    rng = np.random.default_rng(99)
    L = np.array([5.0, 5.0])
    N = 20
    B = 5

    pos_0 = rng.uniform(0, L[0], (B, N, 2)).astype(np.float32)
    pos_1 = rng.uniform(0, L[0], (B, N, 2)).astype(np.float32)
    species = np.tile(np.array([0] * 10 + [1] * 10, dtype=np.int32), (B, 1))
    box = np.tile(L.astype(np.float32), (B, 1))

    x0 = ParticleSystem(positions=pos_0, species=species, box=box)
    x1 = ParticleSystem(positions=pos_1, species=species, box=box)

    for sym in [False, True]:
        ot = EquivariantOptimalTransport(use_box_symmetry=sym)
        (x0_a, _) = ot((x0, x1))
        assert np.array_equal(
            np.asarray(x0_a.species), np.asarray(x1.species)
        ), f"Species mismatch with use_box_symmetry={sym}"

    print("  [PASS] Species preserved after OT (both modes)")


if __name__ == "__main__":
    print("=" * 60)
    print("Verification: EquivariantOptimalTransport")
    print("=" * 60)

    print("\n1. Group generation")
    test_group_size()
    test_orthogonality()
    test_identity_in_group()

    print("\n2. Group closure")
    test_closure()

    print("\n3. Box symmetry range")
    test_box_symmetry_range()

    print("\n4. OT cost reduction")
    test_ot_cost_reduction()

    print("\n5. Exhaustive optimality")
    test_exhaustive_optimality()

    print("\n6. Backward compatibility")
    test_backward_compatibility()

    print("\n7. Species preservation")
    test_species_preserved()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
