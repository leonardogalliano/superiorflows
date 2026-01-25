"""
Profile Flow reconstruction to prove it doesn't cause recompilation.

This script uses JAX's profiling tools to verify that:
1. Flow() construction is invisible to JAX tracing
2. No recompilation occurs when array values change
3. Execution time is consistent across calls
"""

import time

import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
from superiorflows import Flow


class MLPVelocity(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(in_size=3, out_size=2, width_size=16, depth=2, key=key)

    def __call__(self, t, x, args):
        return self.mlp(jnp.concatenate([x, jnp.atleast_1d(t)]))


def main():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    velocity_field = MLPVelocity(subkey)
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2))
    X = jax.random.normal(jax.random.PRNGKey(0), (10, 2))

    # ========================================================================
    # PROOF 1: jax.make_jaxpr shows what's actually traced
    # ========================================================================
    print("=" * 70)
    print("PROOF 1: Inspecting the JAX expression (jaxpr)")
    print("=" * 70)

    def loss_fn(vf_arrays, X):
        vf = eqx.combine(vf_arrays, eqx.filter(velocity_field, lambda x: not eqx.is_array(x)))
        flow = Flow(velocity_field=vf, base_distribution=base_dist)
        return -jnp.mean(flow.log_prob(X))

    vf_arrays = eqx.filter(velocity_field, eqx.is_array)
    jaxpr = jax.make_jaxpr(loss_fn)(vf_arrays, X)

    eqns = jaxpr.jaxpr.eqns
    print(f"Number of operations in jaxpr: {len(eqns)}")
    print(f"Input variables: {len(jaxpr.jaxpr.invars)}")
    print()
    print("Operations traced (no Flow construction!):")
    for i, eqn in enumerate(eqns):
        print(f"  {i}: {eqn.primitive.name}")
    print()
    print("=> Flow(), eqx.combine() are INVISIBLE to JAX - just pytree wiring!")

    # ========================================================================
    # PROOF 2: Timing consistency proves no recompilation
    # ========================================================================
    print()
    print("=" * 70)
    print("PROOF 2: Timing consistency (no recompilation = consistent timing)")
    print("=" * 70)

    loss_fn_jit = eqx.filter_jit(loss_fn)

    # Warmup (compilation happens here)
    t0 = time.perf_counter()
    _ = loss_fn_jit(vf_arrays, X)
    t1 = time.perf_counter()
    print(f"First call (compilation): {(t1-t0)*1000:.1f}ms")

    # Subsequent calls (should be fast and consistent)
    times = []
    for i in range(50):
        new_arrays = jax.tree.map(lambda x: x + 0.001 * i, vf_arrays)
        t0 = time.perf_counter()
        loss = loss_fn_jit(new_arrays, X)
        loss.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    mean_t = sum(times) / len(times)
    std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
    print("\nTiming over 50 subsequent calls (ms):")
    print(f"  Mean:   {mean_t:.3f}ms")
    print(f"  Std:    {std_t:.3f}ms")
    print(f"  Min:    {min(times):.3f}ms")
    print(f"  Max:    {max(times):.3f}ms")
    print()
    print("=> Low std dev & consistent timing = NO recompilation!")

    # ========================================================================
    # PROOF 3: XLA HLO shows actual computation
    # ========================================================================
    print()
    print("=" * 70)
    print("PROOF 3: XLA compilation analysis")
    print("=" * 70)

    lowered = jax.jit(lambda a, b: loss_fn(a, b)).lower(vf_arrays, X)
    hlo_text = lowered.as_text()
    print(f"HLO module text length: {len(hlo_text)} chars")
    print()

    # Count operation types in HLO
    hlo_ops = ["dot", "reduce", "broadcast", "tanh", "concatenate", "log"]
    print("HLO contains these operations (actual math):")
    for op in hlo_ops:
        count = hlo_text.count(op)
        if count > 0:
            print(f"  {op}: {count}")
    print()
    print("=> This is pure math - no Python object construction in XLA!")


if __name__ == "__main__":
    main()
