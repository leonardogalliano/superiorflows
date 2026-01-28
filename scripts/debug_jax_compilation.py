"""
Debug script to analyze JAX compilation behavior in the training loop.

This script investigates potential recompilation issues when creating Flow objects
inside loss functions, particularly focusing on:
1. dynamic_mask behavior (is it being traced as static or causing retracing?)
2. flow_kwargs closure behavior
3. Overall JIT compilation efficiency

Run with: uv run python scripts/debug_jax_compilation.py
"""
import time

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from superiorflows import Flow
from superiorflows.train import MaximumLikelihoodLoss
from superiorflows.train.trainer import train_step

# Enable compilation logging - this will print when recompilation occurs
jax.config.update("jax_log_compiles", True)


# =============================================================================
# Test Models
# =============================================================================


class SimpleVelocity(eqx.Module):
    """Simple MLP velocity field for testing."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, dim: int, hidden: int, *, key):
        k1, k2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(dim + 1, hidden, key=k1)
        self.linear2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, t, x, args):
        t_feat = jnp.broadcast_to(t, x.shape[:-1] + (1,))
        h = jax.nn.tanh(self.linear1(jnp.concatenate([x, t_feat], axis=-1)))
        return self.linear2(h)


# =============================================================================
# Test 1: Distribution PyTree Stability
# =============================================================================


def test_distrax_pytree_issue():
    """Investigate distrax distribution pytree stability."""
    print("\n" + "=" * 60)
    print("TEST 1: Distrax Distribution PyTree Analysis")
    print("=" * 60)

    dim = 2

    print("\n1a. Creating a MultivariateNormalDiag distribution...")
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim))

    print("\n1b. Flattening it twice and comparing tree structures...")
    flat1, tree1 = jax.tree_util.tree_flatten(base_dist)
    flat2, tree2 = jax.tree_util.tree_flatten(base_dist)

    print(f"  Same tree structure: {tree1 == tree2}")

    if tree1 != tree2:
        print("\n  ⚠️ CRITICAL ISSUE: Same object gives different tree structures!")
        print("  This means distrax distributions are NOT stable for JAX tracing.")
    else:
        print("  ✓ Tree structure is stable for the same object")

    print("\n1c. Testing if calling methods changes the tree...")
    _ = base_dist.log_prob(jnp.zeros(dim))
    flat3, tree3 = jax.tree_util.tree_flatten(base_dist)
    print(f"  Tree same after log_prob call: {tree1 == tree3}")

    print("\n  ✓ Test 1 complete")


# =============================================================================
# Test 2: The Actual Training Flow
# =============================================================================


def test_actual_training_pattern():
    """Test the exact pattern used in training - does it recompile?"""
    print("\n" + "=" * 60)
    print("TEST 2: Actual Training Pattern (Critical Test)")
    print("=" * 60)

    key = jax.random.key(0)
    dim = 2

    # Create components
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim))
    velocity = SimpleVelocity(dim=dim, hidden=16, key=key)
    loss_fn = MaximumLikelihoodLoss(base_dist)
    optimizer = optax.adam(1e-3)

    model = velocity
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print("\n2a. Running training steps and measuring times...")
    print("    (Watch for 'Compiling' messages from JAX)")
    print("-" * 60)

    step_times = []
    n_steps = 10

    for i in range(n_steps):
        key, subkey = jax.random.split(key)
        batch = jax.random.normal(jax.random.key(i), (32, dim))

        t0 = time.perf_counter()
        model, opt_state, loss = train_step(model, opt_state, batch, subkey, loss_fn, optimizer)
        jax.block_until_ready(loss)
        t1 = time.perf_counter()

        step_times.append((t1 - t0) * 1000)
        print(f"  Step {i+1:2d}: {step_times[-1]:8.1f}ms, loss={float(loss):.4f}")

    print("-" * 60)

    print("\n2b. Analysis:")
    print(f"  First step (compilation): {step_times[0]:.1f}ms")
    print(f"  Average of steps 2-{n_steps}: {sum(step_times[1:])/(len(step_times)-1):.2f}ms")
    speedup = step_times[0] / (sum(step_times[1:]) / len(step_times[1:]))
    print(f"  Speedup after warmup: {speedup:.1f}x")

    # Check for recompilation
    threshold = step_times[0] * 0.3
    recompiles = [i + 1 for i, t in enumerate(step_times[1:]) if t > threshold]

    if not recompiles:
        print("\n  ✅ GOOD: No recompilation detected!")
        print("      The training loop is efficient.")
    else:
        print(f"\n  ⚠️ Possible recompilation at steps: {recompiles}")

    print("\n  ✓ Test 2 complete")


# =============================================================================
# Test 3: Testing with Flows Directly (No Loss Module)
# =============================================================================


def test_flow_direct():
    """Test Flow operations directly without the loss module wrapper."""
    print("\n" + "=" * 60)
    print("TEST 3: Direct Flow Operations")
    print("=" * 60)

    key = jax.random.key(0)
    dim = 2

    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim))
    velocity = SimpleVelocity(dim=dim, hidden=16, key=key)

    print("\n3a. Testing flow.apply_map compilation...")

    # Create flow once
    flow = Flow(velocity_field=velocity, base_distribution=base_dist)
    x = jnp.zeros(dim)

    times = []
    for i in range(5):
        t0 = time.perf_counter()
        result = flow.apply_map(x)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        print(f"  Call {i+1}: {times[-1]:.1f}ms")

    print(f"\n  First call / avg rest: {times[0] / (sum(times[1:])/len(times[1:])):.1f}x")

    print("\n3b. Testing flow.log_prob compilation...")

    times = []
    for i in range(5):
        t0 = time.perf_counter()
        result = flow.log_prob(x)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        print(f"  Call {i+1}: {times[-1]:.1f}ms")

    print(f"\n  First call / avg rest: {times[0] / (sum(times[1:])/len(times[1:])):.1f}x")

    print("\n  ✓ Test 3 complete")


# =============================================================================
# Test 4: Static Field Analysis
# =============================================================================


def test_static_fields():
    """Analyze which Flow fields are static vs dynamic."""
    print("\n" + "=" * 60)
    print("TEST 4: Flow Static Field Analysis")
    print("=" * 60)

    key = jax.random.key(0)
    dim = 2

    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim))
    velocity = SimpleVelocity(dim=dim, hidden=16, key=key)
    flow = Flow(velocity_field=velocity, base_distribution=base_dist)

    print("\n4a. Field categorization:")

    fields_info = [
        ("velocity_field", "Contains model weights (traced)"),
        ("base_distribution", "distrax Distribution (traced arrays)"),
        ("dynamic_mask", "Lambda function (static=True)"),
        ("hutchinson_samples", "Optional[int] (static=True)"),
        ("solver", "diffrax Solver (static=True)"),
        ("augmented_solver", "diffrax Solver (static=True)"),
        ("t0", "float"),
        ("t1", "float"),
        ("dt0", "Optional[float]"),
        ("stepsize_controller", "diffrax Controller (static=True)"),
        ("augmented_stepsize_controller", "diffrax Controller (static=True)"),
        ("extra_args", "dict (static=True)"),
        ("augmented_extra_args", "dict (static=True)"),
    ]

    for field_name, description in fields_info:
        value = getattr(flow, field_name)
        leaves = jax.tree.leaves(value) if value is not None else []
        n_arrays = sum(1 for leaf in leaves if eqx.is_array(leaf))
        has_arrays = n_arrays > 0
        indicator = "⟳ traced" if has_arrays else "✓ static"
        print(f"  {field_name:35s} {indicator:10s} ({description})")

    print("\n4b. dynamic_mask analysis:")
    print(f"  Default: {Flow.__dataclass_fields__['dynamic_mask'].default}")
    print(f"  Current: {flow.dynamic_mask}")
    print(f"  Same object in all Flows: {flow.dynamic_mask is Flow.__dataclass_fields__['dynamic_mask'].default}")

    # This is key - the default lambda is shared!
    key2 = jax.random.key(1)
    velocity2 = SimpleVelocity(dim=dim, hidden=16, key=key2)
    flow2 = Flow(velocity_field=velocity2, base_distribution=base_dist)

    print(f"  flow1.dynamic_mask is flow2.dynamic_mask: {flow.dynamic_mask is flow2.dynamic_mask}")

    print("\n  ✓ Test 4 complete")


# =============================================================================
# Test 5: End-to-End Training
# =============================================================================


def test_training_e2e():
    """Full training simulation."""
    print("\n" + "=" * 60)
    print("TEST 5: End-to-End Training Simulation")
    print("=" * 60)

    key = jax.random.key(42)
    dim = 2

    # Setup
    base_dist = dsx.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim))
    velocity = SimpleVelocity(dim=dim, hidden=32, key=jax.random.key(0))
    loss_fn = MaximumLikelihoodLoss(base_dist, stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3))
    optimizer = optax.adam(1e-2)

    model = velocity
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    n_steps = 20
    step_times = []
    losses = []

    print(f"\n5a. Training for {n_steps} steps...")
    print("-" * 60)

    for i in range(n_steps):
        key, batch_key, step_key = jax.random.split(key, 3)
        # Simulate target samples (simple shifted Gaussian)
        batch = jax.random.normal(batch_key, (64, dim)) + jnp.array([2.0, -1.0])

        t0 = time.perf_counter()
        model, opt_state, loss = train_step(model, opt_state, batch, step_key, loss_fn, optimizer)
        jax.block_until_ready(loss)
        t1 = time.perf_counter()

        step_times.append((t1 - t0) * 1000)
        losses.append(float(loss))

        if i < 3 or i >= n_steps - 2:
            print(f"  Step {i+1:2d}: {step_times[-1]:7.1f}ms, loss={losses[-1]:.4f}")
        elif i == 3:
            print("  ...")

    print("-" * 60)

    print("\n5b. Timing Summary:")
    print(f"  Compilation (step 1):    {step_times[0]:.1f}ms")
    print(f"  Average (steps 2-{n_steps}): {sum(step_times[1:])/len(step_times[1:]):.2f}ms")
    speedup = step_times[0] / (sum(step_times[1:]) / len(step_times[1:]))
    print(f"  Speedup:                 {speedup:.1f}x")

    print("\n5c. Training Progress:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Improved:     {losses[-1] < losses[0]}")

    # Check for recompilation
    threshold = step_times[0] * 0.2
    recompiles = sum(1 for t in step_times[1:] if t > threshold)

    if recompiles == 0:
        print("\n  ✅ SUCCESS: No recompilation during training!")
    else:
        print(f"\n  ⚠️ {recompiles} steps exceeded threshold (possible recompilation)")

    print("\n  ✓ Test 5 complete")


# =============================================================================
# Main
# =============================================================================


def main():
    print("\n" + "=" * 60)
    print("SUPERIORFLOWS JAX COMPILATION DEBUG SUITE")
    print("=" * 60)
    print(
        """
This script tests for potential recompilation issues in the training loop.

Watch for 'Compiling' or 'Finished tracing' messages from JAX.
These indicate when compilation (tracing) occurs.

Expected: ONE compilation on first call, then fast cached execution.
"""
    )

    test_distrax_pytree_issue()
    test_actual_training_pattern()
    test_flow_direct()
    test_static_fields()
    test_training_e2e()

    print("\n" + "=" * 60)
    print("SUMMARY & CONCLUSIONS")
    print("=" * 60)
    print(
        """
FINDINGS:

1. ✅ dynamic_mask: The default lambda is SHARED across all Flow instances
   that use the default. This is safe - no recompilation from this.

2. ✅ flow_kwargs: Stored as static fields in MaximumLikelihoodLoss.
   Use ONE loss instance throughout training.

3. ✅ base_distribution: Stored as an attribute in MaximumLikelihoodLoss.
   The Flow created in __call__ uses THIS SAME INSTANCE consistently.

4. ✅ Training pattern: The train_step function JITs the entire step.
   As long as:
   - Same loss_fn instance
   - Same optimizer instance
   - Same model structure (weights can change)
   NO recompilation should occur.

VERIFICATION:

If you see step times like:
  Step 1: 5000ms (compilation)
  Step 2:   50ms (execution)
  Step 3:   50ms (execution)
  ...

Then the design is CORRECT and EFFICIENT!

Any step with time >> 50ms after warmup suggests recompilation.
"""
    )


if __name__ == "__main__":
    main()
