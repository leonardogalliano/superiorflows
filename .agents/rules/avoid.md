---
trigger: always_on
---

## Critical anti-patterns to avoid

1. **Never put domain-specific logic in `src/superiorflows/`**. No imports from `particle_systems` in the core library.

2. **Never mutate Equinox modules in-place**. Use `eqx.tree_at` for immutable updates. The one exception is `object.__setattr__` for weight initialisation at construction time (see EGNN `coord_w` init).

3. **Never forget to set both solver pairs** when overriding solver config on an `ODEBijector`. The plain ODE and augmented ODE use independent solvers/controllers.

4. **Always use `jax.random.key` (not the legacy `jax.random.PRNGKey`) with explicit splitting.** Every stochastic operation must consume a fresh subkey obtained via `jax.random.split`. When Hutchinson is enabled, when SDE transport is used, or when a `selection_protocol` is active, `key` is mandatory — forgetting it raises `TypeError` or `ValueError`.

5. **Never construct data-dependent shapes inside `jit`-traced code**. Array shapes must be statically known at trace time. In selection protocols, the number of selected elements $n$ must be a static integer (as in `uniform_index_selection(n)`).

6. **Never use `eval()` on untrusted input**. The `build_schedule_fn` and `build_optimizer` functions (`particle_systems/training_particles.py`) eval config strings — this is intentional and restricted to the operator's own config files.

7. **Never assume `args` is a raw context PyTree without unwrapping**. In bijectors and velocity fields supporting indexed partitions, `args` may contain `(ctx, index_meta, user_args)`. Always unwrap appropriately or use helper functions.

8. **Never add `__pycache__/`, `tmp/`, `.benchmarks/`, or notebook outputs to git**. These are gitignored.

9. **Never use silent defaults**. If a required config field is missing, raise an explicit error with the field name and how to provide it.

10. **Never log metrics outside the `logs` dict mechanism**. Callbacks inject into `trainer.logs` and downstream callbacks (Logger, TensorBoard) consume them. This is the single source of truth for step-level metrics.

11. **Never skip `.block_until_ready()`** in benchmark or timing code. JAX dispatch is async.

12. **Never introduce application-specific parameter unpacking into `src/`**. The core library must stay completely pure and application-agnostic (from toy Gaussian mixtures, image generation, to partial updates and physical systems). Generic components (such as losses, bijectors, transports, updaters) must never assume specific tuple formats (such as `(n_patch, n_context)`) or physical concepts. Context and metadata must flow as abstract `args` PyTrees or kwargs through the pipeline.

13. **Never rely on raw sample means for heavy-tailed physical observables in logs** (e.g. potential energies in steep potentials where rare overlaps skew the mean to astronomical numbers); always log robust order statistics (medians, interquartile ranges, percentiles, finite fractions).