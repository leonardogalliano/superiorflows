---
trigger: always_on
---

## Critical anti-patterns to avoid

1. **Never put domain-specific logic in `src/superiorflows/`**. No imports from `particle_systems` in the core library. Generic components (losses, bijectors, transports, updaters) must never assume specific tuple formats (such as `(n_patch, n_context)`) or physical concepts; the core must work equally for toy Gaussian mixtures, images, partial updates and physical systems. Context and metadata flow as abstract `args` PyTrees or kwargs; domain-specific unpacking belongs in domain packages (e.g. `BoltzmannDistribution.log_prob`).

2. **Never mutate Equinox modules in-place**. Use `eqx.tree_at` for immutable updates. The one exception is `object.__setattr__` for weight initialisation at construction time (see EGNN `coord_w` init in `particle_systems/velocities/egnn.py`).

3. **Never forget to set both solver pairs** when overriding solver config on an `ODEBijector`. The plain ODE and augmented ODE use independent solvers/controllers.

4. **Always use `jax.random.key` (not the legacy `jax.random.PRNGKey`) with explicit splitting.** Every stochastic operation must consume a fresh subkey obtained via `jax.random.split`. When Hutchinson is enabled, when SDE transport is used, or when a `selection_protocol` is active, `key` is mandatory — forgetting it raises `TypeError` or `ValueError`.

5. **Never construct data-dependent shapes inside `jit`-traced code**. Array shapes must be statically known at trace time. In selection protocols, the number of selected elements $n$ must be a static integer (as in `uniform_index_selection(n)`).

6. **Never use `eval()` on untrusted input**. The `build_schedule_fn` and `build_optimizer` functions (`particle_systems/training_particles.py`) eval config strings — this is intentional and restricted to the operator's own config files.

7. **Never assume `args` is a raw context PyTree without unwrapping**. In bijectors and velocity fields supporting indexed partitions, `args` may contain context, index metadata and user args (see "Context passing convention"). Always unwrap appropriately.

8. **Never add `__pycache__/`, `tmp/`, `.benchmarks/`, or notebook outputs to git**. These are gitignored.

9. **Never use silent defaults**. If a required config field is missing, raise an explicit error with the field name and how to provide it.

10. **Never log metrics outside the `logs` dict mechanism**. Callbacks inject into `trainer.logs` and downstream callbacks (Logger, TensorBoard) consume them. This is the single source of truth for step-level metrics.

11. **Never skip `.block_until_ready()`** in benchmark or timing code. JAX dispatch is async.

12. **Never rely on raw sample means for heavy-tailed physical observables in logs** (e.g. potential energies in steep potentials where rare overlaps skew the mean to astronomical numbers); always log robust order statistics (medians, interquartile ranges, percentiles, finite fractions).

13. **Never use the Hutchinson estimator for density or log-weight evaluation.** It biases the log-weights. Hutchinson is acceptable only at training time (e.g. maximum likelihood). Evaluation, $\hat Z$ and ESS always use the exact divergence.

14. **Never cherry-pick benchmark results.** Report oracle *and* trained models, ODE *and* SDE, for every dimension and step count that was run, with solver, step count, seeds and config. Failures are reported as failures (e.g. `NaN`, "collapsed"), never as clamped sentinel values (`1e-12`, `-1e6`) that look like measurements. State limitations explicitly; never overstate (e.g. "unbiased" for an oracle-only result, speedups measured on collapsed models).

15. **Never treat a t-statistic on $\hat Z$ as a valid test when the ESS is low.** Importance weights are heavy-tailed; the sample mean then typically underestimates $Z$ and the SEM is unreliable. Report ESS alongside, use several seeds, and prefer $\log\hat Z$ with bootstrap intervals.
