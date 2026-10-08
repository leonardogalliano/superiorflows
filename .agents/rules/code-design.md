---
trigger: always_on
---

## Core design patterns

This section records invariants and conventions that are not obvious from reading a single file. For signatures and per-method behaviour, read the docstrings in `src/superiorflows/`.

### Transport hierarchy

`AbstractTransport` → `AbstractBijector` → `ODEBijector` (and discrete bijectors); `AbstractTransport` → `SDETransport`. `NeuralSampler` composes any transport with a base distribution; `Flow` subclasses it for bijective transports.

- Public methods (`forward`, `forward_and_log_weight`, `inverse`, …) partition the state once via `_partition_and_apply`; subclasses implement the underscored versions (`_forward`, `_forward_and_log_weight`, `_forward_and_log_det`, `_inverse_and_log_det`) operating **only on dynamic DOFs**, with static context passed through `args`.
- **Unified log-weight convention**: transports return $(y, c)$. For bijectors $c = -\text{log\_det}$; for SDEs $c = \mathcal{S}_M$, the Girsanov path action accumulated in-step by `EulerMaruyamaWithAction`. Then $\text{log\_weight}(y) = \log q_0(x_0) + c$ and the importance weight is $\log w = \log \rho_\star(y) - \text{log\_weight}(y)$.
- `*_log_prob` methods on `Flow` / `PartialUpdater` are backward-compatible aliases of `*_log_weight`; exact `log_prob` / `pull_back_and_log_prob` exist only for bijective transports.
- **`ODEBijector` divergence**: exact (default, diagonal JVP via `jax.linearize`), Hutchinson (`hutchinson_samples=k`, Rademacher, needs `key`), or analytical (`divergence_fn(velocity_field, t, x, args) -> (v, div_v)`).
- **`ODEBijector` dual solvers**: plain ODE (`forward`, `integrate`) uses `solver`/`stepsize_controller`; augmented ODE (`forward_and_log_det`, `integrate_augmented_ode`) uses `augmented_solver`/`augmented_stepsize_controller`. Always set both (anti-pattern 3).

### State-context partitioning (`partial.py`)

`state_context_partition(x, mask) -> (dynamic, static, spec)` generalises `eqx.partition`; `merge_state(new_dynamic, original_x, spec)` and `reconstruct_state(dynamic, static, index_meta)` invert it. Mask formats:
- **Callable** `leaf -> bool` (default: `eqx.is_inexact_array`).
- **Boolean PyTree** matching `x` with scalar boolean leaves (e.g. `ParticleSystem(positions=True, species=False, box=False)`).
- **Index-array PyTree** matching `x` with 1-D integer index arrays (or a single 1-D array for vector states), selecting elements along axis 0. Used for partial updates (e.g. $k$ particles in a box).

`PartialBase` and `PartialUpdater` wrap base distributions / transports for conditional sampling, MCMC/Gibbs moves and patch updates; with `conditional=True` the static context is forwarded to the base distribution. Selection protocols are callables `(key, x) -> dynamic_mask` with static output shape: `uniform_index_selection(n)`, `fixed_selection(mask)`.

### Context passing convention (`args`)

`_pack_args` (`transport.py`) packs the partitioned context into `args` as:
- `args = ctx` (no index metadata, no user args)
- `args = (ctx, user_args)` (no index metadata, user args provided)
- `args = (ctx, index_meta)` or `(ctx, (index_meta, user_args))` (index masks)

Velocity fields that must see the full state reconstruct it. `unwrap_partition_args` and `extract_dynamic_part` are **not** library functions; reference implementations live in `scripts/gaussian_mixture/train_gaussian_mixture.py`:
```python
ctx, index_meta, user_args = unwrap_partition_args(args)
if index_meta is not None:
    x_full = reconstruct_state(x_dyn, ctx, index_meta)
    v_full = self.net(t, x_full)
    return extract_dynamic_part(v_full, x_dyn, index_meta)
return self.net(t, x_dyn)
```

### Losses (`train/losses.py`)

All losses are `eqx.Module`s with `__call__(self, model, batch, key) -> (scalar_loss, aux_dict)`: `MaximumLikelihoodLoss` (target samples), `EnergyBasedLoss` (base samples, reverse KL), `KullbackLeiblerLoss` (hybrid, key required), `StochasticInterpolantLoss` (coupled `(x0, x1)` pairs, key required).
- Optional `selection_protocol=...`: keys are split per batch item and each sample is partitioned with its own mask.
- Per-sample metadata is split off the batch by `_split_batch_metadata` / `_split_paired_batch_metadata` and forwarded **only** as generic `args` (`args_kw = {"args": args_val}`) to model calls and `target_distribution.log_prob(x, **args_kw)`. Losses never interpret `args` (anti-pattern 1).

### Trainer & callbacks (`train/trainer.py`, `train/callbacks.py`)

- JIT-compiled step: `train_step(model, opt_state, batch, key, loss_module, optimizer)`.
- Lifecycle: `on_train_start`, `on_step_end(trainer, step, logs)`, `on_train_end`. Callbacks mutate `trainer.logs` in registration order, so **`TensorBoardLogger` must be registered last**.
- Orbax checkpoints with split items `model`, `optimizer`, `metadata` (includes data iterator state).

### Data pipeline

Grain `MapDataset` pipelines are assembled by the caller and passed to `Trainer.train(dataset, max_steps=...)`.
- `DistributionDataSource` (core): deterministic batches via `jax.random.fold_in`.
- `TrajectoryDataSource` (`particle_system.py`): pre-loads frames into contiguous numpy arrays.
- Stochastic interpolant coupling: `CoupleBaseSamples` + `EquivariantOptimalTransport` (`particle_system.py`); patch analogues `CouplePatchBase` + `PatchEquivariantOptimalTransport` (`patch_updater.py`).
- Sampling, OT (`scipy.optimize.linear_sum_assignment`) and preprocessing run on CPU (`jax.default_device(jax.devices("cpu")[0])`) with `grain.ReadOptions(num_threads=..., prefetch_buffer_size=...)` to avoid GPU starvation; batches transfer to GPU implicitly in `train_step`.

### Configuration

Layered: `DEFAULT_CONFIG` (in each CLI script) → JSON file (`--config`) → CLI overrides. Schedule expressions are eval'd strings with `optax` and `jnp` in scope (anti-pattern 6), e.g.
```json
"lr_schedule": "optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=50, decay_steps=1000, end_value=0.0)"
```
