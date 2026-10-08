---
trigger: always_on
---

## Core design patterns

### Transport, Bijector, SDE, Sampler, and Flow hierarchy

The core library decomposes generative transport and normalising flows into a modular, multi-layer hierarchy:

1. **`AbstractTransport` (`transport.py`)**: The foundational base class for all state transformations on arbitrary PyTrees (deterministic or stochastic). It owns `dynamic_mask` and encapsulates state-context partitioning at the top level via `_partition_and_apply`. Defines the public methods:
   - `forward(x, **kwargs) -> y`
   - `forward_and_log_weight(x, **kwargs) -> (y, c)`
   Subclasses implement `_forward` and `_forward_and_log_weight` operating exclusively on pre-partitioned dynamic DOFs with static context passed via `args`.

2. **`AbstractBijector` (`bijector.py`)**: Subclasses `AbstractTransport` for all invertible maps on arbitrary PyTrees (both continuous ODE flows and discrete architectures like RealNVP or analytical bijectors).
   - Defines: `inverse(y, **kwargs)`, `forward_and_log_det(x, **kwargs)`, and `inverse_and_log_det(y, **kwargs)`.
   - Bridges to the unified transport interface via `_forward_and_log_weight` returning $(y, c)$ with $c = -\text{log\_det}$.
   - Subclasses implement `_forward_and_log_det` and `_inverse_and_log_det`.

3. **`ODEBijector` (`ode.py`)**: Continuous normalising flow bijector generating dynamics $\mathrm{d}x/\mathrm{d}t = v(t, x, \text{args})$ via Diffrax.
   - **Divergence strategies**:
     - *Exact* (default): diagonal JVP extraction via `jax.linearize`, $\mathcal{O}(d^2)$ FLOPs, $\mathcal{O}(d)$ memory.
     - *Hutchinson*: stochastic trace estimation with Rademacher vectors ($\pm 1$), $\mathcal{O}(d \cdot n_{\text{samples}})$. Enabled via `hutchinson_samples=k`.
     - *Analytical*: user-supplied `divergence_fn(velocity_field, t, x, args) \to (v, \text{div}_v)`.
   - **Dual solver configuration**: Plain ODE (`integrate`, `forward`) and augmented ODE (`integrate_augmented_ode`, `forward_and_log_det`) use independent solvers and controllers (`solver`/`augmented_solver`, `stepsize_controller`/`augmented_stepsize_controller`).

4. **`SDETransport` (`sde.py`)**: Stochastic transport via Euler–Maruyama SDE integration:
   $$\mathrm{d}X_t = \left[\hat{b}(t, X_t) + \varepsilon(t)\hat{s}(t, X_t)\right]\mathrm{d}t + \sqrt{2\varepsilon(t)}\,\mathrm{d}W_t$$
   - Uses `EulerMaruyamaWithAction` to compute the in-step Girsanov path action $\mathcal{S}_M = \sum_{k=0}^{M-1}(\log p^+ - \log p^-)$ with zero-allocation, leaf-by-leaf Gaussian residual evaluations.
   - Returns $(y, c)$ where $c = \mathcal{S}_M$, providing exact reweighting for importance sampling.

5. **`NeuralSampler` (`sampler.py`)**: Transport-agnostic sampler composing any `AbstractTransport` (bijective or stochastic) with a `base_distribution`.
   - Exposes: `sample(key, **kwargs)`, `sample_and_log_weight(key, **kwargs)`, and `push_forward_and_log_weight(x, **kwargs)`.
   - Unified log-weight: $\text{log\_weight}(y) = \log q_0(x_0) + c$. Importance weight against target density $\rho_\star$: $\log w = \log \rho_\star(y) - \text{log\_weight}(y)$.

6. **`Flow` (`flow.py`)**: Normalising flow distribution subclassing `NeuralSampler` when `transport` is an `AbstractBijector`.
   - Extends with exact bijective density evaluation: `log_prob(y, **kwargs)` and `pull_back_and_log_prob(y, **kwargs)`.
   - Exposes backward-compatible aliases: `sample_and_log_prob` and `push_forward_and_log_prob` (which delegate to `*_log_weight`), and the `.bijector` property.

---

### Dynamic masking & generalised state-context partitioning (`partial.py`)

The partitioning mechanism in `superiorflows.partial` provides a drop-in generalisation of `eqx.partition` that supports both leaf-level boolean masks and element-level integer index arrays:

```python
dynamic, static, spec = state_context_partition(x, mask)
merged_x = merge_state(new_dynamic, original_x, spec)
reconstructed_x = reconstruct_state(dynamic, static, index_meta)
```

#### Mask formats
- **Callable**: `leaf -> bool` (e.g. default `lambda x: jax.tree.map(eqx.is_inexact_array, x)`).
- **Boolean PyTree**: PyTree matching `x` with scalar boolean leaves (e.g. `ParticleSystem(positions=True, species=False, box=False)`).
- **Index-Array PyTree**: PyTree matching `x` with 1-D integer index arrays (or a single 1-D index array for vector states), selecting specific elements along axis 0.

#### Two primary operational roles
1. **PyTree State-Context Separation**: Decoupling dynamic state variables (e.g. particle positions) from static metadata (e.g. species, box vectors, conditioning labels) so transports and ODE/SDE solvers only integrate active leaves.
2. **Partial Updates & Conditional Sampling**: Flowing or diffusing a subset of degrees of freedom (e.g. $k$ dimensions of $\mathbb{R}^d$ or $k$ particles in a box) while conditioning on the remaining environment.

---

### Partial distributions, updaters, and selection protocols

For conditional generation, local MCMC/Gibbs moves, sub-volume patch updates, and inpainting:

- **`PartialBase(base_distribution, context, dynamic_mask, conditional=False)`**:
  Wraps a base distribution over the dynamic DOFs. `sample(key)` draws dynamic values and merges them with static DOFs from `context`. `log_prob(x)` extracts dynamic DOFs and evaluates prior density. When `conditional=True`, `context=ctx_static` is forwarded to `base_distribution.sample` and `base_distribution.log_prob`.

- **`PartialUpdater(transport, base_distribution, conditional=False)`** (aliased as `PartialFlowUpdater`):
  High-level module coupling any `AbstractTransport` (flow or SDE) and base distribution for partial sampling and reweighting. Exposes:
  - `update(x, dynamic_mask, key, **kwargs)`: returns $x_\text{updated}$.
  - `update_and_log_weight(x, dynamic_mask, key, **kwargs)`: returns $(x_\text{updated}, \text{log\_weight})$.
  - `push_forward_and_log_weight(x, dynamic_mask, **kwargs)`: returns $(x_\text{pushed}, \text{log\_weight})$.
  - `log_prob(x, dynamic_mask, **kwargs)`: exact log-prob (bijective transports only).
  - Aliases: `update_and_log_prob`, `push_forward_and_log_prob`, `.bijector`.

- **Selection Protocols**:
  Callables `(key, x) -> dynamic_mask` returning static-shaped index masks:
  - `uniform_index_selection(n)`: Uniformly samples $n$ indices along axis 0 without replacement.
  - `fixed_selection(mask)`: Deterministic mask ignoring key and state.

---

### Context passing convention (`args`)

When `AbstractBijector` partitions state `x` via `state_context_partition`, it packs context into `args` as:
- `args = ctx` (if no index metadata and no user args)
- `args = (ctx, user_args)` (if no index metadata but user args provided)
- `args = (ctx, (index_meta, user_args))` or `(ctx, index_meta)` (when index masks are used)

Velocity fields and models handle full vs. partial states via `reconstruct_state` and an `extract_dynamic_part` helper. `unwrap_partition_args` and `extract_dynamic_part` are not library functions: reference implementations live in `scripts/gaussian_mixture/train_gaussian_mixture.py`, and the core packs args with `_pack_args` in `transport.py`:
```python
ctx, index_meta, user_args = unwrap_partition_args(args)
if index_meta is not None:
    x_full = reconstruct_state(x_dyn, ctx, index_meta)
    v_full = self.net(t, x_full)
    return extract_dynamic_part(v_full, x_dyn, index_meta)
return self.net(t, x_dyn)
```

---

### Loss functions & training integration

All training losses in `src/superiorflows/train/losses.py` are `eqx.Module` instances with signature `__call__(self, model, batch, key) -> (scalar_loss, aux_dict)`:

| Loss | Input batch | `key` requirement | Description |
|------|------------|-------------------|-------------|
| `MaximumLikelihoodLoss` | target samples `x1` | Optional (required for Hutchinson / selection) | Negative log-likelihood via inverse map. |
| `EnergyBasedLoss` | base samples `x0` | Optional (required for Hutchinson / selection) | Reverse KL against `target_distribution` energy. |
| `KullbackLeiblerLoss` | target samples `x1` | Required | Hybrid $\alpha \text{MLE} + (1-\alpha) \text{Energy}$. |
| `StochasticInterpolantLoss` | coupled pairs `(x0, x1)` | Required | Velocity matching + optional denoiser & boundary TSM. |

#### Unified partial training
All loss functions accept an optional `selection_protocol=...`. When provided:
- Losses split PRNG keys per batch item and evaluate `mask_i = selection_protocol(key_i, x_i)`.
- States are partitioned dynamically per sample, and the loss is evaluated exclusively over dynamic DOFs while conditioning on static context.
- Batches support optional per-sample metadata via `_split_batch_metadata` / `_split_paired_batch_metadata`.

#### Unified context & metadata passing
- Batches support optional per-sample metadata via `_split_batch_metadata` / `_split_paired_batch_metadata`.
- Loss modules must forward metadata strictly as generic `args` (via `args_kw = {"args": args_val}`), passing it uniformly to flow calls (`flow.push_forward_and_log_prob`, `updater.update_and_log_prob`) and target distribution evaluations (`target_distribution.log_prob(x, **args_kw)`).
- Core losses must never unpack or interpret `args` or assume specific application conventions. Domain-specific unpacking (e.g. `(n_patch, n_context)`) belongs strictly within domain packages (e.g. `BoltzmannDistribution.log_prob` in `particle_systems`).

---

### Trainer & callbacks

`Trainer` (`src/superiorflows/train/trainer.py`) is a plain Python class orchestrating the training loop:
- Pure JIT-compiled step function: `train_step(model, opt_state, batch, key, loss_module, optimizer)`.
- Callback lifecycle: `on_train_start(trainer, total_steps=...)`, `on_step_end(trainer, step=..., logs=...)`, `on_train_end(trainer)`.
- Callbacks mutate `trainer.logs` dict in registration order (`TensorBoardLogger` must be registered last).
- Checkpointing uses Orbax with split items (`model`, `optimizer`, `metadata` with data iterator state).

---

### Data pipeline

Grain `MapDataset` pipelines are assembled by the caller and passed to `Trainer.train(dataset, max_steps=...)`.

- `DistributionDataSource` (`src/superiorflows/train/data.py`): Deterministic batch generation via `jax.random.fold_in`.
- `TrajectoryDataSource` (`particle_systems/particle_system.py`): Pre-loads trajectory frames into contiguous numpy arrays at initialisation (zero-copy slicing).
- Stochastic interpolant training uses `CoupleBaseSamples` (`map_with_index`) + `EquivariantOptimalTransport` (`map`), both in `particle_systems/particle_system.py`. Patch training uses the analogues in `particle_systems/patch_updater.py` (`CouplePatchBase`, `PatchEquivariantOptimalTransport`).

**GPU performance guidelines**:
- **CPU data loading**: All sampling and preprocessing run on CPU via `jax.default_device(jax.devices("cpu")[0])`.
- **Asynchronous prefetching**: Configure `grain.ReadOptions(num_threads=..., prefetch_buffer_size=...)` to prevent GPU starvation.
- **CPU transforms**: Optimal transport (`scipy.optimize.linear_sum_assignment`) and permutation sorting remain on CPU threads.
- **Implicit device transfer**: Batches transfer to GPU implicitly upon entering `train_step`.

---

### Configuration system

Layered configuration: `DEFAULT_CONFIG` (hardcoded dictionary) $\to$ JSON file (`--config`) $\to$ CLI overrides (Typer).

Schedule expressions are eval'd Python strings with `optax` and `jnp` in scope, e.g.:
```json
"lr_schedule": "optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=50, decay_steps=1000, end_value=0.0)"
```
