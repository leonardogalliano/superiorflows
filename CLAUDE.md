# CLAUDE.md

Project guidance for Claude Code. Content mirrors `.agents/rules/*.md` and `.agents/workflows/ruff.md` (used by Antigravity); keep both in sync when editing.


## What this project is

**Superiorflows** is a JAX library for sampling via Continuous Normalising Flows (CNFs). The core abstraction operates on **arbitrary PyTree-structured states** — it is not hard-wired to particle systems. The primary research application is sampling Boltzmann distributions of interacting particle systems on periodic flat tori (PBC), developed at the Centre des Sciences des Données, École Normale Supérieure, Paris.

The codebase is inspired by the PyTorch implementation [learndiffeq](https://github.com/h2o64/learndiffeq) but is a ground-up JAX rewrite.

---

## Repository architecture

The repository contains **two installable Python packages** (both declared in `pyproject.toml` under `tool.hatch.build.targets.wheel.packages`):

- **`src/superiorflows/`** — the **core library**, entirely domain-agnostic. Built on a modular hierarchy:
  - `transport.py` (`AbstractTransport`): foundational base class for all transports (deterministic flows or stochastic SDEs), encapsulating dynamic masking and state-context partitioning via `_partition_and_apply`.
  - `bijector.py` (`AbstractBijector`): abstract invertible maps (ODE-based or discrete architectures like RealNVP) with exact forward/inverse log-det Jacobian and dynamic masking.
  - `ode.py` (`ODEBijector`): continuous flow dynamics via Diffrax ODE integration with exact, Hutchinson, or analytical divergence.
  - `sde.py` (`SDETransport`, `EulerMaruyamaWithAction`, `SDETerm`): stochastic continuous transport with in-step Girsanov path action accumulation for exact reweighting and importance sampling.
  - `sampler.py` (`NeuralSampler`): transport-agnostic distribution/sampler composing any `AbstractTransport` with a base distribution to provide `sample`, `sample_and_log_weight`, and `push_forward_and_log_weight`.
  - `flow.py` (`Flow`): normalising flow distribution subclassing `NeuralSampler` with exact bijective density evaluation (`log_prob`, `pull_back_and_log_prob`).
  - `partial.py`: generalised state partitioning (`state_context_partition`, `merge_state`, `reconstruct_state`), `PartialBase`, `PartialUpdater` (aliased as `PartialFlowUpdater`), and stochastic selection protocols.
  - `train/`: training loop orchestration (`Trainer`), loss modules (`losses.py`), the generic `DistributionDataSource` (`data.py`), and callbacks (`callbacks.py`).
  It knows nothing about particles, boxes, species, potentials, or patches. It must remain general and purely mathematical, applicable to any state PyTree (from toy Gaussian mixtures, image generation, to physical systems and partial state updates). Never import from `particle_systems` here, and never assume domain-specific tuple formats or physical semantics in core methods.

- **`particle_systems/`** — the **domain-specific package** for interacting particle systems on periodic flat tori. Contains `ParticleSystem`, distributions (`UniformParticles`, `BoltzmannDistribution`), velocity fields (`ParticlesMLPVelocity`, `ParticlesEGNNVelocity`), potentials (via jax-md), patch updaters and selection (`patch_updater.py`, `selection.py`), the particle data pipeline (`TrajectoryDataSource`, `CoupleBaseSamples`, `EquivariantOptimalTransport` in `particle_system.py`), training/sampling CLI scripts (typer; entry points `training-particles`, `evaluate-log-prob`, `sampling-patches`), example configs, HPC SLURM launchers, and domain-specific tests (in `particle_systems/tests/`).

Other top-level directories: `tests/` (core library tests), `scripts/` (standalone demos), `experiments/` (outputs), `notebooks/` (gitignored), `tmp/` (scratch, gitignored).

---

## Technology stack

| Layer | Tool | Notes |
|-------|------|-------|
| Build & deps | **uv** + hatchling | Standard install: `uv sync`. GPU: add `--extra cuda`. Always run via `uv run`. |
| Models | **Equinox** | All models are `eqx.Module` PyTrees. Parameters = inexact arrays. |
| ODE integration | **Diffrax** | `Tsit5` (default), `Dopri5`, `Euler`. Adaptive via `PIDController`. |
| Distributions | **Distreqx** | `Flow` and custom distributions are duck-typed `eqx.Module` PyTrees with explicit vectorisation. |
| Optimisation | **Optax** | Schedules often defined as eval'd string expressions in config. |
| Checkpointing | **Orbax** | Split items: `model`, `optimizer`, `metadata` (with data iterator state). |
| Data loading | **Grain** | `MapDataset` pipelines: source → shuffle → batch → repeat. |
| Pair potentials | **jax-md** | Energy functions via `smap.pair` with species-indexed parameters. |
| Trajectory I/O | **atooms** | `.xyz` file reading/writing; `atooms-pp` for g(r) post-processing. |
| CLI | **Typer** | Entry points `training-particles`, `evaluate-log-prob`, `sampling-patches` defined in `pyproject.toml`. |
| Linting | **Ruff** | line-length 120, `select = ["E", "F", "I", "W"]`, format on pre-commit. |
| CI | **GitHub Actions** | Matrix: `{ubuntu, macos}` × `{3.12, 3.13}`. Coverage → Codecov. |

**Default install**: most scripts and callbacks depend on TensorBoard (`tensorboardX`), so the standard development install is:
```bash
uv sync
```
For GPU clusters, use: `uv sync --extra cuda`.

## Coding standards

- **British English** in all text.
- **Physics notation**: `N` (particles), `d` (dimensions), `L` (box size), `T` (temperature), `U` (energy), `t` (flow time), `v` (velocity), `x0`/`x1` (base/target samples).
- **No numbered comments** — code must be self-documenting. Comments explain *why*, never *what*.
- **Formal docstrings** at API boundaries with explicit shape annotations (e.g., `shape (N, d)`).
- **Ruff**: line length 120, `select = ["E", "F", "I", "W"]`, double quotes, space indent.
- **Experimental code** goes in `tmp/` (gitignored). Never commit scratchpad files to the main tree.

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

## Testing

Sometimes you find the issue "no module named superiorflow". According to Gemini, this is due to iCloud syncing. To solve this, run chflags -R nohidden .venv.

### Test locations
- `tests/` — core library tests (`test_flow.py`, `test_trainer.py`, `test_partial.py`, `test_particles.py`, etc.)
- `particle_systems/tests/test_*.py` — domain-specific tests for the particle package
- `scripts/gaussian_mixture/test_gaussian_mixture.py` — integration tests for unbounded Gaussian mixtures, discrete RealNVP, ODE flows, and partial DOF training/sampling

### Running tests
```bash
uv run pytest --cov=src --cov=particle_systems --cov-report=xml
```

### Test patterns
- **Fixtures via `pytest.fixture`** with parametrisation over config files (`@pytest.fixture(params=CONFIG_FILES)`).
- **CLI script tests** use `typer.testing.CliRunner` to invoke `training_app` and `sampling_app` programmatically.
- **Checkpoint restart tests**: run short training → find latest run dir → restart with `--load-from-checkpoint`.
- **Performance tests** use `pytest-benchmark` with `benchmark(fn)` pattern and `.block_until_ready()` for JAX async.
- Temporary directories for checkpoints use `tmp_path` (pytest built-in) or the gitignored `tmp/` directory.

### What CI runs
GitHub Actions matrix: `{ubuntu-latest, macos-latest}` × `{python 3.12, 3.13}`. Installs via `uv sync` (TensorBoard is included in dev dependencies), runs pytest with coverage, uploads to Codecov.

---

## HPC deployment

SLURM scripts live in `particle_systems/HPC_scripts/` targeting Jean Zay (IDRIS) and Swift clusters.

Key environment setup for GPU nodes:
```bash
# Dynamic linking fix for JAX XLA
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
VENV_NVIDIA_LIBS=$(find .venv/lib/python3.12/site-packages/nvidia -type d -name "lib" | tr '\n' ':')
export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"
export UV_CACHE_DIR=$WORK/.uv_cache
```

Supports V100, A100, H100 via architecture-specific module loading. Jobs dispatch via `srun uv run`.

---

---

## Lint and format workflow

The agent must automatically perform the following steps to lint and format the codebase:

1. Run Ruff lint check:
   ```bash
   uv run ruff check
   ```

2. Run Ruff format check:
   ```bash
   uv run ruff format --check
   ```

3. If any issues are found:
   - For lint errors: Run `uv run ruff check --fix` to auto-fix safe problems. For any remaining errors, locate and fix them in the source files.
   - For formatting differences: Run `uv run ruff format` to automatically reformat the codebase.
   - Re-run the checks to confirm everything is clean.

4. Provide a clear summary of the status (e.g. "All checks passed!" or details of the fixes applied).
