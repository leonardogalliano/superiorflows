---
trigger: always_on
---

# Superiorflows — Workspace Rules

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
