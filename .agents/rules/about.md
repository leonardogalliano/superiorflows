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

- **`src/superiorflows/`** — the **core library**, entirely domain-agnostic:
  - `transport.py` (`AbstractTransport`): base class for all transports (deterministic or stochastic); owns dynamic masking and state-context partitioning (`_partition_and_apply`, `_pack_args`).
  - `bijector.py` (`AbstractBijector`): invertible maps (ODE-based or discrete, e.g. RealNVP) with forward/inverse log-det.
  - `ode.py` (`ODEBijector`): CNF dynamics via Diffrax with exact, Hutchinson, or analytical divergence.
  - `sde.py` (`SDETransport`, `EulerMaruyamaWithAction`, `SDETerm`): Euler–Maruyama transport with in-step Girsanov path action.
  - `sampler.py` (`NeuralSampler`): composes any `AbstractTransport` with a base distribution.
  - `flow.py` (`Flow`): `NeuralSampler` subclass with exact density evaluation for bijective transports.
  - `partial.py`: state partitioning (`state_context_partition`, `merge_state`, `reconstruct_state`), `PartialBase`, `PartialUpdater` (alias `PartialFlowUpdater`), selection protocols.
  - `train/`: `Trainer`, losses (`losses.py`), `DistributionDataSource` (`data.py`), callbacks (`callbacks.py`).

  It knows nothing about particles, boxes, species, potentials, or patches (see anti-pattern 1).

- **`particle_systems/`** — the **domain-specific package** for interacting particle systems on periodic flat tori: `ParticleSystem`, distributions (`UniformParticles`, `BoltzmannDistribution`), velocity fields (`ParticlesMLPVelocity`, `ParticlesEGNNVelocity` in `velocities/`), potentials (via jax-md), patch updaters and selection (`patch_updater.py`, `selection.py`), the particle data pipeline (`particle_system.py`), Typer CLI scripts, example configs (`config/`), HPC SLURM launchers (`HPC_scripts/`), and domain tests (`tests/`).
  - Scripts with `pyproject.toml` entry points: `training-particles`, `evaluate-log-prob`, `sampling-patches` (run with `uv run <name>`).
  - Scripts without entry points: `training_patches.py`, `sampling_particles.py` — run with `uv run python -m particle_systems.training_patches` (resp. `sampling_particles`).

Other top-level directories: `tests/` (core library tests), `scripts/` (standalone demos; `scripts/gaussian_mixture/` holds the reference toy-problem pipeline), `experiments/` (outputs), `notebooks/` (gitignored), `tmp/` (scratch, gitignored).

**Known packaging caveat**: `particle_systems` ships in the wheel but its runtime dependencies (`jax-md`, `atooms`, `scipy`, `tensorboardX`) are only in the `dev` dependency group. Inside this repo `uv sync` installs them; a plain `pip install` of the wheel would not.

---

## Technology stack

| Layer | Tool | Notes |
|-------|------|-------|
| Build & deps | **uv** + hatchling | `uv sync` (includes dev group, incl. TensorBoard). GPU: `uv sync --extra cuda`. Always run via `uv run`. |
| Models | **Equinox** | All models are `eqx.Module` PyTrees. Parameters = inexact arrays. |
| ODE integration | **Diffrax** | `Tsit5` (default), `Dopri5`, `Euler`. Adaptive via `PIDController`. |
| Distributions | duck-typed `eqx.Module`s | The library does not depend on Distreqx; distributions are `eqx.Module`s exposing `sample`/`log_prob` with explicit vectorisation. **Distreqx** (dev only) is used in tests and scripts as reference base distributions. |
| Optimisation | **Optax** | Schedules often defined as eval'd string expressions in config. |
| Checkpointing | **Orbax** | Split items: `model`, `optimizer`, `metadata` (with data iterator state). |
| Data loading | **Grain** | `MapDataset` pipelines: source → shuffle → batch → repeat. |
| Pair potentials | **jax-md** | Energy functions via `smap.pair` with species-indexed parameters. |
| Trajectory I/O | **atooms** | `.xyz` file reading/writing; `atooms-pp` for g(r) post-processing. |
| CLI | **Typer** | See entry points above. |
| Linting | **Ruff** | line-length 120, `select = ["E", "F", "I", "W"]`, double quotes; also run by pre-commit. |
| CI | **GitHub Actions** | Matrix: `{ubuntu, macos}` × `{3.12, 3.13}`. Coverage → Codecov. |

## Coding standards

- **British English** in all text.
- **Physics notation**: `N` (particles), `d` (dimensions), `L` (box size), `T` (temperature), `U` (energy), `t` (flow time), `v` (velocity), `x0`/`x1` (base/target samples).
- **No numbered comments** — code must be self-documenting. Comments explain *why*, never *what*.
- **Formal docstrings** at API boundaries with explicit shape annotations (e.g., `shape (N, d)`).
- **Experimental code** goes in `tmp/` (gitignored). Never commit scratchpad files to the main tree.
