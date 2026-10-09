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
  - `sde.py` (`SDETransport`, `SDESolution`): Euler–Maruyama transport with in-step Girsanov path action, implemented natively as a `jax.lax.scan` (no Diffrax).
  - `sampler.py` (`NeuralSampler`): composes any `AbstractTransport` with a base distribution.
  - `flow.py` (`Flow`): `NeuralSampler` subclass with exact density evaluation for bijective transports.
  - `partial.py`: state partitioning (`state_context_partition`, `merge_state`, `reconstruct_state`), `PartialBase`, `PartialUpdater` (alias `PartialFlowUpdater`), selection protocols.
  - `train/`: `Trainer`, losses (`losses.py`), `DistributionDataSource` (`data.py`), callbacks (`callbacks.py`).

  It knows nothing about particles, boxes, species, potentials, or patches (see anti-pattern 1).

- **`particle_systems/`** — the **domain-specific package** for interacting particle systems on periodic flat tori: `ParticleSystem`, distributions (`UniformParticles`, `BoltzmannDistribution`), velocity fields (`ParticlesMLPVelocity`, `ParticlesEGNNVelocity` in `velocities/`), potentials (via jax-md), patch updaters and selection (`patch_updater.py`, `selection.py`), the particle data pipeline (`particle_system.py`), Typer CLI scripts, example configs (`config/`), HPC SLURM launchers (`HPC_scripts/`), and domain tests (`tests/`).
  - Scripts with `pyproject.toml` entry points: `training-particles`, `evaluate-log-prob`, `sampling-patches` (run with `uv run <name>`).
  - Scripts without entry points: `training_patches.py`, `sampling_particles.py` — run with `uv run python -m particle_systems.training_patches` (resp. `sampling_particles`).

Other top-level directories: `tests/` (core library tests), `scripts/` (standalone demos; `scripts/gaussian_mixture/` holds the reference toy-problem pipeline, see below), `experiments/` (outputs), `notebooks/` (gitignored), `tmp/` (scratch, gitignored).

### Gaussian-mixture testbed (`scripts/gaussian_mixture/`)

The development testbed for ODE vs SDE sampling, before moving to particles. The target is the anisotropic bimodal mixture of Grenioux, Noble & Gabrié (2025), *Improving the evaluation of samplers on multi-modal targets* ([arXiv:2504.08916](https://arxiv.org/abs/2504.08916)) — separation `a` (default 5.25), weights 2/3 and 1/3, variances `sigma2_max = 0.2`, `sigma2_min = 0.01`, base $\mathcal{N}(0, I_d)$, no PBC. Everything is known in closed form:
- `analytical.py`: oracle marginals, score, denoisers ($\eta_0, \eta_1, \eta_z$), velocity, and `make_oracle_bijector` / `make_oracle_sde_transport`. Oracle builders default to `gamma_fn=None` (γ ≡ 0); pass `gamma_fn` explicitly to match a trained model's interpolant.
- `train_gaussian_mixture.py`: training (velocity, velocity + denoiser, denoiser only, TSM) and the field wrappers that turn trained networks into SDE fields (`ScoreFromGamma`, `ReconstructedScoreFromDenoiser`, `OsmoticDriftFrom*`).
- `compare_sampling_methods.py`, `benchmark_dimension_scaling.py`: evaluation of $\hat Z$, SEM, ESS and wall time for oracle and trained ODE/SDE samplers.
- Scripts import each other as `scripts.gaussian_mixture.*`; run them with `uv run python -m scripts.gaussian_mixture.<name>`.

The research target is the **general case** used for particles (non-Gaussian base, e.g. uniform on the torus): full stochastic interpolant with γ(t) ≠ 0, learning both the velocity $b$ and the noise denoiser $\eta_z$. Identities that only hold for a Gaussian base or γ ≡ 0 (one-sided interpolant, score recovered from the velocity) are useful sanity checks but must not be relied on for conclusions meant to transfer to particles.

Group reference: our paper *Boltzmann generators for amorphous particle systems* shows that ODE discretisation biases the CNF log-weights even when the divergence is computed exactly with autodiff.

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
