---
trigger: always_on
---

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