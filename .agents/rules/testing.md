---
trigger: always_on
---

## Testing

### Test locations
- `tests/` — core library tests (`test_flow.py`, `test_trainer.py`, `test_partial.py`, `test_sde.py`, etc.)
- `particle_systems/tests/` — domain-specific tests for the particle package
- `scripts/gaussian_mixture/test_gaussian_mixture.py` — integration tests for Gaussian mixtures, RealNVP, ODE flows, and partial DOF training/sampling. **Not collected by default** (`testpaths = ["tests", "particle_systems"]`), so neither plain `uv run pytest` nor CI runs it; invoke it explicitly.

### Running tests
```bash
uv run pytest --cov=src --cov=particle_systems --cov-report=xml   # full suite, as in CI
uv run pytest tests/test_flow.py -x                               # single file, stop at first failure
uv run pytest tests/test_flow.py -k "hutchinson"                  # filter by name
uv run pytest --benchmark-skip                                    # skip slow pytest-benchmark tests
uv run pytest scripts/gaussian_mixture/test_gaussian_mixture.py   # integration tests (not in testpaths)
```
`addopts` includes `--benchmark-autosave`, which writes to `.benchmarks/` (gitignored).

### Troubleshooting: `ModuleNotFoundError: No module named 'superiorflows'`
The repo lives in an iCloud-synced folder. iCloud sets the macOS `hidden` file flag on files in `.venv`, and recent Python versions skip hidden `.pth` files, so the editable install silently disappears. Fix with:
```bash
chflags -R nohidden .venv
```

### Test patterns
- **Fixtures via `pytest.fixture`** with parametrisation over config files (`@pytest.fixture(params=CONFIG_FILES)`).
- **CLI script tests** use `typer.testing.CliRunner` to invoke the Typer apps programmatically.
- **Checkpoint restart tests**: run short training → find latest run dir → restart with `--load-from-checkpoint`.
- **Performance tests** use `pytest-benchmark` with `benchmark(fn)` and `.block_until_ready()`.
- Temporary directories for checkpoints use `tmp_path` (pytest built-in) or the gitignored `tmp/` directory.

### What CI runs
GitHub Actions matrix: `{ubuntu-latest, macos-latest}` × `{python 3.12, 3.13}`. Installs via `uv sync`, runs the full-suite command above, uploads coverage to Codecov.

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
