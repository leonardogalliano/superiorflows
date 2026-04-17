from pathlib import Path

import pytest
from typer.testing import CliRunner

from particle_systems.sampling_particles import app as sampling_app
from particle_systems.training_particles import app as training_app

runner = CliRunner()

CONFIG_DIR = Path("particle_systems") / "config"
CONFIG_FILES = sorted(list(CONFIG_DIR.glob("*.json")))


def get_latest_run_dir(base_ckpt_dir: Path) -> Path:
    """Helper to find the most recently created run directory."""
    try:
        run_dirs = [d for d in base_ckpt_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda d: d.stat().st_mtime)
    except FileNotFoundError:
        return None


@pytest.fixture(params=CONFIG_FILES, ids=[p.name for p in CONFIG_FILES])
def config_path(request):
    """Parametrize over all configuration files in particle_systems/config/."""
    return request.param


def test_training_and_restarting(config_path):
    # Run training for 2 steps to generate a checkpoint
    result = runner.invoke(training_app, ["--config", str(config_path)])

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"Training failed: {result.stdout}"

    # Extract checkpoint dir using assumed tmp/ckpt_particles_test or look directly in tmp/
    ckpt_root = Path("tmp") / "ckpt_particles_test"
    run_dir = get_latest_run_dir(ckpt_root)
    assert run_dir is not None, f"Could not find run directory in {ckpt_root}"
    assert (run_dir / "config.json").exists(), f"Configuration file config.json not found in {run_dir}"

    # Test restarting from checkpoint
    result_restart = runner.invoke(
        training_app,
        ["--config", str(config_path), "--load-from-checkpoint", str(run_dir), "--nsteps", "2", "--batch-size", "8"],
    )

    if result_restart.exit_code != 0:
        if result_restart.exception:
            raise result_restart.exception
        assert result_restart.exit_code == 0, f"Restart training failed: {result_restart.stdout}"

    assert "Resumed from" in result_restart.stdout


@pytest.fixture
def trained_model_dir(config_path):
    """Runs a minimal training for the given config and returns the run dir for sampling tests."""
    res = runner.invoke(training_app, ["--config", str(config_path), "--nsteps", "2", "--batch-size", "8"])
    if res.exit_code != 0:
        if res.exception:
            raise res.exception
        assert res.exit_code == 0, f"Fixture training failed: {res.stdout}"

    ckpt_root = Path("tmp") / "ckpt_particles_test"
    run_dir = get_latest_run_dir(ckpt_root)
    assert run_dir is not None, f"Failed to find created run dir in {ckpt_root}"
    return run_dir


def test_sampling_with_density(trained_model_dir, tmp_path):
    output_path = tmp_path / "sampled_out_density"
    result = runner.invoke(
        sampling_app,
        [
            str(trained_model_dir),
            "--batch-size",
            "2",
            "--num-trajectories",
            "1",
            "--output-path",
            str(output_path),
            "--solver-steps",
            "2",
        ],
    )

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"Sampling (with density) failed: {result.stdout}"

    out_dir = output_path / trained_model_dir.name
    assert out_dir.exists(), f"Expected sampling output dir {out_dir} does not exist"

    log_probs_files = list(out_dir.glob("**/log_probs.dat"))
    samples_files = list(out_dir.glob("**/samples.xyz"))
    assert len(log_probs_files) > 0, "log_probs.dat not found in sampling output"
    assert len(samples_files) > 0, "samples.xyz not found in sampling output"


def test_sampling_ignore_density(trained_model_dir, tmp_path):
    output_path = tmp_path / "sampled_out_ignore_density"
    result = runner.invoke(
        sampling_app,
        [
            str(trained_model_dir),
            "--batch-size",
            "2",
            "--num-trajectories",
            "1",
            "--output-path",
            str(output_path),
            "--solver-steps",
            "2",
            "--ignore-density",
        ],
    )

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"Sampling (ignore density) failed: {result.stdout}"

    out_dir = output_path / trained_model_dir.name
    assert out_dir.exists(), f"Expected sampling output dir {out_dir} does not exist"

    log_probs_files = list(out_dir.glob("**/log_probs.dat"))
    samples_files = list(out_dir.glob("**/samples.xyz"))
    assert len(log_probs_files) == 0, "log_probs.dat should not exist when --ignore-density is used"
    assert len(samples_files) > 0, "samples.xyz not found in sampling output"
