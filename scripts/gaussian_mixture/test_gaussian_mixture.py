import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from typer.testing import CliRunner

from scripts.gaussian_mixture.sample_gaussian_mixture import app as sampling_app
from scripts.gaussian_mixture.train_gaussian_mixture import app as training_app

runner = CliRunner()


def get_latest_run_dir(base_ckpt_dir: Path) -> Path:
    """Helper to find the most recently created run directory."""
    try:
        run_dirs = [d for d in base_ckpt_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda d: d.stat().st_mtime)
    except FileNotFoundError:
        return None


def test_training_and_restarting(tmp_path):
    ckpt_path = tmp_path / "ckpt_gaussian_mixture"

    # Run training for 2 steps
    result = runner.invoke(
        training_app,
        [
            "--d",
            "2",
            "--a",
            "2.0",
            "--nsteps",
            "2",
            "--batch-size",
            "4",
            "--ckpt-path",
            str(ckpt_path),
            "--ess",
            "--ess-samples",
            "10",
        ],
    )

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"Training failed: {result.stdout}"

    run_dir = get_latest_run_dir(ckpt_path)
    assert run_dir is not None, f"Could not find run directory in {ckpt_path}"
    assert (run_dir / "config.json").exists(), f"Configuration file config.json not found in {run_dir}"

    # Test restarting from checkpoint
    result_restart = runner.invoke(
        training_app,
        [
            "--d",
            "2",
            "--a",
            "2.0",
            "--nsteps",
            "4",
            "--batch-size",
            "4",
            "--ckpt-path",
            str(ckpt_path),
            "--load-from-checkpoint",
            str(run_dir),
        ],
    )

    if result_restart.exit_code != 0:
        if result_restart.exception:
            raise result_restart.exception
        assert result_restart.exit_code == 0, f"Restart training failed: {result_restart.stdout}"

    assert "Resumed from" in result_restart.stdout


@pytest.fixture
def trained_model_dir(tmp_path):
    """Runs a minimal training for the given config and returns the run dir for sampling tests."""
    ckpt_path = tmp_path / "ckpt_gaussian_mixture_sample_test"
    res = runner.invoke(
        training_app,
        [
            "--d",
            "2",
            "--a",
            "2.0",
            "--nsteps",
            "2",
            "--batch-size",
            "4",
            "--ckpt-path",
            str(ckpt_path),
        ],
    )
    if res.exit_code != 0:
        if res.exception:
            raise res.exception
        assert res.exit_code == 0, f"Fixture training failed: {res.stdout}"

    run_dir = get_latest_run_dir(ckpt_path)
    assert run_dir is not None, f"Failed to find created run dir in {ckpt_path}"
    return run_dir


def test_sampling_with_density(trained_model_dir, tmp_path):
    output_path = tmp_path / "sampled_out_density.npz"
    result = runner.invoke(
        sampling_app,
        [
            str(trained_model_dir),
            "--num-samples",
            "10",
            "--batch-size",
            "5",
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

    assert output_path.exists(), f"Expected sampling output file {output_path} does not exist"

    data = np.load(output_path)
    assert "samples" in data
    assert "log_probs" in data

    assert data["samples"].shape == (10, 2)
    assert data["log_probs"].shape == (10,)


def test_sampling_ignore_density(trained_model_dir, tmp_path):
    output_path = tmp_path / "sampled_out_ignore_density.npz"
    result = runner.invoke(
        sampling_app,
        [
            str(trained_model_dir),
            "--num-samples",
            "10",
            "--batch-size",
            "4",  # Tests batch-splitting edge case (10 % 4 != 0)
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

    assert output_path.exists(), f"Expected sampling output file {output_path} does not exist"

    data = np.load(output_path)
    assert "samples" in data
    assert "log_probs" not in data

    assert data["samples"].shape == (10, 2)
