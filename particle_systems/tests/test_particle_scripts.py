from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from particle_systems.evaluate_log_prob import app as evaluate_app
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
    base_samples_files = list(out_dir.glob("**/base_samples.xyz"))
    assert len(log_probs_files) > 0, "log_probs.dat not found in sampling output"
    assert len(samples_files) > 0, "samples.xyz not found in sampling output"
    assert len(base_samples_files) > 0, "base_samples.xyz not found in sampling output"


def test_sampling_with_density_hutchinson(trained_model_dir, tmp_path):
    output_path = tmp_path / "sampled_out_density_hutchinson"
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
            "--hutchinson-samples",
            "10",
        ],
    )

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"Sampling (with hutchinson) failed: {result.stdout}"

    out_dir = output_path / trained_model_dir.name
    assert out_dir.exists(), f"Expected sampling output dir {out_dir} does not exist"

    log_probs_files = list(out_dir.glob("**/log_probs.dat"))
    samples_files = list(out_dir.glob("**/samples.xyz"))
    base_samples_files = list(out_dir.glob("**/base_samples.xyz"))
    assert len(log_probs_files) > 0, "log_probs.dat not found in sampling output"
    assert len(samples_files) > 0, "samples.xyz not found in sampling output"
    assert len(base_samples_files) > 0, "base_samples.xyz not found in sampling output"


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
    base_samples_files = list(out_dir.glob("**/base_samples.xyz"))
    assert len(log_probs_files) == 0, "log_probs.dat should not exist when --ignore-density is used"
    assert len(samples_files) > 0, "samples.xyz not found in sampling output"
    assert len(base_samples_files) > 0, "base_samples.xyz not found in sampling output"


# ── evaluate_log_prob tests ──────────────────────────────────────────────────


@pytest.fixture
def sampled_with_density(trained_model_dir, tmp_path):
    """Generate samples WITH density for cross-validation against evaluate_log_prob."""
    output_path = tmp_path / "ref_samples"
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
        assert result.exit_code == 0, f"Reference sampling failed: {result.stdout}"

    out_dir = output_path / trained_model_dir.name
    run_dir = get_latest_run_dir(out_dir)
    assert run_dir is not None, f"No run directory found in {out_dir}"

    samples_dir = get_latest_run_dir(run_dir)
    assert samples_dir is not None, f"No seed directory found in {run_dir}"

    return trained_model_dir, samples_dir


def test_evaluate_log_prob(sampled_with_density, tmp_path):
    """Evaluate log-prob without filtering and compare with original forward-pass values."""
    ckpt_dir, samples_dir = sampled_with_density
    eval_output = tmp_path / "eval_out"

    result = runner.invoke(
        evaluate_app,
        [
            str(ckpt_dir),
            str(samples_dir),
            "--batch-size",
            "2",
            "--output-path",
            str(eval_output),
            "--solver-steps",
            "2",
        ],
    )

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"evaluate_log_prob failed: {result.stdout}"

    eval_dir = eval_output / ckpt_dir.name
    assert eval_dir.exists(), f"Expected output dir {eval_dir} does not exist"

    eval_lp_files = sorted(eval_dir.glob("**/log_probs.dat"))
    eval_sample_files = sorted(eval_dir.glob("**/samples.xyz"))
    assert len(eval_lp_files) > 0, "log_probs.dat not found in evaluate output"
    assert len(eval_sample_files) > 0, "samples.xyz not found in evaluate output"

    ref_lp_files = sorted(samples_dir.glob("**/log_probs.dat"))
    ref_lps = np.concatenate([np.loadtxt(f) for f in ref_lp_files])
    eval_lps = np.concatenate([np.loadtxt(f) for f in eval_lp_files])

    assert (
        ref_lps.shape == eval_lps.shape
    ), f"Log-prob shape mismatch: reference {ref_lps.shape} vs evaluated {eval_lps.shape}"
    np.testing.assert_allclose(eval_lps, ref_lps, atol=0.1, err_msg="Reverse-pass log-probs deviate from forward-pass")


def test_evaluate_log_prob_forward(sampled_with_density, tmp_path):
    """Evaluate log-prob with forward integration and compare with original values."""
    ckpt_dir, samples_dir = sampled_with_density
    eval_output = tmp_path / "eval_out_fwd_only"

    result = runner.invoke(
        evaluate_app,
        [
            str(ckpt_dir),
            str(samples_dir),
            "--batch-size",
            "2",
            "--output-path",
            str(eval_output),
            "--solver-steps",
            "2",
            "--forward-ode",
        ],
    )

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"evaluate_log_prob (forward) failed: {result.stdout}"

    eval_dir = eval_output / ckpt_dir.name
    assert eval_dir.exists(), f"Expected output dir {eval_dir} does not exist"

    eval_lp_files = sorted(eval_dir.glob("**/log_probs.dat"))
    eval_sample_files = sorted(eval_dir.glob("**/samples.xyz"))
    assert len(eval_lp_files) > 0, "log_probs.dat not found in evaluate output"
    assert len(eval_sample_files) > 0, "samples.xyz not found in evaluate output"

    ref_lp_files = sorted(samples_dir.glob("**/log_probs.dat"))
    ref_lps = np.concatenate([np.loadtxt(f) for f in ref_lp_files])
    eval_lps = np.concatenate([np.loadtxt(f) for f in eval_lp_files])

    assert (
        ref_lps.shape == eval_lps.shape
    ), f"Log-prob shape mismatch: reference {ref_lps.shape} vs evaluated {eval_lps.shape}"
    np.testing.assert_allclose(
        eval_lps, ref_lps, atol=1e-4, err_msg="Forward-pass log-probs deviate from original generated ones"
    )


def test_evaluate_log_prob_hutchinson(sampled_with_density, tmp_path):
    """Evaluate log-prob with Hutchinson estimator and compare with original values."""
    ckpt_dir, samples_dir = sampled_with_density
    eval_output = tmp_path / "eval_out_hutch"

    result = runner.invoke(
        evaluate_app,
        [
            str(ckpt_dir),
            str(samples_dir),
            "--batch-size",
            "2",
            "--output-path",
            str(eval_output),
            "--solver-steps",
            "2",
            "--hutchinson-samples",
            "10",
        ],
    )

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"evaluate_log_prob (hutchinson) failed: {result.stdout}"

    eval_dir = eval_output / ckpt_dir.name
    assert eval_dir.exists(), f"Expected output dir {eval_dir} does not exist"

    eval_lp_files = sorted(eval_dir.glob("**/log_probs.dat"))
    eval_sample_files = sorted(eval_dir.glob("**/samples.xyz"))
    assert len(eval_lp_files) > 0, "log_probs.dat not found in evaluate output"
    assert len(eval_sample_files) > 0, "samples.xyz not found in evaluate output"

    ref_lp_files = sorted(samples_dir.glob("**/log_probs.dat"))
    ref_lps = np.concatenate([np.loadtxt(f) for f in ref_lp_files])
    eval_lps = np.concatenate([np.loadtxt(f) for f in eval_lp_files])

    assert (
        ref_lps.shape == eval_lps.shape
    ), f"Log-prob shape mismatch: reference {ref_lps.shape} vs evaluated {eval_lps.shape}"
    np.testing.assert_allclose(eval_lps, ref_lps, atol=0.5, err_msg="Hutchinson log-probs deviate from forward-pass")


def test_evaluate_log_prob_with_filtering(sampled_with_density, tmp_path):
    """Evaluate log-prob with energy filtering and verify output structure."""
    ckpt_dir, samples_dir = sampled_with_density
    eval_output = tmp_path / "eval_out_filtered"

    result = runner.invoke(
        evaluate_app,
        [
            str(ckpt_dir),
            str(samples_dir),
            "--batch-size",
            "2",
            "--output-path",
            str(eval_output),
            "--solver-steps",
            "2",
            "--energy-fraction",
            "0.5",
        ],
    )

    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"evaluate_log_prob (filtered) failed: {result.stdout}"

    eval_dir = eval_output / ckpt_dir.name
    assert eval_dir.exists(), f"Expected output dir {eval_dir} does not exist"

    eval_lp_files = sorted(eval_dir.glob("**/log_probs.dat"))
    eval_sample_files = sorted(eval_dir.glob("**/samples.xyz"))
    assert len(eval_lp_files) > 0, "log_probs.dat not found in evaluate output"
    assert len(eval_sample_files) > 0, "samples.xyz not found in evaluate output"

    ref_lp_files = sorted(samples_dir.glob("**/log_probs.dat"))
    n_ref = sum(len(np.loadtxt(f).reshape(-1)) for f in ref_lp_files)
    n_eval = sum(len(np.loadtxt(f).reshape(-1)) for f in eval_lp_files)

    assert n_eval <= n_ref, f"Filtered output ({n_eval}) should have fewer samples than original ({n_ref})"
    assert n_eval > 0, "Filtered output should not be empty"


def test_evaluate_log_prob_forward_vs_backward(sampled_with_density, tmp_path):
    """Evaluate log-prob with both forward and backward integration and compare."""
    ckpt_dir, samples_dir = sampled_with_density
    eval_output_bwd = tmp_path / "eval_out_bwd"
    eval_output_fwd = tmp_path / "eval_out_fwd"

    result_bwd = runner.invoke(
        evaluate_app,
        [
            str(ckpt_dir),
            str(samples_dir),
            "--batch-size",
            "2",
            "--output-path",
            str(eval_output_bwd),
            "--solver-steps",
            "2",
        ],
    )

    if result_bwd.exit_code != 0:
        if result_bwd.exception:
            raise result_bwd.exception
        assert result_bwd.exit_code == 0, f"evaluate_log_prob (backward) failed: {result_bwd.stdout}"

    result_fwd = runner.invoke(
        evaluate_app,
        [
            str(ckpt_dir),
            str(samples_dir),
            "--batch-size",
            "2",
            "--output-path",
            str(eval_output_fwd),
            "--solver-steps",
            "2",
            "--forward-ode",
        ],
    )

    if result_fwd.exit_code != 0:
        if result_fwd.exception:
            raise result_fwd.exception
        assert result_fwd.exit_code == 0, f"evaluate_log_prob (forward) failed: {result_fwd.stdout}"

    bwd_dir = eval_output_bwd / ckpt_dir.name
    fwd_dir = eval_output_fwd / ckpt_dir.name

    bwd_lp_files = sorted(bwd_dir.glob("**/log_probs.dat"))
    fwd_lp_files = sorted(fwd_dir.glob("**/log_probs.dat"))

    assert len(bwd_lp_files) > 0, "log_probs.dat not found in backward evaluate output"
    assert len(fwd_lp_files) > 0, "log_probs.dat not found in forward evaluate output"

    bwd_lps = np.concatenate([np.loadtxt(f) for f in bwd_lp_files])
    fwd_lps = np.concatenate([np.loadtxt(f) for f in fwd_lp_files])

    assert (
        bwd_lps.shape == fwd_lps.shape
    ), f"Log-prob shape mismatch: backward {bwd_lps.shape} vs forward {fwd_lps.shape}"
    np.testing.assert_allclose(fwd_lps, bwd_lps, atol=0.1, err_msg="Forward-pass log-probs deviate from backward-pass")
