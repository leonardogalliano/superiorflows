import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typer.testing import CliRunner

from scripts.gaussian_mixture.analytical import (
    louis_mixture_denoiser,
    louis_mixture_marginal_log_prob,
    louis_mixture_score,
    louis_mixture_velocity,
)
from scripts.gaussian_mixture.compare_sampling_methods import app as compare_app
from scripts.gaussian_mixture.sample_gaussian_mixture import app as sampling_app
from scripts.gaussian_mixture.train_gaussian_mixture import app as training_app

runner = CliRunner()

BENCHMARK_CONFIG_DIR = Path("scripts/gaussian_mixture/config")
BENCHMARK_CONFIG_FILES = sorted(list(BENCHMARK_CONFIG_DIR.glob("*.json")))


def get_latest_run_dir(base_ckpt_dir: Path) -> Path:
    """Helper to find the most recently created run directory."""
    try:
        run_dirs = [d for d in base_ckpt_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return None
        return max(run_dirs, key=lambda d: d.stat().st_mtime)
    except FileNotFoundError:
        return None


# ── Analytical Module Unit Tests ─────────────────────────────────────────────


def test_analytical_exactness():
    """Verify analytical score, denoiser, and velocity against JAX autodiff."""
    d = 3
    a = 2.0
    t = 0.35
    x = jnp.array([0.5, -1.2, 0.8])

    # 1. Score vs Autodiff Gradient of Log-Prob
    exact_score = louis_mixture_score(t, x, d=d, a=a)
    ad_score = jax.grad(lambda x_val: louis_mixture_marginal_log_prob(t, x_val, d=d, a=a))(x)
    assert jnp.allclose(exact_score, ad_score, atol=1e-5), f"Score mismatch: {exact_score} vs {ad_score}"

    # 2. Tweedie Denoiser Identity for One-Sided Linear Interpolant
    eta_0, eta_1, eta_z = louis_mixture_denoiser(t, x, d=d, a=a)
    alpha_t = 1.0 - t
    assert jnp.allclose(eta_0, -alpha_t * exact_score, atol=1e-5)

    # 3. Consistency of Velocity Field
    dot_alpha_t = -1.0
    dot_beta_t = 1.0
    expected_v = dot_alpha_t * eta_0 + dot_beta_t * eta_1
    exact_v = louis_mixture_velocity(t, x, d=d, a=a)
    assert jnp.allclose(exact_v, expected_v, atol=1e-5)


# ── Oracle Sampling Tests ───────────────────────────────────────────────────


def test_oracle_ode_and_sde_sampling(tmp_path):
    """Test sampling using exact analytical oracles via CLI."""
    # Create a minimal config.json for oracle reference
    oracle_dir = tmp_path / "oracle_ref"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "target": {"d": 2, "a": 2.0, "sigma2_max": 0.2, "sigma2_min": 0.01, "weight": 2.0 / 3.0},
        "solver": {"type": "tsit5", "atol": 1e-5, "rtol": 1e-5, "solver_steps": None},
        "sde": {"noise_schedule": "0.1 * (1-t) * t", "steps": 50},
        "stochastic_interpolant": {"interpolant_scheduler": "t", "noise_scheduler": "None"},
        "velocity": {"type": "mlp", "width": 16, "depth": 2},
        "training": {"loss_type": "stochastic_interpolant"},
    }
    with open(oracle_dir / "config.json", "w") as f:
        json.dump(config, f)

    # 1. Oracle ODE Sampling
    ode_out = tmp_path / "oracle_ode_out.npz"
    res_ode = runner.invoke(
        sampling_app,
        [
            str(oracle_dir),
            "--transport",
            "oracle_ode",
            "--num-samples",
            "20",
            "--batch-size",
            "10",
            "--output-path",
            str(ode_out),
            "--solver-steps",
            "10",
        ],
    )
    assert res_ode.exit_code == 0, f"Oracle ODE sampling failed: {res_ode.stdout}"
    data_ode = np.load(ode_out)
    assert data_ode["samples"].shape == (20, 2)
    assert "log_weights" in data_ode

    # 2. Oracle SDE Sampling
    sde_out = tmp_path / "oracle_sde_out.npz"
    res_sde = runner.invoke(
        sampling_app,
        [
            str(oracle_dir),
            "--transport",
            "oracle_sde",
            "--num-samples",
            "20",
            "--batch-size",
            "10",
            "--output-path",
            str(sde_out),
            "--sde-steps",
            "20",
        ],
    )
    assert res_sde.exit_code == 0, f"Oracle SDE sampling failed: {res_sde.stdout}"
    data_sde = np.load(sde_out)
    assert data_sde["samples"].shape == (20, 2)
    assert "log_weights" in data_sde
    assert "importance_weights" in data_sde


# ── Training & Sampling Integration Tests ────────────────────────────────────


def test_training_stochastic_interpolant_denoiser_only(tmp_path):
    """Test training denoiser-only model and sampling via both ODE and SDE."""
    ckpt_path = tmp_path / "ckpt_si_denoiser"

    # 1. Train denoiser-only for 2 steps
    config_file = tmp_path / "si_denoiser_config.json"
    config = {
        "target": {"d": 2, "a": 2.0},
        "training": {
            "nsteps": 2,
            "batch_size": 4,
            "loss_type": "stochastic_interpolant",
            "ckpt_path": str(ckpt_path),
        },
        "stochastic_interpolant": {
            "interpolant_scheduler": "t",
            "noise_scheduler": "None",
            "denoiser_only": True,
        },
        "sde": {"noise_schedule": "0.1 * (1-t) * t", "steps": 10},
    }
    with open(config_file, "w") as f:
        json.dump(config, f)

    res_train = runner.invoke(training_app, ["--config", str(config_file)])
    assert res_train.exit_code == 0, f"SI denoiser training failed: {res_train.stdout}"

    run_dir = get_latest_run_dir(ckpt_path)
    assert run_dir is not None

    # 2. ODE Sampling from denoiser model
    out_ode = tmp_path / "si_ode_samples.npz"
    res_ode = runner.invoke(
        sampling_app,
        [
            str(run_dir),
            "--transport",
            "ode",
            "--num-samples",
            "8",
            "--batch-size",
            "4",
            "--output-path",
            str(out_ode),
            "--solver-steps",
            "4",
        ],
    )
    assert res_ode.exit_code == 0, f"ODE sampling failed: {res_ode.stdout}"
    data_ode = np.load(out_ode)
    assert data_ode["samples"].shape == (8, 2)
    assert "log_weights" in data_ode

    # 3. SDE Sampling from denoiser model
    out_sde = tmp_path / "si_sde_samples.npz"
    res_sde = runner.invoke(
        sampling_app,
        [
            str(run_dir),
            "--transport",
            "sde",
            "--num-samples",
            "8",
            "--batch-size",
            "4",
            "--output-path",
            str(out_sde),
            "--sde-steps",
            "10",
        ],
    )
    assert res_sde.exit_code == 0, f"SDE sampling failed: {res_sde.stdout}"
    data_sde = np.load(out_sde)
    assert data_sde["samples"].shape == (8, 2)
    assert "log_weights" in data_sde
    assert "importance_weights" in data_sde


def test_training_stochastic_interpolant_joint_denoiser(tmp_path):
    """Test training joint velocity+denoiser model with gamma and SDE sampling."""
    ckpt_path = tmp_path / "ckpt_si_joint"

    config_file = tmp_path / "si_joint_config.json"
    config = {
        "target": {"d": 2, "a": 2.0},
        "training": {
            "nsteps": 2,
            "batch_size": 4,
            "loss_type": "stochastic_interpolant",
            "ckpt_path": str(ckpt_path),
        },
        "stochastic_interpolant": {
            "interpolant_scheduler": "t",
            "noise_scheduler": "0.5 * (1-t) * t",
            "learn_denoiser": True,
        },
        "sde": {"noise_schedule": "0.1 * (1-t) * t", "steps": 10},
    }
    with open(config_file, "w") as f:
        json.dump(config, f)

    res_train = runner.invoke(training_app, ["--config", str(config_file)])
    assert res_train.exit_code == 0, f"SI joint training failed: {res_train.stdout}"

    run_dir = get_latest_run_dir(ckpt_path)
    assert run_dir is not None

    out_sde = tmp_path / "joint_sde_samples.npz"
    res_sde = runner.invoke(
        sampling_app,
        [
            str(run_dir),
            "--transport",
            "sde",
            "--num-samples",
            "6",
            "--batch-size",
            "3",
            "--output-path",
            str(out_sde),
            "--sde-steps",
            "8",
        ],
    )
    assert res_sde.exit_code == 0, f"Joint SDE sampling failed: {res_sde.stdout}"
    data_sde = np.load(out_sde)
    assert data_sde["samples"].shape == (6, 2)
    assert "log_weights" in data_sde


def test_training_and_restarting(tmp_path):
    """Test checkpoint save and restart for MLE training."""
    ckpt_path = tmp_path / "ckpt_gaussian_mixture"

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

    assert result.exit_code == 0, f"Training failed: {result.stdout}"

    run_dir = get_latest_run_dir(ckpt_path)
    assert run_dir is not None, f"Could not find run directory in {ckpt_path}"
    assert (run_dir / "config.json").exists(), f"Configuration file config.json not found in {run_dir}"

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

    assert result.exit_code == 0, f"Sampling (with density) failed: {result.stdout}"
    assert output_path.exists(), f"Expected sampling output file {output_path} does not exist"

    data = np.load(output_path)
    assert "samples" in data
    assert "log_weights" in data

    assert data["samples"].shape == (10, 2)
    assert data["log_weights"].shape == (10,)


def test_sampling_ignore_density(trained_model_dir, tmp_path):
    output_path = tmp_path / "sampled_out_ignore_density.npz"
    result = runner.invoke(
        sampling_app,
        [
            str(trained_model_dir),
            "--num-samples",
            "10",
            "--batch-size",
            "4",
            "--output-path",
            str(output_path),
            "--solver-steps",
            "2",
            "--ignore-density",
        ],
    )

    assert result.exit_code == 0, f"Sampling (ignore density) failed: {result.stdout}"
    assert output_path.exists(), f"Expected sampling output file {output_path} does not exist"

    data = np.load(output_path)
    assert "samples" in data
    assert "log_weights" not in data

    assert data["samples"].shape == (10, 2)


def test_discrete_realnvp_training_and_sampling(tmp_path):
    """Test training RealNVP discrete flow and sampling with exact density."""
    ckpt_path = tmp_path / "ckpt_realnvp"

    config_file = tmp_path / "realnvp_config.json"
    config = {
        "target": {"d": 2, "a": 2.0},
        "training": {
            "nsteps": 2,
            "batch_size": 4,
            "loss_type": "maximum_likelihood",
            "ckpt_path": str(ckpt_path),
        },
        "velocity": {
            "type": "realnvp",
            "num_layers": 2,
            "width": 16,
            "depth": 2,
        },
    }
    with open(config_file, "w") as f:
        json.dump(config, f)

    res_train = runner.invoke(training_app, ["--config", str(config_file)])
    assert res_train.exit_code == 0, f"RealNVP training failed: {res_train.stdout}"

    run_dir = get_latest_run_dir(ckpt_path)
    assert run_dir is not None

    out_path = tmp_path / "realnvp_samples.npz"
    res_sample = runner.invoke(
        sampling_app,
        [
            str(run_dir),
            "--transport",
            "discrete",
            "--num-samples",
            "8",
            "--batch-size",
            "4",
            "--output-path",
            str(out_path),
        ],
    )
    assert res_sample.exit_code == 0, f"RealNVP sampling failed: {res_sample.stdout}"
    data = np.load(out_path)
    assert data["samples"].shape == (8, 2)
    assert "log_weights" in data
    assert "importance_weights" in data


def test_sampling_with_noise_ratio(tmp_path):
    """Verify SDE sampling with direct noise_ratio override and osmotic drift."""
    runner = CliRunner()
    ckpt_path = tmp_path / "ckpt_si_ratio"

    config_file = tmp_path / "si_ratio_config.json"
    config = {
        "target": {"d": 2, "a": 2.0},
        "training": {
            "nsteps": 2,
            "batch_size": 4,
            "loss_type": "stochastic_interpolant",
            "ckpt_path": str(ckpt_path),
        },
        "stochastic_interpolant": {
            "denoiser_only": True,
            "interpolant_scheduler": "t",
            "noise_scheduler": "None",
        },
        "sde": {
            "noise_ratio": "0.1 * t",
            "steps": 10,
        },
    }
    with open(config_file, "w") as f:
        json.dump(config, f)

    res_train = runner.invoke(training_app, ["--config", str(config_file)])
    assert res_train.exit_code == 0, f"Training failed: {res_train.stdout}"

    run_dir = get_latest_run_dir(ckpt_path)
    assert run_dir is not None

    out_path = tmp_path / "sde_ratio_samples.npz"
    res_sample = runner.invoke(
        sampling_app,
        [
            str(run_dir),
            "--transport",
            "sde",
            "--num-samples",
            "8",
            "--batch-size",
            "4",
            "--output-path",
            str(out_path),
        ],
    )
    assert res_sample.exit_code == 0, f"SDE sampling with noise_ratio failed: {res_sample.stdout}"
    data = np.load(out_path)
    assert data["samples"].shape == (8, 2)
    assert not np.isnan(data["samples"]).any()
    assert "log_weights" in data
    assert "importance_weights" in data

    # Test CLI --noise-ratio override
    out_path_override = tmp_path / "sde_ratio_override.npz"
    res_override = runner.invoke(
        sampling_app,
        [
            str(run_dir),
            "--transport",
            "sde",
            "--noise-ratio",
            "0.2 * t",
            "--num-samples",
            "4",
            "--output-path",
            str(out_path_override),
        ],
    )
    assert res_override.exit_code == 0, f"CLI override failed: {res_override.stdout}"


# ── Benchmark Configuration and Comparison Suite Tests ───────────────────────


@pytest.mark.parametrize("config_path", BENCHMARK_CONFIG_FILES, ids=[p.stem for p in BENCHMARK_CONFIG_FILES])
def test_benchmark_configs_training(config_path, tmp_path):
    """Verify that each benchmark configuration file can train and checkpoint without errors."""
    ckpt_path = tmp_path / config_path.stem
    result = runner.invoke(
        training_app,
        [
            "--config",
            str(config_path),
            "--nsteps",
            "2",
            "--batch-size",
            "4",
            "--ckpt-path",
            str(ckpt_path),
        ],
    )
    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"Training failed for {config_path.name}: {result.stdout}"

    run_dir = get_latest_run_dir(ckpt_path)
    assert run_dir is not None, f"No run directory found in {ckpt_path}"
    assert (run_dir / "config.json").exists(), f"Configuration file not found in {run_dir}"


def test_compare_sampling_methods_oracle_only(tmp_path):
    """Verify comparison suite execution for analytical oracles without trained models."""
    out_json = tmp_path / "oracle_results.json"
    result = runner.invoke(
        compare_app,
        [
            "--steps-list",
            "10,20",
            "--num-samples",
            "10",
            "--batch-size",
            "5",
            "--output-json",
            str(out_json),
        ],
    )
    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"compare_sampling_methods failed: {result.stdout}"

    assert out_json.exists(), f"Output JSON file {out_json} was not created"
    with open(out_json) as f:
        data = json.load(f)

    assert "summary_table" in data
    assert "sde_scaling" in data
    assert "Oracle: ODE (Exact)" in data["summary_table"]
    assert "Oracle: SDE (Exact)" in data["summary_table"]
    assert "steps_10" in data["sde_scaling"]
    assert "steps_20" in data["sde_scaling"]


def test_compare_sampling_methods_with_trained_models(tmp_path):
    """Verify comparison suite execution with trained ODE and SDE checkpoints."""
    bench_dir = tmp_path / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Train Run 1 (ODE, velocity-only, gamma=0)
    cfg1 = BENCHMARK_CONFIG_DIR / "01_ode_velocity_no_gamma.json"
    ckpt1_base = bench_dir / "01_ode_velocity_no_gamma"
    res1 = runner.invoke(
        training_app,
        ["--config", str(cfg1), "--nsteps", "2", "--batch-size", "4", "--ckpt-path", str(ckpt1_base)],
    )
    assert res1.exit_code == 0, f"Run 1 training failed: {res1.stdout}"
    run1_dir = get_latest_run_dir(ckpt1_base)

    # Train Run 4 (SDE, one-sided denoiser-only)
    cfg4 = BENCHMARK_CONFIG_DIR / "04_sde_one_sided_denoiser_only.json"
    ckpt4_base = bench_dir / "04_sde_one_sided_denoiser_only"
    res4 = runner.invoke(
        training_app,
        ["--config", str(cfg4), "--nsteps", "2", "--batch-size", "4", "--ckpt-path", str(ckpt4_base)],
    )
    assert res4.exit_code == 0, f"Run 4 training failed: {res4.stdout}"
    run4_dir = get_latest_run_dir(ckpt4_base)

    out_json = tmp_path / "comparison_results.json"
    result = runner.invoke(
        compare_app,
        [
            "--ckpt-run1",
            str(run1_dir),
            "--ckpt-run4",
            str(run4_dir),
            "--steps-list",
            "10,20",
            "--num-samples",
            "10",
            "--batch-size",
            "5",
            "--output-json",
            str(out_json),
        ],
    )
    if result.exit_code != 0:
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, f"compare_sampling_methods failed: {result.stdout}"

    assert out_json.exists(), f"Output JSON file {out_json} was not created"
    with open(out_json) as f:
        data = json.load(f)

    assert "summary_table" in data
    assert "field_accuracy" in data
    assert "sde_scaling" in data
    assert "Oracle: ODE (Exact)" in data["summary_table"]
    assert "Run 1: ODE (Velocity-only, γ=0)" in data["summary_table"]
    assert "Run 4: SDE (Score-based, eps=1)" in data["summary_table"]
    assert "Run 1" in data["field_accuracy"]
    assert "steps_10" in data["sde_scaling"]
    assert "trained_run4_sde" in data["sde_scaling"]["steps_10"]
