"""Single source of truth for the SDE vs ODE study (see tmp/sde_vs_ode/PLAN.md)."""

import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK_DIR = ROOT / "tmp" / "sde_vs_ode"
REPORT_DIR = WORK_DIR / "report"

TARGET = {"a": 5.25, "sigma2_max": 0.2, "sigma2_min": 0.01, "weight": 2.0 / 3.0}
GAMMA_EXPR = "0.5 * (1-t) * t"

DIMS = [2, 4, 8, 16, 32, 64]
STEPS = [25, 50, 100, 200, 500, 1000]
VARIANTS = {"si": 0.0, "si_tsm": 1.0}
MODELS = [("si_tsm", 0), ("si_tsm", 1), ("si", 0)]
MAIN_NSTEPS = 100_000

ORACLE_EPSILONS = [0.03, 0.1, 0.3, 1.0, 3.0]
EPSILONS = [0.03, 0.1, 0.3, 1.0]
SCORE_TREATMENTS = ["raw", "two_sided", "gemini"]
MAIN_TREATMENT = "two_sided"
CONTROL_TREATMENTS = ["raw", "gemini"]
CONTROL_EPSILONS = [0.3, 1.0]
CONTROL_SEED = 0
HYBRID_EPSILONS = [0.1, 0.3, 1.0]
HYBRID_MODEL = ("si_tsm", 0)

N_REPLICATES = 5
N_PER_REPLICATE = 2000

BLEND_SIGMA = 0.08
KAPPA = 0.67

NETWORK = {"type": "mlp", "width": 256, "depth": 3, "time_embedding_dim": 32}
BATCH_SIZE = 1024
PEAK_LR = 1e-3
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.0
LOG_FREQ = 500
NUM_CHECKPOINTS = 4

CALIBRATION_DIMS = [8, 64]
CALIBRATION_NSTEPS = 100_000
CALIBRATION_NUM_CHECKPOINTS = 10
CALIBRATION_EXTRA_LR = {64: [3e-4]}


def gamma_fn(t):
    return 0.5 * (1.0 - t) * t


def training_config(
    d: int,
    variant: str,
    seed: int,
    nsteps: int,
    ckpt_path: Path,
    peak_lr: float = PEAK_LR,
    num_checkpoints: int = NUM_CHECKPOINTS,
) -> dict:
    """Full config for ``scripts.gaussian_mixture.train_gaussian_mixture`` (no defaults left implicit).

    Schedule-free AdamW has no budget-dependent decay, so a snapshot at step k of a longer run
    is the same model a k-step run would give; all ``num_checkpoints`` snapshots are kept.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(VARIANTS)}")
    return {
        "target": {"d": d, **TARGET},
        "data": {"num_workers": 1, "prefetch_buffer_size": 2},
        "training": {
            "nsteps": nsteps,
            "batch_size": BATCH_SIZE,
            "seed": seed,
            "loss_type": "stochastic_interpolant",
            "log_freq": LOG_FREQ,
            "ckpt_path": str(ckpt_path),
            "overwrite": True,
            "num_checkpoints": num_checkpoints,
            "max_to_keep": num_checkpoints,
            "load_from_checkpoint": None,
            "partial_dofs": None,
        },
        "optimizer": {
            "type": "schedule_free_adamw",
            "lr_schedule": str(peak_lr),
            "warmup_steps": min(WARMUP_STEPS, nsteps // 10),
            "weight_decay": WEIGHT_DECAY,
            "b1": 0.9,
            "b2": 0.999,
            "clip": 1.0,
        },
        "velocity": copy.deepcopy(NETWORK),
        "solver": {"type": "euler", "atol": 1e-5, "rtol": 1e-5, "solver_steps": 100},
        "stochastic_interpolant": {
            "interpolant_scheduler": "t",
            "noise_scheduler": GAMMA_EXPR,
            "learn_denoiser": True,
            "denoiser_weight": 1.0,
            "denoiser_only": False,
            "tsm_weight": VARIANTS[variant],
        },
        "sde": {"noise_schedule": "1.0", "noise_ratio": None, "steps": 100},
        "callbacks": {
            "ess": {"enabled": False, "freq": 250, "samples": 1000},
            "tensorboard": {"enabled": False, "freq": 100, "log_dir": "tmp/tb_logs"},
            "profile": {"enabled": False, "log_dir": "tmp/profiles", "warmup": 50, "steps": None},
        },
    }
