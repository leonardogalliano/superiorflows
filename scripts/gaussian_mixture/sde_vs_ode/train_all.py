"""Train the models of the SDE vs ODE study (phase 1 calibration or phase 2 main runs).

Each run is a separate process running ``scripts.gaussian_mixture.train_gaussian_mixture`` with a
fully explicit config written to ``tmp/sde_vs_ode/<phase>/<run>/config.json``; stdout goes to
``train.log`` next to it. A run whose final checkpoint exists is skipped (resumable). Nothing is
started without ``--run``: by default the planned runs are only listed.

Calibration: SI + TSM, seed 0, one ``CALIBRATION_NSTEPS`` run per (d, peak lr) with a snapshot every
``CALIBRATION_NSTEPS / CALIBRATION_NUM_CHECKPOINTS`` steps. With schedule-free AdamW each snapshot
is what a shorter run would have produced, so one run covers every candidate budget.

Main: every (d, variant, seed) at the chosen budget; runs already covered by a calibration run
(same d, variant, seed, peak lr, with a snapshot at the budget) are not retrained.

Examples::

    uv run python -m scripts.gaussian_mixture.sde_vs_ode.train_all --phase calibration --run
    uv run python -m scripts.gaussian_mixture.sde_vs_ode.train_all --phase main --nsteps 50000 --run
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer

from scripts.gaussian_mixture.sde_vs_ode import configs

app = typer.Typer(pretty_exceptions_show_locals=False)


def run_name(r: dict) -> str:
    return f"d{r['d']}_{r['variant']}_lr{r['lr']:g}_n{r['nsteps']}_s{r['seed']}"


def planned_runs(phase: str, nsteps: int | None) -> list[dict]:
    if phase == "calibration":
        return [
            {"d": d, "variant": "si_tsm", "seed": 0, "nsteps": configs.CALIBRATION_NSTEPS, "lr": lr}
            for d in configs.CALIBRATION_DIMS
            for lr in [configs.PEAK_LR] + configs.CALIBRATION_EXTRA_LR.get(d, [])
        ]
    if phase == "main":
        if nsteps is None:
            raise typer.BadParameter("--nsteps is required for the main phase (chosen from the calibration).")
        return [
            {"d": d, "variant": v, "seed": s, "nsteps": nsteps, "lr": configs.PEAK_LR}
            for d in configs.DIMS
            for v, s in configs.MODELS
        ]
    raise typer.BadParameter(f"Unknown phase '{phase}'. Use 'calibration' or 'main'.")


def checkpoint_dir(phase: str, r: dict) -> Path | None:
    """Inner checkpoint directory (created by the training script) of a finished run, or None."""
    run_dir = configs.WORK_DIR / phase / run_name(r)
    status = run_dir / "status.json"
    if not status.exists() or json.loads(status.read_text())["status"] != "OK":
        return None
    inner = [p for p in (run_dir / "ckpt").glob("*") if (p / "config.json").exists()]
    return inner[0] if len(inner) == 1 else None


def find_model(d: int, variant: str, seed: int, nsteps: int, lr: float = configs.PEAK_LR) -> Path | None:
    """Checkpoint directory holding a snapshot at ``nsteps`` for this model, from main or calibration."""
    main = checkpoint_dir("main", {"d": d, "variant": variant, "seed": seed, "nsteps": nsteps, "lr": lr})
    if main is not None:
        return main
    cal = {"d": d, "variant": variant, "seed": seed, "nsteps": configs.CALIBRATION_NSTEPS, "lr": lr}
    cal_dir = checkpoint_dir("calibration", cal)
    if cal_dir is not None and (cal_dir / str(nsteps)).is_dir():
        return cal_dir
    return None


def train_one(phase: str, r: dict) -> str:
    run_dir = configs.WORK_DIR / phase / run_name(r)
    run_dir.mkdir(parents=True, exist_ok=True)
    n_ckpt = configs.CALIBRATION_NUM_CHECKPOINTS if phase == "calibration" else configs.NUM_CHECKPOINTS
    cfg = configs.training_config(
        r["d"], r["variant"], r["seed"], r["nsteps"], run_dir / "ckpt", peak_lr=r["lr"], num_checkpoints=n_ckpt
    )
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    if sys.platform == "darwin":
        # The repo lives in iCloud, which re-hides files in .venv within minutes; hidden .pth files
        # are skipped by Python and the editable install disappears (see .agents/rules/testing.md).
        subprocess.run(["chflags", "-R", "nohidden", str(configs.ROOT / ".venv")], check=False)
    cmd = [sys.executable, "-m", "scripts.gaussian_mixture.train_gaussian_mixture", "--config", str(cfg_path)]
    t0 = time.time()
    with open(run_dir / "train.log", "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=configs.ROOT)
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    (run_dir / "status.json").write_text(
        json.dumps({"status": status, "returncode": proc.returncode, "wall_s": time.time() - t0})
    )
    message = f"{run_name(r)}: {status} in {time.time() - t0:.0f} s"
    print(f"        {message}", flush=True)
    return message


@app.command()
def main(
    phase: str = typer.Option(..., help="'calibration' or 'main'."),
    nsteps: int | None = typer.Option(None, help="Training steps for the main phase."),
    run: bool = typer.Option(False, "--run/--dry-run", help="Actually train (default: only list)."),
    only: str | None = typer.Option(None, help="Only runs whose name contains this substring."),
    parallel: int = typer.Option(1, help="Number of trainings run at the same time."),
):
    runs = [r for r in planned_runs(phase, nsteps) if only is None or only in run_name(r)]
    print(f"{len(runs)} planned runs in {configs.WORK_DIR / phase}")
    todo = []
    for r in runs:
        if phase == "main":
            done = find_model(r["d"], r["variant"], r["seed"], r["nsteps"], r["lr"]) is not None
        else:
            done = checkpoint_dir(phase, r) is not None
        print(f"  {'DONE' if done else 'TODO'}  {run_name(r)}")
        if not done:
            todo.append(r)
    if not run or not todo:
        return
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(lambda r: train_one(phase, r), todo))


if __name__ == "__main__":
    app()
