"""Errors of learnt fields against the oracle across time, for every saved training step.

For each checkpoint step (50%, 75%, 100% of training) and each t in ``T_GRID`` (including the
SDE grid points 1/M and 1 − 1/M actually used), draws x_t from the exact interpolant law and
reports mean squared errors (and errors relative to the oracle's mean squared norm) of
``b``, ``η_z``, and every score treatment in ``configs.SCORE_TREATMENTS``. Output: one JSON per run.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import typer

from scripts.gaussian_mixture.sde_vs_ode import configs
from scripts.gaussian_mixture.sde_vs_ode.fields import (
    Problem,
    learnt_fields,
    oracle_denoiser,
    oracle_fields,
    restore_model,
)

app = typer.Typer(pretty_exceptions_show_locals=False)

BASE_T = [1e-4, 1e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1 - 1e-4]
T_GRID = sorted(set(BASE_T + [1.0 / m for m in configs.STEPS] + [1.0 - 1.0 / m for m in configs.STEPS]))
N_POINTS = 2000


def sample_xt(problem: Problem, t: float, key):
    k0, k1, kz = jax.random.split(key, 3)
    x0 = jax.random.normal(k0, (N_POINTS, problem.d))
    x1 = jax.vmap(problem.target.sample)(jax.random.split(k1, N_POINTS))
    z = jax.random.normal(kz, (N_POINTS, problem.d))
    return (1 - t) * x0 + t * x1 + configs.gamma_fn(t) * z


def errors_at(fields_learnt: dict, fields_oracle: dict, t: float, x) -> dict:
    out = {}
    for name, f_learnt in fields_learnt.items():
        ref = jax.vmap(lambda xx: fields_oracle[name](t, xx))(x)
        pred = jax.vmap(lambda xx: f_learnt(t, xx))(x)
        mse = float(jnp.mean(jnp.sum((pred - ref) ** 2, axis=-1)))
        ref_sq = float(jnp.mean(jnp.sum(ref**2, axis=-1)))
        finite = bool(jnp.all(jnp.isfinite(pred)))
        out[name] = {"mse": mse, "ref_mean_sq": ref_sq, "rel": mse / ref_sq if ref_sq > 0 else float("nan")}
        out[name]["all_finite"] = finite
    return out


def diagnose(run_dir: Path) -> dict:
    manager = ocp.CheckpointManager(run_dir.resolve(), item_names=("model", "optimizer", "metadata"))
    steps = sorted(manager.all_steps())
    _, cfg, _ = restore_model(run_dir)
    problem = Problem(cfg["target"]["d"])
    b_or, s_or = oracle_fields(problem)
    oracle = {"b": b_or, "eta_z": oracle_denoiser(problem), **{f"s_{tr}": s_or for tr in configs.SCORE_TREATMENTS}}
    result = {"run_dir": str(run_dir), "d": problem.d, "config": cfg, "t_grid": T_GRID, "steps": {}}
    for step in steps:
        model, _, _ = restore_model(run_dir, step)
        learnt = {"b": model.velocity_field, "eta_z": model.denoiser}
        for treatment in configs.SCORE_TREATMENTS:
            learnt[f"s_{treatment}"] = learnt_fields(model, problem, treatment)[1]
        per_t = []
        for i, t in enumerate(T_GRID):
            x = sample_xt(problem, t, jax.random.fold_in(jax.random.key(1234), i))
            per_t.append(errors_at(learnt, oracle, t, x))
        result["steps"][str(step)] = per_t
    return result


@app.command()
def main(
    run_dirs: list[Path] = typer.Argument(..., help="Run directories containing config.json and checkpoints."),
):
    for run_dir in run_dirs:
        out = run_dir / "field_errors.json"
        res = diagnose(run_dir)
        out.write_text(json.dumps(res, indent=1, default=float))
        last = res["steps"][max(res["steps"], key=int)]
        i_mid = T_GRID.index(0.5)
        i_first = T_GRID.index(1.0 / max(configs.STEPS))
        print(
            f"{run_dir.name}: steps={list(res['steps'])}; at t=0.5 rel. error b={last[i_mid]['b']['rel']:.3g}, "
            f"eta_z={last[i_mid]['eta_z']['rel']:.3g}; at t=1/{max(configs.STEPS)}: "
            f"s_raw={last[i_first]['s_raw']['rel']:.3g}, s_{configs.MAIN_TREATMENT}="
            f"{last[i_first]['s_' + configs.MAIN_TREATMENT]['rel']:.3g}"
        )
        np.testing.assert_equal(len(last), len(T_GRID))


if __name__ == "__main__":
    app()
