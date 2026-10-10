"""Phase 1 summary: for every calibration run and snapshot, field errors and a quick sampling check.

Sampling check per snapshot: learnt ODE and learnt SDE (blended score, ε = 1), M = 200, one
replicate of n = 2000. Prints a table and writes ``tmp/sde_vs_ode/calibration/summary.json``.
Requires ``field_diagnostics`` to have been run on each calibration run first.
"""

import json

import jax
import numpy as np

from scripts.gaussian_mixture.sde_vs_ode import configs
from scripts.gaussian_mixture.sde_vs_ode.field_diagnostics import T_GRID
from scripts.gaussian_mixture.sde_vs_ode.fields import (
    Problem,
    compiled_sampler,
    learnt_fields,
    ode_transport,
    restore_model,
    sde_transport,
)
from scripts.gaussian_mixture.sde_vs_ode.metrics import replicate_metrics
from scripts.gaussian_mixture.sde_vs_ode.train_all import checkpoint_dir, planned_runs, run_name

M_CHECK = 200
T_REPORT = [1e-3, 0.05, 0.5, 0.95, 0.999]


def main():
    summary = []
    for r in planned_runs("calibration", None):
        ckpt = checkpoint_dir("calibration", r)
        if ckpt is None:
            print(f"{run_name(r)}: MISSING or FAILED run")
            summary.append({"run": run_name(r), "status": "MISSING"})
            continue
        errors = json.loads((ckpt / "field_errors.json").read_text())
        problem = Problem(r["d"])
        print(f"\n{run_name(r)}")
        print(
            "  step   | "
            + " | ".join(f"b@{t:g}" for t in T_REPORT)
            + " | "
            + " | ".join(f"eta@{t:g}" for t in T_REPORT)
            + " | s_blend@0.05 | ODE ESS, logZ | SDE ESS, logZ"
        )
        for step in sorted(errors["steps"], key=int):
            per_t = errors["steps"][step]
            rel = {f: [per_t[T_GRID.index(t)][f]["rel"] for t in T_REPORT] for f in ["b", "eta_z"]}
            s_blend = per_t[T_GRID.index(0.05)]["s_blend"]["rel"]
            model, _, _ = restore_model(ckpt, int(step))
            checks = {}
            for name, tr in [
                ("ode", ode_transport(model.velocity_field, M_CHECK)),
                ("sde", sde_transport(*learnt_fields(model, problem, configs.MAIN_TREATMENT), 1.0, M_CHECK)),
            ]:
                fn = compiled_sampler(tr, problem, configs.N_PER_REPLICATE)
                y, lw = jax.block_until_ready(fn(jax.random.key(int(step))))
                checks[name] = replicate_metrics(np.asarray(y), np.asarray(lw))
            print(
                f"  {int(step):6d} | "
                + " | ".join(f"{v:.2e}" for v in rel["b"] + rel["eta_z"])
                + f" | {s_blend:.2e}"
                + f" | {checks['ode']['ess']:.3f}, {checks['ode']['log_z']:+.3f}"
                + f" | {checks['sde']['ess']:.3f}, {checks['sde']['log_z']:+.3f}",
                flush=True,
            )
            summary.append(
                {"run": run_name(r), **r, "step": int(step), "rel_b": rel["b"], "rel_eta_z": rel["eta_z"],
                 "rel_s_blend_0.05": s_blend, "t_report": T_REPORT, "ode": checks["ode"], "sde": checks["sde"]}
            )  # fmt: skip
    (configs.WORK_DIR / "calibration" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
