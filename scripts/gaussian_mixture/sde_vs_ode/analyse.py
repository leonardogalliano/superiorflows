"""Phase 5: analysis of the SDE vs ODE study, computed only from the raw JSONL rows and field-error JSONs.

Outputs in ``tmp/sde_vs_ode/report/``: ``aggregates.csv`` (every configuration), ``convergence.csv``,
``equal_cost.csv``, figures, and a printed summary.

Equal-cost model. Each configuration has 5 replicates of n = 2000 samples, a cost c (CPU seconds per sample),
a mean error m and a replicate variance v at n = 2000. With a budget of B seconds it can draw N = B / c
samples, and the RMSE is estimated as sqrt(bias^2 + v * 2000 / N), with
  ODE: bias = m (the discretisation bias is systematic and measured precisely: small variance);
  SDE: bias = 0 (Z-hat is unbiased by construction; log Z-hat has an O(1/N) Jensen bias that vanishes).
Configurations with median PSIS k-hat > 0.7 are excluded from "best configuration" selections, because
the 1/N variance scaling is not trustworthy for them; they remain in every table.
"""

import csv
import glob
import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.gaussian_mixture.sde_vs_ode import configs  # noqa: E402
from scripts.gaussian_mixture.sde_vs_ode.train_all import find_model  # noqa: E402

OUT = configs.REPORT_DIR
KEY = ("d", "M", "source", "variant", "seed", "sampler", "eps", "treatment")
TRUE_MODE1 = configs.TARGET["weight"]
KHAT_MAX = 0.7
N_REF = configs.N_PER_REPLICATE
BUDGETS = [10.0, 60.0, 600.0, 3600.0]


def load_rows():
    rows = []
    for f in sorted(glob.glob(str(configs.WORK_DIR / "samples" / "d*.jsonl"))):
        rows += [json.loads(line) for line in open(f) if line.strip()]
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in KEY)].append(r)
    agg = {}
    for k, rs in groups.items():
        ok = [r for r in rs if r["status"] == "OK"]
        a = dict(zip(KEY, k), n_replicates=len(rs), n_failed=len(rs) - len(ok))
        if ok:
            lz = np.array([r["log_z"] for r in ok])
            m1 = np.array([r["mode1_snis"] for r in ok])
            a.update(
                ess_mean=float(np.mean([r["ess"] for r in ok])),
                ess_min=float(np.min([r["ess"] for r in ok])),
                khat_median=float(np.median([r["khat"] for r in ok])),
                logz_mean=float(lz.mean()),
                logz_sd=float(lz.std(ddof=1)),
                logz_rmse=float(np.sqrt(np.mean(lz**2))),
                mode1_mean=float(m1.mean()),
                mode1_sd=float(m1.std(ddof=1)),
                mode1_rmse=float(np.sqrt(np.mean((m1 - TRUE_MODE1) ** 2))),
                mode1_unweighted=float(np.mean([r["mode1_unweighted"] for r in ok])),
                sec_per_sample=float(np.mean([r["wall_s"] for r in ok]) / ok[0]["n"]),
                network_passes=ok[0]["network_passes_per_sample"],
            )
        agg[k] = a
    return agg


def get(agg, d, M, source, variant=None, seed=None, sampler="sde", eps=None, treatment=None):
    return agg.get((d, M, source, variant, seed, sampler, eps, treatment))


def best_over_eps(agg, d, M, source, variant, seed, treatment, eps_list, metric="ess_mean"):
    cands = [(e, get(agg, d, M, source, variant, seed, "sde", e, treatment)) for e in eps_list]
    cands = [(e, a) for e, a in cands if a and "ess_mean" in a]
    return max(cands, key=lambda ea: ea[1][metric]) if cands else (None, None)


def rmse_at_budget(a, budget, observable, is_ode):
    n = budget / a["sec_per_sample"]
    if observable == "logz":
        bias, sd = (a["logz_mean"] if is_ode else 0.0), a["logz_sd"]
    else:
        bias, sd = (a["mode1_mean"] - TRUE_MODE1 if is_ode else 0.0), a["mode1_sd"]
    return math.sqrt(bias**2 + sd**2 * N_REF / n)


def equal_cost(agg):
    """Best configuration per (d, method, budget, observable), excluding k-hat > 0.7."""
    out = []
    methods = {
        "learnt ODE": dict(source="learnt", sampler="ode", variant="si_tsm", seed=0, treatment=None, eps=[None]),
        "learnt SDE (two-sided)": dict(
            source="learnt", sampler="sde", variant="si_tsm", seed=0, treatment="two_sided", eps=configs.EPSILONS
        ),
        "oracle ODE": dict(source="oracle", sampler="ode", variant=None, seed=None, treatment=None, eps=[None]),
        "oracle SDE": dict(
            source="oracle", sampler="sde", variant=None, seed=None, treatment=None, eps=configs.ORACLE_EPSILONS
        ),
    }
    for d in configs.DIMS:
        for name, m in methods.items():
            cands = [
                (M, e, get(agg, d, M, m["source"], m["variant"], m["seed"], m["sampler"], e, m["treatment"]))
                for M in configs.STEPS
                for e in m["eps"]
            ]
            cands = [(M, e, a) for M, e, a in cands if a and "ess_mean" in a]
            reliable = [(M, e, a) for M, e, a in cands if a["khat_median"] <= KHAT_MAX]
            for obs in ["logz", "mode1"]:
                for B in BUDGETS:
                    if not reliable:
                        out.append(dict(d=d, method=name, observable=obs, budget_s=B, rmse=float("nan"),
                                        M=None, eps=None, note="no configuration with k-hat <= 0.7"))  # fmt: skip
                        continue
                    best = min(reliable, key=lambda x: rmse_at_budget(x[2], B, obs, m["sampler"] == "ode"))
                    out.append(dict(d=d, method=name, observable=obs, budget_s=B, M=best[0], eps=best[1],
                                    rmse=rmse_at_budget(best[2], B, obs, m["sampler"] == "ode"),
                                    n_samples=int(B / best[2]["sec_per_sample"]), note=""))  # fmt: skip
    return out


def convergence():
    """Field-error change over the last quarter of training (flag if still dropping by > 10%)."""
    out = []
    for d in configs.DIMS:
        for variant, seed in configs.MODELS:
            path = find_model(d, variant, seed, configs.MAIN_NSTEPS)
            res = json.loads((path / "field_errors.json").read_text())
            T = np.array(res["t_grid"])
            mask = (T >= 0.01) & (T <= 0.99)
            steps = sorted(map(int, res["steps"]))
            s100, s75 = steps[-1], min(steps, key=lambda s: abs(s - 0.75 * configs.MAIN_NSTEPS))

            def integral(step, field):
                per_t = res["steps"][str(step)]
                return float(np.trapezoid([per_t[i][field]["mse"] for i in np.where(mask)[0]], T[mask]))

            row = dict(d=d, variant=variant, seed=seed)
            for f in ["b", "eta_z", "s_raw", "s_two_sided"]:
                e75, e100 = integral(s75, f), integral(s100, f)
                row[f"{f}_int_mse"] = e100
                row[f"{f}_drop_last_quarter"] = (e75 - e100) / e75
            row["converged"] = row["b_drop_last_quarter"] <= 0.10 and row["eta_z_drop_last_quarter"] <= 0.10
            out.append(row)
    return out


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in KEY, k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def figures(agg, eq):
    dims = configs.DIMS
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for ax, d in zip(axes.flat, dims):
        series = {
            "oracle ODE": [get(agg, d, M, "oracle", sampler="ode") for M in configs.STEPS],
            "oracle SDE (best ε)": [
                best_over_eps(agg, d, M, "oracle", None, None, None, configs.ORACLE_EPSILONS)[1] for M in configs.STEPS
            ],  # fmt: skip
            "learnt ODE": [get(agg, d, M, "learnt", "si_tsm", 0, "ode") for M in configs.STEPS],
            "learnt SDE two-sided (best ε)": [
                best_over_eps(agg, d, M, "learnt", "si_tsm", 0, "two_sided", configs.EPSILONS)[1] for M in configs.STEPS
            ],
            "learnt SDE raw (best ε)": [
                best_over_eps(agg, d, M, "learnt", "si_tsm", 0, "raw", configs.CONTROL_EPSILONS)[1]
                for M in configs.STEPS
            ],
            "learnt SDE Gemini (best ε)": [
                best_over_eps(agg, d, M, "learnt", "si_tsm", 0, "gemini", configs.CONTROL_EPSILONS)[1]
                for M in configs.STEPS
            ],
        }
        for (name, vals), style in zip(series.items(), ["k--", "k-", "C0--", "C0-", "C3:", "C1:"]):
            ax.plot(configs.STEPS, [v["ess_mean"] for v in vals], style, marker="o", ms=3, label=name)
        ax.axhline(1 / N_REF, color="grey", lw=0.8, ls=":")
        ax.set(xscale="log", yscale="log", ylim=(3e-4, 1.2), title=f"d = {d}", xlabel="M", ylabel="ESS / n")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("ESS vs number of steps (SI+TSM seed 0; SDE: best ε per M; dotted: 1/n)")
    fig.tight_layout()
    fig.savefig(OUT / "ess_vs_steps.png", dpi=120)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for ax, d in zip(axes.flat, dims):
        for name, vals, c in [
            ("learnt ODE", [get(agg, d, M, "learnt", "si_tsm", 0, "ode") for M in configs.STEPS], "C0"),
            (
                "learnt SDE two-sided (best ε)",
                [
                    best_over_eps(agg, d, M, "learnt", "si_tsm", 0, "two_sided", configs.EPSILONS)[1]
                    for M in configs.STEPS
                ],
                "C2",
            ),  # fmt: skip
            ("oracle ODE", [get(agg, d, M, "oracle", sampler="ode") for M in configs.STEPS], "k"),
        ]:
            ax.errorbar(configs.STEPS, [v["logz_mean"] for v in vals], yerr=[v["logz_sd"] for v in vals],
                        color=c, marker="o", ms=3, capsize=2, label=name)  # fmt: skip
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(xscale="log", title=f"d = {d}", xlabel="M", ylabel="log Ẑ (truth 0), mean ± sd over 5 replicates")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "logz_vs_steps.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, obs, label in zip(axes, ["logz", "mode1"], ["log Ẑ", "mode-1 weight"]):
        for method, c in [("learnt ODE", "C0"), ("learnt SDE (two-sided)", "C2"), ("oracle ODE", "k"),
                          ("oracle SDE", "grey")]:  # fmt: skip
            for B, ls in [(60.0, "-"), (3600.0, "--")]:
                ys = [next(r["rmse"] for r in eq if r["d"] == d and r["method"] == method and r["observable"] == obs
                           and r["budget_s"] == B) for d in dims]  # fmt: skip
                ax.plot(dims, ys, color=c, ls=ls, marker="o", ms=3, label=f"{method}, {B:.0f} s")
        ax.set(xscale="log", yscale="log", xlabel="d", ylabel=f"RMSE of {label}", title=f"Equal-cost RMSE: {label}")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "equal_cost_rmse.png", dpi=120)


def print_summary(agg, eq, conv):
    print("== Failures: every configuration has 5 replicates; failed replicates per d:")
    for d in configs.DIMS:
        print(f"  d={d}: {sum(a['n_failed'] for k, a in agg.items() if k[0] == d)}")

    print("\n== Convergence (integrated field MSE over t in [0.01, 0.99]; drop over the last quarter of training)")
    for r in conv:
        print(f"  d={r['d']:>2} {r['variant']:6s} s{r['seed']}:"
              f" b {r['b_int_mse']:.3g} ({100*r['b_drop_last_quarter']:+.0f}%),"
              f" eta_z {r['eta_z_int_mse']:.3g} ({100*r['eta_z_drop_last_quarter']:+.0f}%),"
              f" s two-sided {r['s_two_sided_int_mse']:.3g}, s raw {r['s_raw_int_mse']:.3g}"
              f" -> {'converged' if r['converged'] else 'NOT CONVERGED'}")  # fmt: skip

    print("\n== Equal-cost RMSE (best reliable configuration per method; k-hat <= 0.7)")
    for obs in ["logz", "mode1"]:
        for B in BUDGETS:
            print(f"  {obs}, budget {B:.0f} s:")
            for d in configs.DIMS:
                cells = []
                for method in ["learnt ODE", "learnt SDE (two-sided)", "oracle ODE", "oracle SDE"]:
                    r = next(x for x in eq if x["d"] == d and x["method"] == method and x["observable"] == obs
                             and x["budget_s"] == B)  # fmt: skip
                    cells.append(f"{method}: {r['rmse']:.4f} (M={r['M']}, eps={r['eps']})" if not r["note"]
                                 else f"{method}: {r['note']}")  # fmt: skip
                print(f"    d={d:>2}: " + " | ".join(cells))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    agg = aggregate(rows)
    write_csv(OUT / "aggregates.csv", list(agg.values()))
    eq = equal_cost(agg)
    write_csv(OUT / "equal_cost.csv", eq)
    conv = convergence()
    write_csv(OUT / "convergence.csv", conv)
    figures(agg, eq)
    print_summary(agg, eq, conv)
    print(f"\n{len(rows)} rows, {len(agg)} configurations; outputs in {OUT}")


if __name__ == "__main__":
    main()
