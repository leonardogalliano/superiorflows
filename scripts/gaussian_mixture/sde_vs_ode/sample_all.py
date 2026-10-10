"""Phase 4: sampling sweep. One JSONL row per (configuration, replicate), append-only and resumable.

For each dimension d and step count M (grid of PLAN.md section 6):
- oracle ODE; oracle SDE for every ε in ORACLE_EPSILONS;
- for each trained model in MODELS: learnt ODE; learnt SDE with the MAIN_TREATMENT score for every ε in EPSILONS;
- for the CONTROL_SEED models: learnt SDE with each of CONTROL_TREATMENTS for ε in CONTROL_EPSILONS;
- for HYBRID_MODEL: oracle b + learnt (MAIN_TREATMENT) s, and learnt b + oracle s, for ε in HYBRID_EPSILONS.

A missing or failed trained model produces explicit ``MISSING_MODEL`` rows, never a silent gap.
Rows already present in the output file are skipped, so the sweep can be interrupted and resumed.

Example::

    uv run python -m scripts.gaussian_mixture.sde_vs_ode.sample_all --nsteps 100000 --dims 2,4
"""

import json
import time
import zlib
from pathlib import Path

import jax
import numpy as np
import typer

from scripts.gaussian_mixture.sde_vs_ode import configs
from scripts.gaussian_mixture.sde_vs_ode.fields import (
    Problem,
    compiled_sampler,
    learnt_fields,
    ode_transport,
    oracle_fields,
    restore_model,
    sde_transport,
)
from scripts.gaussian_mixture.sde_vs_ode.metrics import replicate_metrics
from scripts.gaussian_mixture.sde_vs_ode.train_all import find_model

app = typer.Typer(pretty_exceptions_show_locals=False)

ROW_KEY = ("d", "M", "source", "variant", "seed", "sampler", "eps", "treatment", "replicate")


def network_passes(sampler: str, source: str, d: int, M: int) -> int | None:
    """Network forward-pass equivalents per sample (None for closed-form oracle fields).

    ODE Euler step with exact divergence: 1 primal + d tangent passes (jax.linearize).
    SDE: b and η_z are evaluated M + 1 times each.
    """
    if source == "oracle":
        return None
    if sampler == "ode":
        return M * (1 + d)
    return 2 * (M + 1)


def jobs_for(d: int, M: int, problem: Problem, nsteps: int):
    """Yield (descriptor, transport factory or None). ``None`` means the model is missing."""
    b_or, s_or = oracle_fields(problem)
    base = {"d": d, "M": M, "variant": None, "seed": None, "treatment": None}
    yield {**base, "source": "oracle", "sampler": "ode", "eps": None}, lambda: ode_transport(b_or, M)
    for eps in configs.ORACLE_EPSILONS:
        yield {**base, "source": "oracle", "sampler": "sde", "eps": eps}, lambda e=eps: sde_transport(b_or, s_or, e, M)

    main = configs.MAIN_TREATMENT
    for variant, seed in configs.MODELS:
        desc = {**base, "variant": variant, "seed": seed}
        path = find_model(d, variant, seed, nsteps)
        model = restore_model(path, nsteps)[0] if path is not None else None
        arms = [({**desc, "source": "learnt", "sampler": "ode", "eps": None}, "ode", None)]
        arms += [
            ({**desc, "source": "learnt", "sampler": "sde", "eps": eps, "treatment": main}, "sde", main)
            for eps in configs.EPSILONS
        ]
        if seed == configs.CONTROL_SEED:
            arms += [
                ({**desc, "source": "learnt", "sampler": "sde", "eps": eps, "treatment": tr}, "sde", tr)
                for tr in configs.CONTROL_TREATMENTS
                for eps in configs.CONTROL_EPSILONS
            ]
        if (variant, seed) == configs.HYBRID_MODEL:
            arms += [
                ({**desc, "source": src, "sampler": "sde", "eps": eps, "treatment": tr}, src, tr)
                for eps in configs.HYBRID_EPSILONS
                for src, tr in [("hybrid_oracle_b", main), ("hybrid_oracle_s", None)]
            ]
        for descriptor, kind, treatment in arms:
            if model is None:
                yield descriptor, None
                continue
            if kind == "ode":
                make = lambda m=model: ode_transport(m.velocity_field, M)  # noqa: E731
            elif kind == "sde":
                make = lambda m=model, tr=treatment, e=descriptor["eps"]: sde_transport(  # noqa: E731
                    *learnt_fields(m, problem, tr), e, M
                )
            elif kind == "hybrid_oracle_b":
                make = lambda m=model, e=descriptor["eps"]: sde_transport(  # noqa: E731
                    b_or, learnt_fields(m, problem, main)[1], e, M
                )
            else:
                make = lambda m=model, e=descriptor["eps"]: sde_transport(m.velocity_field, s_or, e, M)  # noqa: E731
            yield descriptor, make


def key_of(row: dict) -> tuple:
    return tuple(row.get(k) for k in ROW_KEY)


@app.command()
def main(
    nsteps: int = typer.Option(configs.MAIN_NSTEPS, help="Training budget of the models to load."),
    dims: str = typer.Option(",".join(map(str, configs.DIMS))),
    steps: str = typer.Option(",".join(map(str, configs.STEPS))),
    replicates: int = typer.Option(configs.N_REPLICATES),
    n: int = typer.Option(configs.N_PER_REPLICATE, help="Samples per replicate."),
    out_dir: Path = typer.Option(configs.WORK_DIR / "samples"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in [int(v) for v in dims.split(",")]:
        out = out_dir / f"d{d}.jsonl"
        done = set()
        if out.exists():
            done = {key_of(json.loads(line)) for line in out.read_text().splitlines() if line.strip()}
        problem = Problem(d)
        for M in [int(v) for v in steps.split(",")]:
            for desc, make in jobs_for(d, M, problem, nsteps):
                todo = [r for r in range(replicates) if key_of({**desc, "replicate": r}) not in done]
                if not todo:
                    continue
                with open(out, "a") as f:
                    if make is None:
                        for r in todo:
                            f.write(json.dumps({**desc, "replicate": r, "status": "MISSING_MODEL"}) + "\n")
                        continue
                    t0 = time.perf_counter()
                    fn = compiled_sampler(make(), problem, n)
                    compile_s = time.perf_counter() - t0
                    job_seed = zlib.crc32(json.dumps(desc, sort_keys=True).encode())
                    for r in todo:
                        t0 = time.perf_counter()
                        y, log_w = jax.block_until_ready(fn(jax.random.fold_in(jax.random.key(job_seed), r)))
                        wall = time.perf_counter() - t0
                        row = {
                            **desc,
                            "replicate": r,
                            **replicate_metrics(np.asarray(y), np.asarray(log_w)),
                            "wall_s": wall,
                            "compile_s": compile_s,
                            "network_passes_per_sample": network_passes(desc["sampler"], desc["source"], d, M),
                            "nsteps_training": nsteps,
                        }
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                last = json.loads(out.read_text().splitlines()[-1])
                print(
                    f"d={d} M={M} {desc['source']:16s} {desc['sampler']} eps={desc['eps']} "
                    f"v={desc['variant']} s={desc['seed']} tr={desc['treatment']}: {last.get('status')} "
                    f"ESS={last.get('ess', float('nan')):.3g} logZ={last.get('log_z', float('nan')):+.3g}"
                )


if __name__ == "__main__":
    app()
