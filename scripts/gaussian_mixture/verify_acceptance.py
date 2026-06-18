"""Verify Metropolis-Hastings acceptance rates for a trained flow model."""

import json
import time
from pathlib import Path

import distreqx.distributions as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import typer
from typing_extensions import Annotated

from scripts.gaussian_mixture.sample_gaussian_mixture import load_trained_flow
from scripts.gaussian_mixture.train_gaussian_mixture import build_louis_mixture
from superiorflows.partial import PartialFlowUpdater
from superiorflows.selection import uniform_index_selection

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    ckpt_path: Path = typer.Argument(
        ...,
        help="Path to checkpoint directory (containing config.json & checkpoint)",
        exists=True,
        dir_okay=True,
    ),
    num_states: int = typer.Option(1000, help="Number of initial states M to sample from target"),
    num_proposals: int = typer.Option(10, help="Number of proposals M' per state"),
    seed: int = typer.Option(0, help="Random seed for verification"),
    device: str = typer.Option(None, help="JAX device: cpu | gpu"),
    eval_dofs: Annotated[
        int | None,
        typer.Option(help="Patch size to use for evaluation (defaults to trained partial_dofs, or d for global)"),
    ] = None,
):
    """Verify Metropolis-Hastings acceptance rates on a trained flow model."""
    if device is not None:
        jax.config.update("jax_platform_name", device)

    print(f"\n{'=' * 60}")
    print("Metropolis-Hastings Acceptance Verification")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Num states M   : {num_states}")
    print(f"  Proposals M'   : {num_proposals}")
    print(f"  Seed           : {seed}")
    print(f"  JAX process    : {jax.process_index()}/{jax.process_count()}")
    print(f"  JAX devices    : {jax.devices()}")
    print(f"{'=' * 60}\n")

    # Load flow and config
    print("Loading trained flow and config...")
    flow, d, a, config = load_trained_flow(ckpt_path)
    bijector = flow.bijector

    # Re-build target distribution
    tgcfg = config["target"]
    sigma2_max = tgcfg.get("sigma2_max", 0.2)
    sigma2_min = tgcfg.get("sigma2_min", 0.01)
    weight = tgcfg.get("weight", 2.0 / 3.0)
    target_dist = build_louis_mixture(d, a, sigma2_max, sigma2_min, weight)

    # Read partial DOFs configuration
    partial_dofs = config["training"].get("partial_dofs")
    if eval_dofs is not None:
        n_eval = eval_dofs
        print(f"Evaluating model using patch size n_eval = {n_eval} (explicit override)")
    else:
        n_eval = partial_dofs if partial_dofs is not None else d
        print(f"Evaluating model using default patch size n_eval = {n_eval}")

    # Build base distribution of size n_eval and PartialFlowUpdater
    base_dist_partial = dsx.MultivariateNormalDiag(jnp.zeros(n_eval), jnp.ones(n_eval))
    updater = PartialFlowUpdater(bijector, base_dist_partial)

    # Selection protocol
    selection_protocol = uniform_index_selection(n_eval)

    # Generate initial states x from the target distribution
    key = jax.random.key(seed)
    key_states, key_proposals = jax.random.split(key)
    print(f"Generating {num_states} initial states from target mixture...")
    sample_keys = jax.random.split(key_states, num_states)
    states_x = jax.vmap(target_dist.sample)(sample_keys)

    print("Compiling JAX graph for MH acceptance computation...")
    t_comp_start = time.time()

    # Define computation for a single pair of (x, key_proposal)
    def compute_single_acceptance(x, key_prop):
        key_sel, key_upd, key_hutch, k_lp1, k_lp2 = jax.random.split(key_prop, 5)
        mask = selection_protocol(key_sel, x)

        # Propose update
        kwargs = {}
        if bijector.hutchinson_samples is not None:
            kwargs["key"] = key_hutch

        x_prime = updater.update(x, mask, key=key_upd, **kwargs)

        # Compute M-H ratio
        logp_x = target_dist.log_prob(x)
        logp_x_prime = target_dist.log_prob(x_prime)

        # Transition probabilities q(x | x_prime) and q(x_prime | x)
        kwargs1 = dict(kwargs, key=k_lp1) if "key" in kwargs else {}
        kwargs2 = dict(kwargs, key=k_lp2) if "key" in kwargs else {}

        logq_reverse = updater.log_prob(x, mask, **kwargs1)
        logq_forward = updater.log_prob(x_prime, mask, **kwargs2)

        log_h = logp_x_prime + logq_reverse - logp_x - logq_forward
        alpha = jnp.minimum(1.0, jnp.exp(log_h))
        return alpha

    # Vmap over M' proposals for a single state
    def compute_proposals_for_state(x, key_state):
        keys = jax.random.split(key_state, num_proposals)
        return jax.vmap(lambda k: compute_single_acceptance(x, k))(keys)

    # Vmap over M states
    @eqx.filter_jit
    def compute_all_acceptances(states, key_all):
        keys = jax.random.split(key_all, num_states)
        return jax.vmap(compute_proposals_for_state)(states, keys)

    # Trigger compilation
    _ = compute_all_acceptances(states_x, key_proposals)
    print(f"Compiled successfully in {time.time() - t_comp_start:.1f}s.")

    # Run evaluation
    print("Evaluating Metropolis-Hastings acceptance rates...")
    t_run_start = time.time()
    alphas = compute_all_acceptances(states_x, key_proposals)
    alphas.block_until_ready()
    t_elapsed = time.time() - t_run_start

    total_pairs = num_states * num_proposals
    time_per_proposal_ms = (t_elapsed * 1000.0) / total_pairs

    # Compute statistics
    alphas_flat = np.array(alphas).flatten()  # shape (M * M',)
    mean_acc = np.mean(alphas_flat)
    std_acc = np.std(alphas_flat)
    sem_acc = std_acc / np.sqrt(len(alphas_flat))

    print(f"\nVerification completed in {t_elapsed:.2f}s:")
    print(f"  Mean Acceptance Rate : {mean_acc:.4f}")
    print(f"  Standard Deviation   : {std_acc:.4f}")
    print(f"  SEM (Standard Error) : {sem_acc:.4f}")
    print(f"  Total Pairs Evaluated: {len(alphas_flat)}")
    print(f"  Time per Proposal    : {time_per_proposal_ms:.4f} ms")

    # Save to JSON
    results = {
        "mean_acceptance": float(mean_acc),
        "std_acceptance": float(std_acc),
        "sem_acceptance": float(sem_acc),
        "total_pairs": int(len(alphas_flat)),
        "num_states_M": num_states,
        "num_proposals_M_prime": num_proposals,
        "partial_dofs": partial_dofs,
        "eval_dofs": n_eval,
        "time_per_proposal_ms": float(time_per_proposal_ms),
        "total_time_s": float(t_elapsed),
        "d": d,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    }

    output_file = ckpt_path / "acceptance_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {output_file}\n")


if __name__ == "__main__":
    app()
