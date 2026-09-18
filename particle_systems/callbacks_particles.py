"""Particle-system-specific training callbacks.

Contains :class:`BoltzmannCallback` — a rich evaluation callback that
samples from the flow during training and logs physical observables
(energies, g(r), particle snapshots) to TensorBoard.
"""

from typing import Any, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from particle_systems.particle_system import (
    ParticleSystem,
    batch_to_trajectory,
)
from particle_systems.selection import (
    check_patch_remainder_separation,
    soft_log_pi_ratio,
    torus_distances_from_point,
)
from superiorflows import Flow, ODEBijector
from superiorflows.train import Callback

# ── Helpers ───────────────────────────────────────────────────────────────────


def render_particle_grid(
    systems: ParticleSystem,
    n_show: int,
    species_radii: np.ndarray,
    title: str = "",
    cmap_name: str = "tab10",
):
    """Render a grid of particle snapshots as a matplotlib Figure.

    Args:
        systems: Batched ``ParticleSystem`` with shape ``(B, N, d)``.
        n_show: Number of snapshots to display (capped at batch size).
        species_radii: Per-species display radii, shape ``(n_species,)``.
        title: Super-title for the figure.
        cmap_name: Matplotlib colormap name for species colouring.

    Returns:
        A ``matplotlib.Figure`` ready for ``add_figure``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    positions = np.asarray(systems.positions)
    species = np.asarray(systems.species)
    boxes = np.asarray(systems.box)

    B = positions.shape[0]
    n_show = min(n_show, B)

    ncols = min(5, n_show)
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3 * ncols, 3 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx in range(n_show):
        ax = axes_flat[idx]
        box = boxes if boxes.ndim == 1 else boxes[idx]
        pos = positions[idx] % box  # fold into box
        spec = species[idx].astype(int)

        radii = species_radii[spec]
        patches = [Circle((x, y), r) for (x, y), r in zip(pos, radii)]

        collection = PatchCollection(patches, edgecolors="k", linewidths=0.3)
        collection.set_array(spec.astype(float))
        collection.set_cmap(cmap_name)
        n_species = len(species_radii)
        collection.set_clim(-0.5, n_species - 0.5)

        ax.add_collection(collection)
        ax.set_xlim(0, box[0])
        ax.set_ylim(0, box[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[n_show:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ── BoltzmannCallback ────────────────────────────────────────────────────────


class BoltzmannCallback(Callback):
    """Periodically samples from the flow and computes physical observables.

    Draws ``n_samples`` configurations via ``flow.sample`` (fast, no log-prob),
    then computes:

    - **Scalars** (injected into ``logs``): ``energy_median``, ``energy_mean``
    - **Energy figure** (TensorBoard): model vs target energy histogram,
      filtered to ±4σ of the target distribution
    - **Total g(r) figure** (TensorBoard): total pair correlation comparison
    - **Partial g(r) figure** (TensorBoard): species-resolved pair correlations
    - **Sample snapshots** (TensorBoard): particle configuration grids

    Rich data (matplotlib figures) are written directly to a shared TensorBoard
    writer so they appear under the same run entry as scalar training metrics.

    Requires ``atooms-pp`` for g(r) computation.
    """

    def __init__(
        self,
        energy_fn,
        base_distribution,
        ref_species,
        target_source,
        flow_kwargs: dict | None = None,
        make_bijector=None,
        n_samples: int = 20,
        n_target_samples: int = 256,
        eval_freq: int = 250,
        tb_writer=None,
        energy_filter_sigma: float = 10.0,
        species_radii: np.ndarray | None = None,
        n_show: int = 10,
        seed: int = 0,
    ):
        self.energy_fn = energy_fn
        self.base_distribution = base_distribution
        self.ref_species = ref_species
        self.target_source = target_source
        if make_bijector is not None:
            self.make_bijector = make_bijector
        elif flow_kwargs is not None:
            fk = dict(flow_kwargs)
            self.make_bijector = lambda model: ODEBijector(model, **fk)
        else:
            self.make_bijector = lambda model: ODEBijector(model)
        self.n_samples = n_samples
        self.n_target_samples = n_target_samples
        self.eval_freq = eval_freq
        self.tb_writer = tb_writer
        self.energy_filter_sigma = energy_filter_sigma
        self.species_radii = species_radii
        self.n_show = n_show
        self.seed = seed

        # Pre-computed target observables
        self._target_energies = None
        self._target_energy_mean = None
        self._target_energy_std = None
        self._target_gr_grid = None
        self._target_gr_value = None
        self._target_partial_gr = {}
        self._target_samples_fig = None  # fixed figure, logged once
        self._species_list = sorted(int(s) for s in np.unique(np.asarray(ref_species)))

    def on_train_start(self, trainer, **kwargs):
        self.total_steps = kwargs.get("total_steps", -1)
        self._precompute_target_observables()

    def _precompute_target_observables(self):
        """Compute target energy distribution, g(r), and sample snapshot once."""
        import atooms.postprocessing as pp
        from atooms.trajectory.decorators import fold

        n = min(self.n_target_samples, len(self.target_source))
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(self.target_source), size=n, replace=False)

        # Target energies (batched for speed)
        target_samples = [self.target_source[int(i)] for i in indices]
        positions = np.stack([s.positions for s in target_samples])
        species = np.stack([s.species for s in target_samples])
        box = np.stack([s.box for s in target_samples])

        target_positions = jnp.asarray(positions)
        target_species = jnp.asarray(species)

        def energy_single(pos, spec):
            return self.energy_fn(pos, spec)

        self._target_energies = np.asarray(jax.vmap(energy_single)(target_positions, target_species))
        self._target_energy_mean = float(np.mean(self._target_energies))
        self._target_energy_std = float(np.std(self._target_energies))

        # Target g(r) via atooms postprocessing
        target_batch = ParticleSystem(positions=positions, species=species, box=box)
        trj = batch_to_trajectory(target_batch)
        trj.add_callback(fold)

        # Total g(r)
        gr = pp.RadialDistributionFunction(trj)
        gr.compute()
        self._target_gr_grid = np.array(gr.grid)
        self._target_gr_value = np.array(gr.value)

        # Partial g(r) for each species pair
        if len(self._species_list) > 1:
            gr_partial = pp.Partial(pp.RadialDistributionFunction, species=self._species_list, trajectory=trj)
            gr_partial.compute()
            for key in gr_partial.partial:
                if key[0] <= key[1]:
                    g = gr_partial.partial[key]
                    self._target_partial_gr[key] = (np.array(g.grid), np.array(g.value))

        # Target sample snapshot (fixed, logged once) - Only for 2D
        if self.tb_writer is not None and self.species_radii is not None and self.target_source.d == 2:
            self._target_samples_fig = render_particle_grid(
                target_batch,
                n_show=self.n_show,
                species_radii=self.species_radii,
                title="Target samples",
            )
            self.tb_writer.add_figure("boltzmann/samples_target", self._target_samples_fig, 0)
            self.tb_writer.flush()

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        is_last = hasattr(self, "total_steps") and step == self.total_steps
        if step % self.eval_freq != 0 and step != 1 and not is_last:
            return

        bijector = self.make_bijector(trainer.model)
        flow = Flow(bijector, self.base_distribution)

        try:
            key, subkey = jax.random.split(trainer.key)
            keys = jax.random.split(subkey, self.n_samples)
            samples = jax.vmap(flow.sample)(keys)

            def single_energy(sample: ParticleSystem):
                return self.energy_fn(sample.positions, sample.species)

            model_energies = np.asarray(jax.vmap(single_energy)(samples))

            # Scalar metrics → logs dict (picked up by LoggerCallback + TensorBoardLogger)
            logs["energy_median"] = float(np.median(model_energies))
            logs["energy_mean"] = float(np.mean(model_energies))

            # TensorBoard: figures
            if self.tb_writer is not None:
                self._write_tb(step, samples, model_energies)

        except Exception as e:
            logs["energy_median"] = float("nan")
            logs["energy_mean"] = float("nan")
            print(f"[BoltzmannCallback] Error at step {step}: {e}")

    # ── TensorBoard writers ───────────────────────────────────────────────

    def _write_tb(self, step, samples, model_energies):
        """Write energy figure, g(r) figures, and sample snapshots to TensorBoard."""
        import atooms.postprocessing as pp
        import matplotlib
        from atooms.trajectory.decorators import fold

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ── Energy comparison figure ──────────────────────────────────────
        self._write_energy_figure(step, model_energies, plt)

        # ── g(r) figures ──────────────────────────────────────────────────
        try:
            trj = batch_to_trajectory(samples)
            trj.add_callback(fold)

            # Total g(r)
            gr = pp.RadialDistributionFunction(trj)
            gr.compute()

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(gr.grid, gr.value, label="Model", color="#2196F3")
            ax.plot(
                self._target_gr_grid,
                self._target_gr_value,
                label="Target",
                color="#FF9800",
            )
            ax.set_xlabel("$r$")
            ax.set_ylabel("$g(r)$")
            ax.legend()
            ax.set_title(f"step {step}")
            fig.tight_layout()
            self.tb_writer.add_figure("boltzmann/gr", fig, step)
            plt.close(fig)

            # Partial g(r) — only for multi-species systems
            if len(self._species_list) > 1:
                self._write_partial_gr(step, trj, pp, plt)

        except Exception as e:
            print(f"[BoltzmannCallback] g(r) computation failed at step {step}: {e}")

        # ── Sample snapshots ──────────────────────────────────────────────
        if self.species_radii is not None and samples.d == 2:
            try:
                fig = render_particle_grid(
                    samples,
                    n_show=self.n_show,
                    species_radii=self.species_radii,
                    title=f"Model samples — step {step}",
                )
                self.tb_writer.add_figure("boltzmann/samples_model", fig, step)
                plt.close(fig)
            except Exception as e:
                print(f"[BoltzmannCallback] sample rendering failed at step {step}: {e}")

        self.tb_writer.flush()

    def _write_energy_figure(self, step, model_energies, plt):
        """Energy histogram: model vs target, filtered to ±Nσ of target."""
        sigma = self.energy_filter_sigma
        lo = self._target_energy_mean - sigma * self._target_energy_std
        hi = self._target_energy_mean + sigma * self._target_energy_std

        mask = (model_energies >= lo) & (model_energies <= hi)
        filtered = model_energies[mask]
        pct = 100.0 * len(filtered) / max(len(model_energies), 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            bins = np.linspace(lo, hi, 30)
        else:
            bins = 30

        _safe_hist(
            ax,
            self._target_energies,
            bins=bins,
            alpha=0.6,
            color="#FF9800",
            label="Target",
            density=True,
        )
        if len(filtered) > 0:
            _safe_hist(
                ax,
                filtered,
                bins=bins,
                alpha=0.6,
                color="#2196F3",
                label=f"Model ({pct:.0f}% samples)",
                density=True,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No model samples\nin target range",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#2196F3",
            )
        ax.set_xlabel("$U$")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"step {step}")
        fig.tight_layout()
        self.tb_writer.add_figure("boltzmann/energy", fig, step)
        plt.close(fig)

    def _write_partial_gr(self, step, trj, pp, plt):
        """Partial g(r) for each species pair, model vs target."""
        gr_partial = pp.Partial(pp.RadialDistributionFunction, species=self._species_list, trajectory=trj)
        gr_partial.compute()

        keys = sorted(k for k in gr_partial.partial if k[0] <= k[1])
        n = len(keys)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, key in zip(axes_flat, keys):
            g = gr_partial.partial[key]
            i, j = key
            ax.plot(g.grid, g.value, label="Model", color="#2196F3")
            if key in self._target_partial_gr:
                tg, tv = self._target_partial_gr[key]
                ax.plot(tg, tv, label="Target", color="#FF9800")
            ax.set_xlabel("$r$")
            ax.set_ylabel(f"$g_{{{i}{j}}}(r)$")
            ax.legend(fontsize=8)

        for ax in axes_flat[len(keys) :]:
            ax.axis("off")

        fig.suptitle(f"Partial $g(r)$ — step {step}")
        fig.tight_layout()
        self.tb_writer.add_figure("boltzmann/gr_partial", fig, step)
        plt.close(fig)


@eqx.filter_jit
def compute_acceptance_metrics(
    patch_updater,
    batch,
    subkeys,
    masked_energy_fn,
    temperature: float,
    r_mean: float,
    sigma_r: float,
    r_cut: float,
):
    # Propose new systems and get forward log q
    systems_new, log_qs, sels = jax.vmap(patch_updater.update_and_log_prob)(batch, subkeys)

    # Evaluate backward flow log q
    log_q_bwds = jax.vmap(patch_updater.log_prob)(batch, sels)

    # Selection probability ratio
    dists_before = jax.vmap(torus_distances_from_point)(sels.anchor, batch.positions, batch.box)
    dists_after = jax.vmap(torus_distances_from_point)(sels.anchor, systems_new.positions, systems_new.box)

    log_pi_ratios = jax.vmap(soft_log_pi_ratio, in_axes=(0, 0, 0, 0, 0, 0, None, None))(
        dists_before,
        dists_after,
        sels.patch_idx,
        sels.context_idx,
        sels.n_patch,
        sels.n_context,
        r_mean,
        sigma_r,
    )

    # Subsystem energy difference (only active particles)
    active_old = jax.vmap(patch_updater._build_subsystem)(batch, sels)
    active_new = jax.vmap(patch_updater._build_subsystem)(systems_new, sels)

    U_old = jax.vmap(masked_energy_fn)(active_old.positions, active_old.species, sels.n_patch, sels.n_context)
    U_new = jax.vmap(masked_energy_fn)(active_new.positions, active_new.species, sels.n_patch, sels.n_context)
    delta_U = U_new - U_old
    delta_U_T = delta_U / temperature

    # Safety check: patch must be separated from remainder
    pos_patch_new = jax.vmap(lambda pos, idx: pos[idx])(systems_new.positions, sels.patch_idx)
    is_safe = jax.vmap(check_patch_remainder_separation, in_axes=(0, 0, 0, 0, 0, 0, 0, None))(
        batch.positions,
        pos_patch_new,
        sels.patch_idx,
        sels.context_idx,
        sels.n_patch,
        sels.n_context,
        batch.box,
        r_cut,
    )

    log_q_ratios = log_q_bwds - log_qs

    # Acceptance ratio
    log_alpha = -delta_U_T + log_pi_ratios + log_q_ratios
    log_alpha = jnp.where(is_safe, log_alpha, -jnp.inf)
    accept_probs = jnp.exp(jnp.minimum(log_alpha, 0.0))

    return accept_probs, is_safe, log_pi_ratios, log_q_ratios, delta_U_T, systems_new, sels, log_qs


def _safe_mean(arr):
    # Filter out infinite and NaN values for statistics
    arr = np.asarray(arr)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


def _safe_median(arr):
    # Filter out infinite and NaN values for statistics
    arr = np.asarray(arr)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return float("nan")
    return float(np.median(valid))


def _safe_hist(ax, arr, bins=30, **kwargs):
    """Plot a histogram safely, handling empty, non-finite, and zero-variance data."""
    if arr is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="gray")
        return
    arr_np = np.asarray(arr)
    finite_data = arr_np[np.isfinite(arr_np)]
    if len(finite_data) == 0:
        ax.text(
            0.5,
            0.5,
            "No finite data\n(NaNs / Infs)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="red",
        )
    elif np.ptp(finite_data) == 0:
        ax.hist(finite_data, bins=1, **kwargs)
    else:
        if isinstance(bins, np.ndarray):
            if np.all(np.isfinite(bins)) and np.all(np.diff(bins) > 0):
                ax.hist(finite_data, bins=bins, **kwargs)
            else:
                ax.hist(finite_data, bins=30, **kwargs)
        else:
            ax.hist(finite_data, bins=bins, **kwargs)


class PatchAcceptanceCallback(Callback):
    def __init__(
        self,
        val_data,
        make_patch_updater,
        masked_energy_fn,
        temperature: float,
        r_mean: float,
        sigma_r: float,
        r_cut: float,
        r_context: float,
        eval_freq: int = 250,
        tb_writer=None,
        species_radii=None,
    ):
        self.val_data = val_data
        self.make_patch_updater = make_patch_updater
        self.masked_energy_fn = masked_energy_fn
        self.temperature = temperature
        self.r_mean = r_mean
        self.sigma_r = sigma_r
        self.r_cut = r_cut
        self.r_context = r_context
        self.eval_freq = eval_freq
        self.tb_writer = tb_writer
        self.species_radii = species_radii

    def on_train_start(self, trainer, **kwargs):
        self.total_steps = kwargs.get("total_steps", -1)

    def on_step_end(self, trainer, step: int, logs: dict, **kwargs):
        is_last = hasattr(self, "total_steps") and step == self.total_steps
        if step % self.eval_freq != 0 and step != 1 and not is_last:
            return

        try:
            patch_updater = self.make_patch_updater(trainer.model)

            all_accept_probs = []
            all_is_safe = []
            all_log_pi_ratios = []
            all_log_q_ratios = []
            all_delta_U_T = []

            for i, batch in enumerate(self.val_data):
                key, subkey = jax.random.split(trainer.key)
                trainer.key = key

                # Put the batch on device manually just in case
                batch = jax.device_put(batch)
                batch_size = batch.positions.shape[0]
                subkeys = jax.random.split(subkey, batch_size)

                ap, safe, pi, q, du, systems_new, sels, log_qs = compute_acceptance_metrics(
                    patch_updater,
                    batch,
                    subkeys,
                    self.masked_energy_fn,
                    self.temperature,
                    self.r_mean,
                    self.sigma_r,
                    self.r_cut,
                )

                all_accept_probs.append(ap)
                all_is_safe.append(safe)
                all_log_pi_ratios.append(pi)
                all_log_q_ratios.append(q)
                all_delta_U_T.append(du)

                if i == 0:
                    first_batch = batch
                    first_systems_new = systems_new
                    first_sels = sels
                    first_log_q_ratios = q
                    first_delta_U_T = du

            if not all_accept_probs:
                return

            # Combine across batches
            ap = jnp.concatenate(all_accept_probs)
            safe = jnp.concatenate(all_is_safe)
            pi = jnp.concatenate(all_log_pi_ratios)
            q = jnp.concatenate(all_log_q_ratios)
            du = jnp.concatenate(all_delta_U_T)

            logs["acc_mean"] = _safe_mean(ap)
            logs["log_pi_ratio_mean"] = _safe_mean(pi)
            logs["log_q_ratio_mean"] = _safe_mean(q)
            logs["dU_T_mean"] = _safe_mean(du)
            logs["safe_rate"] = float(jnp.mean(safe))

            if self.tb_writer is not None:
                self._write_tb(step, ap, pi, q, du)
                self._write_patch_grids(
                    step, first_batch, first_systems_new, first_sels, first_log_q_ratios, first_delta_U_T
                )
        except Exception as e:
            logs["acc_mean"] = float("nan")
            logs["safe_rate"] = float("nan")
            print(f"[PatchAcceptanceCallback] Warning at step {step}: {e}")

    def _write_patch_grids(self, step, batch, systems_new, sels, log_q_ratios, delta_U_T):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        batch_size = min(6, batch.positions.shape[0])
        box_size = np.array(batch.box[0] if batch.box.ndim > 0 else batch.box)
        L = box_size[0] if box_size.ndim > 0 else box_size
        r_context_val = self.r_context

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        sigmas = self.species_radii if self.species_radii is not None else np.array([1.0])
        base_marker_size = 200

        # Cast to numpy
        b_pos = np.asarray(batch.positions)
        b_spec = np.asarray(batch.species)
        n_pos = np.asarray(systems_new.positions)
        l_qs = np.asarray(log_q_ratios)
        d_u_t = np.asarray(delta_U_T)

        for idx in range(batch_size):
            ax = axes[idx]

            pos_orig = b_pos[idx]
            pos_new = n_pos[idx]
            species = b_spec[idx].astype(int)
            sel = jax.tree.map(lambda x: np.asarray(x[idx]), sels)

            p_idx = sel.patch_idx[: sel.n_patch]
            c_idx = sel.context_idx[: sel.n_context]
            active_idx = np.concatenate([p_idx, c_idx])

            sub_orig = pos_orig[active_idx]
            sub_new = pos_new[active_idx]
            sub_spec = species[active_idx]

            anchor = sel.anchor
            center = L / 2.0
            shift = center - anchor

            shifted_orig = (sub_orig + shift) % L
            shifted_new = (sub_new + shift) % L

            colors = np.full(len(active_idx), "blue", dtype=object)
            colors[: sel.n_patch] = "red"

            spec_bound = np.clip(sub_spec, 0, len(sigmas) - 1)
            sizes = base_marker_size * (sigmas[spec_bound] ** 2)

            ax.scatter(shifted_orig[:, 0], shifted_orig[:, 1], s=sizes, c=colors, edgecolors="k")

            ax.scatter(
                shifted_new[: sel.n_patch, 0],
                shifted_new[: sel.n_patch, 1],
                s=sizes[: sel.n_patch],
                c="lightgreen",
                edgecolors="green",
                label="Updated Patch" if idx == 0 else "",
            )

            for j in range(sel.n_patch):
                d_disp = shifted_new[j] - shifted_orig[j]
                ax.arrow(
                    shifted_orig[j, 0],
                    shifted_orig[j, 1],
                    d_disp[0],
                    d_disp[1],
                    color="black",
                    linestyle=":",
                    head_width=0.15,
                    head_length=0.15,
                    length_includes_head=True,
                )

            center_pt = np.array([center, center])
            ax.add_patch(
                plt.Circle(center_pt, float(sel.radius), facecolor="red", edgecolor="darkred", alpha=0.08, zorder=1)
            )
            ax.add_patch(
                plt.Circle(
                    center_pt, r_context_val, fill=False, edgecolor="blue", linestyle="--", linewidth=1.5, zorder=1
                )
            )

            ax.set_xlim(L / 4, 3 * L / 4)
            ax.set_ylim(L / 4, 3 * L / 4)
            ax.set_aspect("equal")
            ax.set_title(f"$\\log q$ ratio: {l_qs[idx]:.2f} | $\\Delta U/T$: {d_u_t[idx]:.2f}", fontsize=12)

        for idx in range(batch_size, 6):
            axes[idx].axis("off")

        # fig.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=12)
        fig.suptitle(f"Patch updates — step {step}")
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        self.tb_writer.add_figure("acceptance/patches", fig, step)
        plt.close(fig)

    def _write_tb(self, step, ap, pi, q, du):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        _safe_hist(axes[0, 0], ap, bins=30, color="#2196F3", alpha=0.7)
        axes[0, 0].set_title("Acceptance Probability")

        _safe_hist(axes[0, 1], pi, bins=30, color="#4CAF50", alpha=0.7)
        axes[0, 1].set_title(r"$\log \pi$ ratio")

        _safe_hist(axes[1, 0], q, bins=30, color="#FF9800", alpha=0.7)
        axes[1, 0].set_title(r"$\log q$ ratio")

        _safe_hist(axes[1, 1], du, bins=30, color="#F44336", alpha=0.7)
        axes[1, 1].set_title(r"$\Delta U / T$")

        fig.suptitle(f"Acceptance Metrics — step {step}")
        fig.tight_layout()

        self.tb_writer.add_figure("acceptance/histograms", fig, step)
        plt.close(fig)


class PatchLocalEnergyCallback(Callback):
    """Evaluates patch local interaction energy U_loc(xp, xc) = U_p(xp) + U_pc(xp, xc).

    Precomputes target local energy distribution from validation target configurations,
    and periodically samples updated patches using fast forward ODE flow (patch_updater.update,
    no log-prob integration) to track u_loc_mean, u_loc_median, and output a TensorBoard
    histogram figure comparing Target vs Model local patch energies.
    """

    def __init__(
        self,
        val_data,
        make_patch_updater,
        masked_local_energy_fn,
        eval_freq: int = 250,
        tb_writer=None,
        energy_filter_sigma: float = 10.0,
        seed: int = 0,
    ):
        self.val_data = val_data
        self.make_patch_updater = make_patch_updater
        self.masked_local_energy_fn = masked_local_energy_fn
        self.eval_freq = eval_freq
        self.tb_writer = tb_writer
        self.energy_filter_sigma = energy_filter_sigma
        self.seed = seed

        self._target_u_loc = None
        self._target_u_loc_mean = None
        self._target_u_loc_std = None

    def on_train_start(self, trainer, **kwargs):
        self.total_steps = kwargs.get("total_steps", -1)
        self._precompute_target_observables(trainer)

    def _precompute_target_observables(self, trainer):
        """Compute target patch local energies U_loc from validation data."""
        patch_updater = self.make_patch_updater(trainer.model)
        target_u_locs = []

        key = jax.random.key(self.seed)
        for batch in self.val_data:
            key, subkey = jax.random.split(key)
            batch = jax.device_put(batch)
            batch_size = batch.positions.shape[0]
            subkeys = jax.random.split(subkey, batch_size)

            sels = jax.vmap(patch_updater.selection_fn)(subkeys, batch)
            active = jax.vmap(patch_updater._build_subsystem)(batch, sels)

            u_loc = jax.vmap(self.masked_local_energy_fn)(
                active.positions, active.species, sels.n_patch, sels.n_context
            )
            target_u_locs.append(np.asarray(u_loc))

        if target_u_locs:
            self._target_u_loc = np.concatenate(target_u_locs)
            self._target_u_loc_mean = float(np.mean(self._target_u_loc))
            self._target_u_loc_std = float(np.std(self._target_u_loc))

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        is_last = hasattr(self, "total_steps") and step == self.total_steps
        if step % self.eval_freq != 0 and step != 1 and not is_last:
            return

        patch_updater = self.make_patch_updater(trainer.model)
        all_model_u_locs = []

        try:
            for batch in self.val_data:
                key, subkey = jax.random.split(trainer.key)
                trainer.key = key

                batch = jax.device_put(batch)
                batch_size = batch.positions.shape[0]
                subkeys = jax.random.split(subkey, batch_size)

                systems_new, sels = jax.vmap(patch_updater.update)(batch, subkeys)

                active_new = jax.vmap(patch_updater._build_subsystem)(systems_new, sels)

                u_loc_new = jax.vmap(self.masked_local_energy_fn)(
                    active_new.positions, active_new.species, sels.n_patch, sels.n_context
                )
                all_model_u_locs.append(np.asarray(u_loc_new))

            if not all_model_u_locs:
                return

            model_u_loc = np.concatenate(all_model_u_locs)
            logs["u_loc_median"] = _safe_median(model_u_loc)
            logs["u_loc_mean"] = _safe_mean(model_u_loc)

            if self.tb_writer is not None and self._target_u_loc is not None:
                self._write_tb(step, model_u_loc)

        except Exception as e:
            logs["u_loc_median"] = float("nan")
            logs["u_loc_mean"] = float("nan")
            print(f"[PatchLocalEnergyCallback] Warning at step {step}: {e}")

    def _write_tb(self, step, model_u_loc):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sigma = self.energy_filter_sigma
        lo = self._target_u_loc_mean - sigma * self._target_u_loc_std
        hi = self._target_u_loc_mean + sigma * self._target_u_loc_std

        mask = (model_u_loc >= lo) & (model_u_loc <= hi)
        filtered = model_u_loc[mask]
        pct = 100.0 * len(filtered) / max(len(model_u_loc), 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            bins = np.linspace(lo, hi, 30)
        else:
            bins = 30

        _safe_hist(
            ax,
            self._target_u_loc,
            bins=bins,
            alpha=0.6,
            color="#FF9800",
            label="Target",
            density=True,
        )
        if len(filtered) > 0:
            _safe_hist(
                ax,
                filtered,
                bins=bins,
                alpha=0.6,
                color="#2196F3",
                label=f"Model ({pct:.0f}% samples)",
                density=True,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No model samples\nin target range",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#2196F3",
            )
        ax.set_xlabel(r"$U_{\text{loc}}$")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Patch local energy — step {step}")
        fig.tight_layout()
        self.tb_writer.add_figure("patch/local_energy", fig, step)
        plt.close(fig)
