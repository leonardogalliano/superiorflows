"""Particle-system-specific training callbacks.

Contains :class:`BoltzmannCallback` — a rich evaluation callback that
samples from the flow during training and logs physical observables
(energies, g(r), particle snapshots) to TensorBoard.
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from superiorflows import Flow
from superiorflows.train import Callback

from particle_systems.particle_system import (
    ParticleSystem,
    batch_to_trajectory,
)

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
        n_samples: int = 20,
        n_target_samples: int = 256,
        eval_freq: int = 250,
        tb_writer=None,
        energy_filter_sigma: float = 10.0,
        species_radii: np.ndarray | None = None,
        n_show: int = 10,
    ):
        self.energy_fn = energy_fn
        self.base_distribution = base_distribution
        self.ref_species = ref_species
        self.target_source = target_source
        self.flow_kwargs = flow_kwargs or {}
        self.n_samples = n_samples
        self.n_target_samples = n_target_samples
        self.eval_freq = eval_freq
        self.tb_writer = tb_writer
        self.energy_filter_sigma = energy_filter_sigma
        self.species_radii = species_radii
        self.n_show = n_show

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
        indices = np.random.choice(len(self.target_source), size=n, replace=False)

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

        flow = Flow(
            velocity_field=trainer.model,
            base_distribution=self.base_distribution,
            **self.flow_kwargs,
        )

        try:
            key, subkey = jax.random.split(trainer.key)
            samples = flow.sample(seed=subkey, sample_shape=(self.n_samples,))

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
        bins = np.linspace(lo, hi, 30)
        ax.hist(
            self._target_energies,
            bins=bins,
            alpha=0.6,
            color="#FF9800",
            label="Target",
            density=True,
        )
        if len(filtered) > 0:
            ax.hist(
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
