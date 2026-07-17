import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from particle_systems.selection import (
    check_patch_remainder_separation,
    soft_log_pi_ratio,
    torus_distances_from_point,
)
from superiorflows.train.callbacks import Callback


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

    n_valids = sels.n_patch + sels.n_context
    U_old = jax.vmap(masked_energy_fn)(active_old.positions, active_old.species, n_valids)
    U_new = jax.vmap(masked_energy_fn)(active_new.positions, active_new.species, n_valids)
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


class AcceptanceCallback(Callback):
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

        def _filter(arr):
            arr = np.asarray(arr)
            return arr[np.isfinite(arr)]

        ap_val = _filter(ap)
        pi_val = _filter(pi)
        q_val = _filter(q)
        du_val = _filter(du)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].hist(ap_val, bins=30, color="#2196F3", alpha=0.7)
        axes[0, 0].set_title("Acceptance Probability")

        axes[0, 1].hist(pi_val, bins=30, color="#4CAF50", alpha=0.7)
        axes[0, 1].set_title(r"$\log \pi$ ratio")

        axes[1, 0].hist(q_val, bins=30, color="#FF9800", alpha=0.7)
        axes[1, 0].set_title(r"$\log q$ ratio")

        axes[1, 1].hist(du_val, bins=30, color="#F44336", alpha=0.7)
        axes[1, 1].set_title(r"$\Delta U / T$")

        fig.suptitle(f"Acceptance Metrics — step {step}")
        fig.tight_layout()

        self.tb_writer.add_figure("acceptance/histograms", fig, step)
        plt.close(fig)
