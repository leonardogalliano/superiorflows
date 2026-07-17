from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from superiorflows.flow import Flow
from superiorflows.partial import PartialFlowUpdater, PartitionSpec, state_context_partition


def _pack_args(ctx, index_meta, user_args=None):
    """Combine partition context, index mapping metadata, and user-supplied args."""
    if index_meta is None:
        return (ctx, user_args) if user_args is not None else ctx
    return (ctx, (index_meta, user_args)) if user_args is not None else (ctx, index_meta)


def _split_batch_metadata(batch):
    """Separate a single-element batch from trailing per-sample metadata.

    Supports two conventions:

    * Standard: ``batch = pytree``  →  ``(pytree, None)``
    * Extended: ``batch = (pytree, meta_0, meta_1, ...)``  →  ``(pytree, (meta_0, meta_1, ...))``

    The extended format is detected when ``batch`` is a tuple of length >= 3
    and the second element has a ``.shape`` attribute (i.e. it is an array,
    not a pytree).  This heuristic avoids false positives for normal pytree
    tuples.
    """
    if isinstance(batch, tuple) and len(batch) >= 3:
        candidate = batch[1]
        if hasattr(candidate, "shape") and hasattr(batch[2], "shape"):
            return batch[0], tuple(batch[1:])
    return batch, None


def _split_paired_batch_metadata(batch):
    """Separate a paired-element batch from trailing per-sample metadata.

    Supports two conventions:

    * Standard: ``batch = (x0, x1)``  →  ``(x0, x1, None)``
    * Extended: ``batch = (x0, x1, meta_0, ...)``  →  ``(x0, x1, (meta_0, ...))``
    """
    if len(batch) > 2:
        return batch[0], batch[1], tuple(batch[2:])
    return batch[0], batch[1], None


class MaximumLikelihoodLoss(eqx.Module):
    """Negative log-likelihood loss for any normalising flow."""

    base_distribution: eqx.Module
    make_bijector: Callable = eqx.field(static=True)
    selection_protocol: Optional[Callable] = eqx.field(default=None, static=True)
    flow_kwargs: dict = eqx.field(default_factory=dict, static=True)

    @eqx.filter_jit
    def __call__(self, model, batch, key=None):
        bijector = self.make_bijector(model)
        samples, per_sample_args = _split_batch_metadata(batch)
        batch_size = jax.tree.leaves(samples)[0].shape[0]

        if self.selection_protocol is not None and key is None:
            raise ValueError("key is required when selection_protocol is set.")

        def single_log_prob(xi, key_i, *ua_i):
            fk = {k: v for k, v in self.flow_kwargs.items() if k != "args"}
            args_val = ua_i if ua_i else self.flow_kwargs.get("args")
            args_kw = {"args": args_val} if args_val is not None else {}
            if self.selection_protocol is None:
                flow = Flow(bijector, self.base_distribution)
                if key_i is not None:
                    return flow.log_prob(xi, key=key_i, **args_kw, **fk)
                return flow.log_prob(xi, **args_kw, **fk)
            else:
                k_sel, k_lp = jax.random.split(key_i)
                mask = self.selection_protocol(k_sel, xi)
                updater = PartialFlowUpdater(bijector, self.base_distribution)
                return updater.log_prob(xi, mask, key=k_lp, **args_kw, **fk)

        if key is not None:
            keys = jax.random.split(key, batch_size)
            if per_sample_args is not None:
                log_probs = jax.vmap(single_log_prob)(samples, keys, *per_sample_args)
            else:
                log_probs = jax.vmap(single_log_prob)(samples, keys)
        else:
            if per_sample_args is not None:
                log_probs = jax.vmap(lambda xi, *a: single_log_prob(xi, None, *a))(samples, *per_sample_args)
            else:
                log_probs = jax.vmap(lambda xi: single_log_prob(xi, None))(samples)

        return -jnp.mean(log_probs), {}


class EnergyBasedLoss(eqx.Module):
    """Energy-based (reverse KL) loss for any normalising flow."""

    base_distribution: eqx.Module
    target_distribution: eqx.Module
    make_bijector: Callable = eqx.field(static=True)
    selection_protocol: Optional[Callable] = eqx.field(default=None, static=True)
    flow_kwargs: dict = eqx.field(default_factory=dict, static=True)

    @eqx.filter_jit
    def __call__(self, model, batch, key=None):
        bijector = self.make_bijector(model)
        samples, per_sample_args = _split_batch_metadata(batch)
        batch_size = jax.tree.leaves(samples)[0].shape[0]

        if self.selection_protocol is not None and key is None:
            raise ValueError("key is required when selection_protocol is set.")

        def single_energy(xi, key_i, *ua_i):
            fk = {k: v for k, v in self.flow_kwargs.items() if k != "args"}
            if self.selection_protocol is None:
                flow = Flow(bijector, self.base_distribution)
                if ua_i:
                    n_patch, n_context = ua_i
                    args_kw = {"args": (n_patch, n_context)}
                    if key_i is not None:
                        x1_i, logq_i = flow.push_forward_and_log_prob(xi, key=key_i, **args_kw, **fk)
                    else:
                        x1_i, logq_i = flow.push_forward_and_log_prob(xi, **args_kw, **fk)
                    logp_i = self.target_distribution.log_prob(x1_i, n_valid=n_patch + n_context)
                else:
                    args_val = self.flow_kwargs.get("args")
                    args_kw = {"args": args_val} if args_val is not None else {}
                    if key_i is not None:
                        x1_i, logq_i = flow.push_forward_and_log_prob(xi, key=key_i, **args_kw, **fk)
                    else:
                        x1_i, logq_i = flow.push_forward_and_log_prob(xi, **args_kw, **fk)
                    logp_i = self.target_distribution.log_prob(x1_i)
                return logq_i - logp_i
            else:
                k_sel, k_lp = jax.random.split(key_i)
                mask = self.selection_protocol(k_sel, xi)
                updater = PartialFlowUpdater(bijector, self.base_distribution)
                args_val = ua_i if ua_i else self.flow_kwargs.get("args")
                args_kw = {"args": args_val} if args_val is not None else {}
                x1_i, logq_i = updater.update_and_log_prob(xi, mask, key=k_lp, **args_kw, **fk)
                logp_i = self.target_distribution.log_prob(x1_i)
                return logq_i - logp_i

        if key is not None:
            keys = jax.random.split(key, batch_size)
            if per_sample_args is not None:
                per_sample = jax.vmap(single_energy)(samples, keys, *per_sample_args)
            else:
                per_sample = jax.vmap(single_energy)(samples, keys)
        else:
            if per_sample_args is not None:
                per_sample = jax.vmap(lambda xi, *a: single_energy(xi, None, *a))(samples, *per_sample_args)
            else:
                per_sample = jax.vmap(lambda xi: single_energy(xi, None))(samples)

        return jnp.mean(per_sample), {}


class KullbackLeiblerLoss(eqx.Module):
    """Hybrid forward/reverse KL loss for any normalising flow."""

    mle_loss: MaximumLikelihoodLoss
    energy_loss: EnergyBasedLoss
    base_distribution: eqx.Module
    alpha: float

    def __init__(
        self,
        base_distribution,
        target_distribution,
        make_bijector,
        alpha=0.5,
        selection_protocol=None,
        flow_kwargs=None,
    ):
        fk = flow_kwargs or {}
        self.mle_loss = MaximumLikelihoodLoss(
            base_distribution, make_bijector, selection_protocol=selection_protocol, flow_kwargs=fk
        )
        self.energy_loss = EnergyBasedLoss(
            base_distribution, target_distribution, make_bijector, selection_protocol=selection_protocol, flow_kwargs=fk
        )
        self.base_distribution = base_distribution
        self.alpha = alpha

    @eqx.filter_jit
    def __call__(self, model, batch, key):
        samples, per_sample_args = _split_batch_metadata(batch)
        batch_size = jax.tree.leaves(samples)[0].shape[0]
        key1, key2, key3 = jax.random.split(key, 3)
        mle_term, _ = self.mle_loss(model, batch, key=key2)

        if self.mle_loss.selection_protocol is not None:
            energy_term, _ = self.energy_loss(model, batch, key=key3)
        else:
            keys = jax.random.split(key1, batch_size)
            x0 = jax.vmap(self.base_distribution.sample)(keys)
            if per_sample_args is not None:
                energy_batch = (x0, *per_sample_args)
            else:
                energy_batch = x0
            energy_term, _ = self.energy_loss(model, energy_batch, key=key3)

        return self.alpha * mle_term + (1 - self.alpha) * energy_term, {}


class StochasticInterpolantLoss(eqx.Module):
    """Stochastic Interpolant loss for flow matching training."""

    interpolant: Callable = eqx.field(static=True)
    gamma: Optional[Callable] = eqx.field(static=True)
    velocity_kwargs: dict = eqx.field(static=True)
    dt_interpolant: Callable = eqx.field(static=True)
    dt_gamma: Optional[Callable] = eqx.field(static=True)
    denoiser_weight: float = eqx.field(static=True)
    _get_velocity: Callable = eqx.field(static=True)
    _get_denoiser: Optional[Callable] = eqx.field(static=True)
    dynamic_mask: object = eqx.field(default=None)
    selection_protocol: Optional[Callable] = eqx.field(default=None, static=True)

    def __init__(
        self,
        interpolant: Callable,
        gamma: Optional[Callable] = None,
        dynamic_mask: Optional[object] = None,
        get_velocity: Optional[Callable] = None,
        get_denoiser: Optional[Callable] = None,
        denoiser_weight: float = 1.0,
        selection_protocol: Optional[Callable] = None,
        **velocity_kwargs,
    ):
        self.interpolant = interpolant
        self.gamma = gamma
        if dynamic_mask is not None:
            if callable(dynamic_mask) and not hasattr(dynamic_mask, "__jax_tree_flatten__"):
                self.dynamic_mask = jax.tree_util.Partial(dynamic_mask)
            else:
                self.dynamic_mask = dynamic_mask
        else:
            self.dynamic_mask = jax.tree_util.Partial(lambda x: jax.tree.map(eqx.is_inexact_array, x))
        self.selection_protocol = selection_protocol
        self.velocity_kwargs = velocity_kwargs
        self.denoiser_weight = denoiser_weight
        self._get_velocity = get_velocity if get_velocity is not None else lambda m: m
        self._get_denoiser = get_denoiser

        # Precompute time derivatives via JVP
        def _dt_interpolant(t, x0, x1):
            _, tangent = jax.jvp(
                lambda s: interpolant(s, x0, x1),
                (t,),
                (jnp.ones_like(t),),
            )
            return tangent

        self.dt_interpolant = _dt_interpolant

        if gamma is not None:

            def _dt_gamma(t):
                _, tangent = jax.jvp(gamma, (t,), (jnp.ones_like(t),))
                return tangent

            self.dt_gamma = _dt_gamma
        else:
            self.dt_gamma = None

    def __check_init__(self):
        if self._get_denoiser is not None and self.gamma is None:
            raise ValueError(
                "Denoiser learning requires a noise schedule gamma(t). Pass gamma= to StochasticInterpolantLoss."
            )

    @eqx.filter_jit
    def __call__(self, model, batch, key):
        x0, x1, per_sample_args = _split_paired_batch_metadata(batch)
        batch_size = jax.tree.leaves(x0)[0].shape[0]

        key1, key2, key_sel = jax.random.split(key, 3)
        t = jax.random.uniform(key1, (batch_size,))

        user_args = per_sample_args if per_sample_args is not None else self.velocity_kwargs.get("args")
        velocity_field = self._get_velocity(model)
        denoiser = self._get_denoiser(model) if self._get_denoiser is not None else None

        sel_keys = jax.random.split(key_sel, batch_size) if self.selection_protocol is not None else None
        noise_keys = jax.random.split(key2, batch_size) if self.gamma is not None else None

        def single_interpolant_loss(ti, x0i, x1i, sel_key_i, noise_key_i, *ua_i):
            mask_i = (
                self.selection_protocol(sel_key_i, x1i) if self.selection_protocol is not None else self.dynamic_mask
            )
            y0i, _, _ = state_context_partition(x0i, mask_i)
            y1i, ctxi, spec_i = state_context_partition(x1i, mask_i)

            if self.gamma is not None:
                # Generate noise matching y0i structure
                y0i_leaves, y0i_treedef = jax.tree.flatten(y0i)
                nk = jax.random.split(noise_key_i, len(y0i_leaves))
                zi = jax.tree.unflatten(
                    y0i_treedef,
                    [jax.random.normal(k, leaf.shape) for k, leaf in zip(nk, y0i_leaves)],
                )
                interp = self.interpolant(ti, y0i, y1i)
                gamma_t = self.gamma(ti)
                yt = jax.tree.map(lambda i, zp: i + gamma_t * zp, interp, zi)

                dt_interp = self.dt_interpolant(ti, y0i, y1i)
                dt_gamma_t = self.dt_gamma(ti)
                vel_target = jax.tree.map(lambda d, zp: d + dt_gamma_t * zp, dt_interp, zi)
            else:
                yt = self.interpolant(ti, y0i, y1i)
                vel_target = self.dt_interpolant(ti, y0i, y1i)
                zi = None

            index_meta = spec_i.index_meta if isinstance(spec_i, PartitionSpec) else None
            args_i = _pack_args(ctxi, index_meta, ua_i if ua_i else user_args)

            v_pred = velocity_field(ti, yt, args_i)
            vel_sq = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), v_pred, vel_target))
            vel_loss_i = sum(vel_sq)

            if denoiser is not None:
                eta_pred = denoiser(ti, yt, args_i)
                den_sq = jax.tree.leaves(jax.tree.map(lambda p, tgt: jnp.sum((p - tgt) ** 2), eta_pred, zi))
                den_loss_i = sum(den_sq)
                return vel_loss_i, den_loss_i

            return vel_loss_i

        # Setup standard mapping arguments
        vmap_args = (t, x0, x1)

        # Add sel_keys if present, otherwise None mapped as constant
        if sel_keys is not None:
            vmap_args += (sel_keys,)
            in_axes_sel = 0
        else:
            vmap_args += (None,)
            in_axes_sel = None

        # Add noise_keys if present, otherwise None mapped as constant
        if noise_keys is not None:
            vmap_args += (noise_keys,)
            in_axes_noise = 0
        else:
            vmap_args += (None,)
            in_axes_noise = None

        # Add user_args if they are per-sample
        is_per_sample = (
            user_args is not None
            and isinstance(user_args, tuple)
            and hasattr(user_args[0], "ndim")
            and user_args[0].ndim > 0
        )

        if is_per_sample:
            in_axes = (0, 0, 0, in_axes_sel, in_axes_noise) + (0,) * len(user_args)
            vmap_args += user_args
        else:
            in_axes = (0, 0, 0, in_axes_sel, in_axes_noise)

        out = jax.vmap(single_interpolant_loss, in_axes=in_axes)(*vmap_args)

        if denoiser is not None:
            vel_per_sample, den_per_sample = out
            vel_loss = jnp.mean(vel_per_sample)
            den_loss = jnp.mean(den_per_sample)
            total = vel_loss + self.denoiser_weight * den_loss
            return total, {"velocity_loss": vel_loss, "denoiser_loss": den_loss}

        vel_loss = jnp.mean(out)
        return vel_loss, {}
