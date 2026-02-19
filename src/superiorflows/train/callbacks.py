"""Callback system for training events.

Callbacks hook into the training loop at key points, enabling logging,
progress tracking, checkpointing, and custom behaviors without modifying
the core `Trainer` logic.
"""
from pathlib import Path
from typing import Any, Callable, Dict

import equinox as eqx
import optax
import orbax.checkpoint as ocp
from tqdm.auto import tqdm

__all__ = [
    "Callback",
    "LoggerCallback",
    "ProgressBarCallback",
    "CheckpointCallback",
    "ProfilingCallback",
    "TensorBoardLogger",
    "ESSCallback",
]


class Callback:
    """Base class for training callbacks.

    Subclass this and override the hooks you need. All methods receive
    the `trainer` instance and hook-specific keyword arguments.

    Hooks:
        on_train_start(trainer, total_steps): Called before training begins.
        on_train_end(trainer): Called after training completes.
        on_step_end(trainer, step, logs): Called after each training step.
        on_validation_end(trainer, metrics): Called after each validation run.
    """

    def on_train_start(self, trainer, **kwargs):
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        """Called after each training step.

        Args:
            trainer: The Trainer instance.
            step: Current training step (1-indexed).
            logs: Dict containing at least `{"loss": ...}`.
        """
        pass

    def on_validation_end(self, trainer, metrics: Dict[str, Any], **kwargs):
        """Called after each validation run.

        Args:
            trainer: The Trainer instance.
            metrics: Dict containing at least `{"val_loss": ...}`.
        """
        pass


class LoggerCallback(Callback):
    """Logs training metrics to stdout via tqdm.write.

    Prints formatted metrics every `log_freq` steps. Subclass and override
    `log_metrics` to redirect to custom backends (e.g., WandB, TensorBoard).

    Args:
        log_freq: Print every N steps.
    """

    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq

    def log_metrics(self, step: int, metrics: Dict[str, Any], prefix: str = ""):
        """Format and print metrics.

        Override this method to log to external systems.

        Args:
            step: Current training step.
            metrics: Dict of metric names to values.
            prefix: Optional prefix (e.g., "Train", "Val").
        """
        log_str = f"Step {step:<6}"
        if prefix:
            log_str += f" [{prefix}]".ljust(16)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                log_str += f" | {k}: {v:.4f}"
            elif hasattr(v, "item"):
                log_str += f" | {k}: {v.item():.4f}"
            else:
                log_str += f" | {k}: {v}"
        tqdm.write(log_str)

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        if step % self.log_freq == 0:
            print_logs = logs.copy()

            if "grads" in print_logs:
                grad_norm = optax.tree_utils.tree_norm(print_logs["grads"])
                print_logs["grad_norm"] = grad_norm
                del print_logs["grads"]

            self.log_metrics(step, print_logs, prefix="Train")

    def on_validation_end(self, trainer, metrics: Dict[str, Any], **kwargs):
        step = getattr(trainer, "step", 0)
        self.log_metrics(step, metrics, prefix="Val")


class ProgressBarCallback(Callback):
    """Displays a tqdm progress bar during training.

    Updates the bar every `refresh_rate` steps to minimize overhead.

    Args:
        refresh_rate: Update the progress bar every N steps.
    """

    def __init__(self, refresh_rate: int = 50):
        self.total_steps = 0
        self.refresh_rate = refresh_rate
        self.pbar = None

    def on_train_start(self, trainer, **kwargs):
        self.total_steps = kwargs.get("total_steps", 0)
        self.pbar = tqdm(total=self.total_steps, desc="Training", leave=True)

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        if self.pbar and step % self.refresh_rate == 0:
            self.pbar.update(self.refresh_rate)

            if "loss" in logs:
                val = logs["loss"]
                if hasattr(val, "item"):
                    val = val.item()
                self.pbar.set_postfix(loss=f"{val:.4f}")

    def on_train_end(self, trainer, **kwargs):
        if self.pbar:
            self.pbar.close()


class CheckpointCallback(Callback):
    """Saves model checkpoints using Orbax.

    Checkpoints are split into separate items following Orbax best practices:

    - **model**: model parameters (``StandardSave``)
    - **optimizer**: optimizer state (``StandardSave``)
    - **metadata**: step number and data iterator state (``JsonSave``)

    Uses ``CheckpointManager`` for automatic cleanup of old checkpoints.

    Args:
        ckpt_path: Directory to save checkpoints.
        save_freq: Save every N steps.
        max_to_keep: Maximum number of checkpoints to retain.
        overwrite: If True, overwrite existing checkpoints at the same step.
                   If False, skip saving if the step already exists.
    """

    def __init__(
        self,
        ckpt_path: str,
        save_freq: int = 1000,
        max_to_keep: int = 3,
        overwrite: bool = False,
    ):
        self.save_freq = save_freq
        self.overwrite = overwrite

        self.ckpt_path = Path(ckpt_path).resolve()
        self.last_saved_step = -1

        options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
        self.checkpointer = ocp.CheckpointManager(
            self.ckpt_path,
            item_names=("model", "optimizer", "metadata"),
            options=options,
        )

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        if step > 0 and step % self.save_freq == 0:
            self._save(trainer, step, force=self.overwrite)
            self.last_saved_step = step

    def on_train_end(self, trainer, **kwargs):
        step = getattr(trainer, "step", 0)
        if step > 0 and step != self.last_saved_step:
            if step in self.checkpointer.all_steps() and not self.overwrite:
                tqdm.write(f"Checkpoint for step {step} already exists. " "Skipping save on exit (overwrite=False).")
                return
            self._save(trainer, step, force=True)

        self.checkpointer.wait_until_finished()

    def _save(self, trainer, step, force=False):
        """Persist checkpoint to disk.

        Args:
            trainer: The Trainer instance.
            step: The current step number.
            force: If True, overwrite existing checkpoint at this step.
        """
        model_params = eqx.filter(trainer.model, eqx.is_array)

        if force and step in self.checkpointer.all_steps():
            self.checkpointer.wait_until_finished()
            self.checkpointer.delete(step)

        metadata = {"step": step}
        if trainer._data_iter is not None:
            metadata["data_state"] = trainer._data_iter.get_state()

        save_args = ocp.args.Composite(
            model=ocp.args.StandardSave(model_params),
            optimizer=ocp.args.StandardSave(trainer.opt_state),
            metadata=ocp.args.JsonSave(metadata),
        )

        self.checkpointer.save(step, args=save_args, force=force)


class ProfilingCallback(Callback):
    """Captures JAX/XLA traces for performance analysis.

    Wraps ``jax.profiler.start_trace`` / ``stop_trace`` to produce
    TensorBoard-compatible traces. A configurable warmup period lets
    JIT compilation finish before profiling begins.

    View traces with::

        tensorboard --logdir <log_dir>

    Args:
        log_dir: Directory to write trace files.
        warmup_steps: Steps to skip before starting the trace (lets JIT
            compilation finish so you only see steady-state performance).
        profile_steps: Number of steps to profile after warmup.
            ``None`` means profile until training ends.
    """

    def __init__(
        self,
        log_dir: str | Path = Path("tmp/profiles"),
        warmup_steps: int = 50,
        profile_steps: int | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self._is_profiling = False

    def on_train_start(self, trainer, **kwargs):
        total = kwargs.get("total_steps", "?")
        tqdm.write(
            f"[Profiling] Configured: warmup={self.warmup_steps}, "
            f"profile_steps={self.profile_steps or 'until end'}, "
            f"total_steps={total}, log_dir={self.log_dir}"
        )

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        import jax.profiler

        if step == self.warmup_steps:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            tqdm.write(f"[Profiling] Starting trace at step {step}...")
            jax.profiler.start_trace(str(self.log_dir))
            self._is_profiling = True

        elif self.profile_steps is not None and step == self.warmup_steps + self.profile_steps and self._is_profiling:
            jax.profiler.stop_trace()
            self._is_profiling = False
            tqdm.write(f"[Profiling] Stopped trace at step {step}. " f"Trace saved to {self.log_dir}")

    def on_train_end(self, trainer, **kwargs):
        import jax.profiler

        if self._is_profiling:
            jax.profiler.stop_trace()
            self._is_profiling = False
            tqdm.write(f"[Profiling] Stopped trace at training end. " f"Trace saved to {self.log_dir}")


class TensorBoardLogger(Callback):
    """Logs training metrics to TensorBoard for remote monitoring.

    Writes scalar metrics from the ``logs`` dict every ``log_freq`` steps.
    Works alongside ``LoggerCallback`` — both consume the same ``logs``.

    Requires ``tensorboard`` to be installed::

        pip install tensorboard
        # or: pip install superiorflows[monitoring]

    View logs remotely with::

        tensorboard --logdir <log_dir> --bind_all

    Args:
        log_dir: Directory for TensorBoard event files.
        log_freq: Log scalars every N steps.
    """

    def __init__(
        self,
        log_dir: str | Path = Path("tmp/tb_logs"),
        log_freq: int = 100,
        hparams: Dict[str, Any] | None = None,
    ):
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard logging requires the `tensorboardX` package. "
                "Install it with: pip install tensorboardX  "
                "or: pip install superiorflows[monitoring]"
            ) from None

        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.hparams = hparams
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir))

    def on_train_start(self, trainer, **kwargs):
        """Log hyperparameters at the start of training."""
        if self.hparams:
            self._writer.add_hparams(self.hparams, metric_dict={}, name=".", global_step=0)

    def _write_scalars(self, step: int, metrics: Dict[str, Any], prefix: str = ""):
        """Write scalar metrics to TensorBoard."""
        import optax

        for k, v in metrics.items():
            if k == "grads":
                grad_norm = optax.tree_utils.tree_norm(v)
                self._writer.add_scalar(f"{prefix}grad_norm", float(grad_norm), step)
                continue
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"{prefix}{k}", v, step)
            elif hasattr(v, "item"):
                self._writer.add_scalar(f"{prefix}{k}", v.item(), step)

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        if step % self.log_freq == 0:
            self._write_scalars(step, logs, prefix="train/")
            self._writer.flush()

    def on_validation_end(self, trainer, metrics: Dict[str, Any], **kwargs):
        step = getattr(trainer, "step", 0)
        self._write_scalars(step, metrics, prefix="val/")

    def on_train_end(self, trainer, **kwargs):
        self._writer.flush()
        self._writer.close()


class ESSCallback(Callback):
    """Monitors Effective Sample Size (ESS) during training.

    Periodically draws samples from the current flow and computes the
    normalised ESS via importance weights against a target log-probability:

    .. math::

        \\text{ESS} = \\frac{1}{N \\sum_i w_i^2}, \\quad w_i = \\text{softmax}(\\log p(x_i) - \\log q(x_i))

    The ESS is injected into the ``logs`` dict so downstream callbacks
    (``LoggerCallback``, ``TensorBoardLogger``) automatically pick it up.

    Args:
        target_log_prob: Callable ``(x) -> log_prob``. Can be a distrax
            distribution's ``.log_prob`` method, an unnormalised energy
            function, or any ``(batch,) -> (batch,)`` callable.
        base_distribution: The base/prior distribution for the flow.
        flow_kwargs: Extra kwargs passed to ``Flow(...)`` (e.g., ``dt0``).
        n_samples: Number of samples for ESS estimation.
        eval_freq: Compute ESS every N steps.
    """

    def __init__(
        self,
        target_log_prob: Callable,
        base_distribution,
        flow_kwargs: dict | None = None,
        n_samples: int = 1000,
        eval_freq: int = 250,
    ):
        self.target_log_prob = target_log_prob
        self.base_distribution = base_distribution
        self.flow_kwargs = flow_kwargs or {}
        self.n_samples = n_samples
        self.eval_freq = eval_freq

    def on_step_end(self, trainer, step: int, logs: Dict[str, Any], **kwargs):
        if step % self.eval_freq != 0:
            return

        import jax
        import jax.numpy as jnp

        from superiorflows import Flow

        flow = Flow(
            velocity_field=trainer.model,
            base_distribution=self.base_distribution,
            **self.flow_kwargs,
        )

        key, subkey = jax.random.split(trainer.key)
        X, log_q = flow.sample_and_log_prob(seed=subkey, sample_shape=(self.n_samples,))

        log_p = self.target_log_prob(X)
        log_weights = log_p - log_q
        weights = jax.nn.softmax(log_weights)

        ess = 1.0 / jnp.sum(weights**2) / self.n_samples

        logs["ess"] = ess
