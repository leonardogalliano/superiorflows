"""Callback system for training events.

Callbacks hook into the training loop at key points, enabling logging,
progress tracking, checkpointing, and custom behaviors without modifying
the core `Trainer` logic.
"""
from pathlib import Path
from typing import Any, Dict

import equinox as eqx
import orbax.checkpoint as ocp
from tqdm.auto import tqdm

__all__ = ["Callback", "LoggerCallback", "ProgressBarCallback", "CheckpointCallback"]


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
            self.log_metrics(step, logs, prefix="Train")

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

    Checkpoints include the model parameters, optimizer state, and step number.
    Uses `CheckpointManager` for automatic cleanup of old checkpoints.

    Args:
        ckpt_path: Directory to save checkpoints.
        save_freq: Save every N steps.
        max_to_keep: Maximum number of checkpoints to retain.
        overwrite: If True, overwrite existing checkpoints at the same step.
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
            item_names=("state",),
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

        ckpt = {
            "model": model_params,
            "opt_state": trainer.opt_state,
            "step": step,
        }

        save_args = ocp.args.Composite(state=ocp.args.StandardSave(ckpt))

        self.checkpointer.save(step, args=save_args, force=force)
