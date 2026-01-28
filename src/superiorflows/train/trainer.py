"""Core training orchestration for flow-based models.

This module provides a `Trainer` class that wraps the standard training loop,
handling optimizer state, PRNG key management, and callback dispatch.
"""
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from superiorflows.train.callbacks import Callback

__all__ = ["Trainer", "train_step"]


@eqx.filter_jit
def train_step(model, opt_state, batch, key, loss_module, optimizer):
    """Execute a single gradient-descent training step.

    Args:
        model: The Equinox model to train.
        opt_state: Current optimizer state.
        batch: A batch of training data.
        key: PRNG key for stochasticity in the loss function.
        loss_module: A callable `(model, batch, key) -> loss`.
        optimizer: An Optax optimizer.

    Returns:
        A tuple `(updated_model, updated_opt_state, loss)`.
    """
    loss, grads = eqx.filter_value_and_grad(loss_module)(model, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


class Trainer:
    """Orchestrates the training loop for Equinox models.

    The Trainer manages the training state (model, optimizer, step counter)
    and dispatches events to registered callbacks.

    Attributes:
        model: The Equinox model being trained.
        optimizer: The Optax optimizer instance.
        loss_module: Callable loss function `(model, batch, key) -> loss`.
        opt_state: Current optimizer state.
        key: Current PRNG key.
        callbacks: List of registered callbacks.
        step: Current training step (0-indexed before first step).

    Example:
        >>> trainer = Trainer(model, optax.adam(1e-3), loss_fn)
        >>> trained_model = trainer.train(data_loader, max_steps=1000)
    """

    def __init__(
        self,
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        loss_module: Callable,
        seed: int = 0,
        callbacks: Optional[List[Callback]] = None,
    ):
        """Initialize the Trainer.

        Args:
            model: The Equinox model to train.
            optimizer: An Optax optimizer (e.g., `optax.adam(1e-3)`).
            loss_module: Loss callable with signature `(model, batch, key) -> loss`.
            seed: Random seed for PRNG initialization.
            callbacks: Optional list of `Callback` instances.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_module = loss_module

        self.opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        self.key = jax.random.key(seed)
        self.callbacks: List[Callback] = callbacks if callbacks is not None else []
        self.step = 0

    def add_callback(self, callback: Callback):
        """Register a callback to receive training events.

        Args:
            callback: A `Callback` instance.
        """
        self.callbacks.append(callback)

    def train(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        max_steps: int = 1000,
        val_freq: int = 100,
    ):
        """Run the training loop.

        Iterates over `train_loader`, performing gradient updates until
        `max_steps` is reached or the loader is exhausted.

        Args:
            train_loader: An iterable yielding training batches.
            val_loader: Optional iterable for validation (consumed each val run).
            max_steps: Maximum number of training steps.
            val_freq: Frequency (in steps) of validation runs.

        Returns:
            The trained model.
        """
        self._run_callbacks("on_train_start", total_steps=max_steps)

        iter_data = iter(train_loader)

        while self.step < max_steps:
            self.step += 1
            try:
                batch = next(iter_data)
            except StopIteration:
                print("Training loader exhausted.")
                break

            self.key, subkey = jax.random.split(self.key)

            self.model, self.opt_state, loss = train_step(
                self.model,
                self.opt_state,
                batch,
                subkey,
                self.loss_module,
                self.optimizer,
            )

            logs = {"loss": loss}
            self._run_callbacks("on_step_end", step=self.step, logs=logs)

            if val_loader and self.step % val_freq == 0:
                val_metrics = self._run_validation(val_loader, self.loss_module)
                self._run_callbacks("on_validation_end", metrics=val_metrics)

        self._run_callbacks("on_train_end")
        return self.model

    def _run_validation(self, val_loader, loss_module):
        """Compute validation loss over the entire validation set.

        Args:
            val_loader: An iterable yielding validation batches.
            loss_module: The loss callable.

        Returns:
            A dict `{"val_loss": float}`.
        """
        val_losses = []

        for batch in val_loader:
            self.key, subkey = jax.random.split(self.key)
            loss = loss_module(self.model, batch, subkey)
            val_losses.append(loss)

        if not val_losses:
            return {"val_loss": 0.0}

        all_losses = jnp.stack(val_losses)
        avg_loss = jnp.mean(all_losses)

        return {"val_loss": avg_loss.item()}

    def load_checkpoint(self, ckpt_path: str, step: Optional[int] = None):
        """Restore model and optimizer state from an Orbax checkpoint.

        Args:
            ckpt_path: Directory containing Orbax checkpoints.
            step: Specific step to restore. If None, restores the latest.

        Returns:
            True if restoration succeeded, False otherwise.
        """
        ckpt_path = Path(ckpt_path).resolve()
        checkpointer = ocp.CheckpointManager(ckpt_path, item_names=("state",))

        available_steps = checkpointer.all_steps()
        if not available_steps:
            print(f"No checkpoint found at {ckpt_path}")
            return False

        print(f"Found checkpoints for steps: {available_steps}")

        if step is None:
            step = checkpointer.latest_step()

        if step is None:
            print(f"No checkpoint found at {ckpt_path}")
            return False

        model_params = eqx.filter(self.model, eqx.is_array)

        target = {
            "model": model_params,
            "opt_state": self.opt_state,
            "step": 0,
        }

        try:
            args = ocp.args.Composite(state=ocp.args.StandardRestore(target))

            restored = checkpointer.restore(step, args=args)
            state = restored["state"]

            static_model = eqx.filter(self.model, eqx.is_array, inverse=True)
            self.model = eqx.combine(state["model"], static_model)

            self.opt_state = state["opt_state"]
            self.step = state["step"]

            print(f"Restored checkpoint from step {self.step}")
            return True
        except Exception as e:
            print(f"Failed to restore checkpoint: {e}")
            return False

    def _run_callbacks(self, hook_name, **kwargs):
        """Dispatch a hook to all registered callbacks."""
        for cb in self.callbacks:
            method = getattr(cb, hook_name, None)
            if method:
                method(self, **kwargs)
