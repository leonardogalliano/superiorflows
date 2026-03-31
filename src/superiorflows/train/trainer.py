"""Core training orchestration for flow-based models.

This module provides a `Trainer` class that wraps the standard training loop,
handling optimizer state, PRNG key management, and callback dispatch.
"""
from pathlib import Path
from typing import Callable, List, Optional

import equinox as eqx
import grain
import jax
import optax
import orbax.checkpoint as ocp

from superiorflows.train.callbacks import Callback

__all__ = ["Trainer", "train_step", "DatasetExhausted"]


class DatasetExhausted(Exception):
    """Raised when the dataset iterator runs out of data before ``max_steps``.

    This happens when the ``grain.MapDataset`` passed to ``Trainer.train()``
    has not been ``.repeat()``-ed and contains fewer elements than the
    requested number of training steps.

    To train for multiple epochs, call ``.repeat()`` on your dataset::

        dataset = grain.MapDataset.source(source).repeat()
        trainer.train(dataset, max_steps=10000)
    """


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
        A tuple `(updated_model, updated_opt_state, loss, grads)`.
    """
    loss, grads = eqx.filter_value_and_grad(loss_module)(model, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, grads


class Trainer:
    """Orchestrates the training loop for Equinox models.

    The Trainer manages the training state (model, optimizer, step counter)
    and dispatches events to registered callbacks. Data loading is handled
    via a ``grain.MapDataset`` that the caller assembles with any desired
    transformations (shuffle, map, batch, repeat, etc.).

    Attributes:
        model: The Equinox model being trained.
        optimizer: The Optax optimizer instance.
        loss_module: Callable loss function `(model, batch, key) -> loss`.
        opt_state: Current optimizer state.
        key: Current PRNG key.
        callbacks: List of registered callbacks.
        step: Current training step (0-indexed before first step).

    Example:
        >>> import grain
        >>> from superiorflows.data import DistributionDataSource
        >>> source = DistributionDataSource(target_dist, batch_size=32)
        >>> dataset = grain.MapDataset.source(source).repeat()
        >>> trainer = Trainer(model, optax.adam(1e-3), loss_fn)
        >>> trained_model = trainer.train(dataset, max_steps=1000)
    """

    def __init__(
        self,
        model: eqx.Module,
        optimizer: optax.GradientTransformation,
        loss_module: Callable,
        seed: int | jax.Array = 0,
        callbacks: Optional[List[Callback]] = None,
    ):
        """Initialize the Trainer.

        Args:
            model: The Equinox model to train.
            optimizer: An Optax optimizer (e.g., `optax.adam(1e-3)`).
            loss_module: Loss callable with signature `(model, batch, key) -> loss`.
            seed: Integer seed or `jax.random.PRNGKey` for initialization.
            callbacks: Optional list of `Callback` instances.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_module = loss_module

        self.opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        if isinstance(seed, int):
            self.key = jax.random.key(seed)
        else:
            self.key = seed

        self.callbacks: List[Callback] = callbacks if callbacks is not None else []
        self.step = 0
        self.logs: dict = {}
        self._data_iter = None
        self._restored_data_state = None

    def add_callback(self, callback: Callback):
        """Register a callback to receive training events.

        Args:
            callback: A `Callback` instance.
        """
        self.callbacks.append(callback)

    def train(
        self,
        dataset: grain.MapDataset,
        max_steps: int = 1000,
        read_options: Optional[grain.ReadOptions] = None,
    ):
        """Run the training loop.

        Converts the given ``dataset`` into an iterator with prefetching,
        then iterates until ``max_steps`` is reached or the dataset is
        exhausted.

        The caller is responsible for assembling the full grain pipeline
        (source → shuffle → map → batch → repeat) before passing it here.
        If the dataset is not repeated and runs out before ``max_steps``,
        a ``DatasetExhausted`` exception is raised.

        Args:
            dataset: A ``grain.MapDataset`` with all desired transformations
                already applied (shuffling, batching, repeating, etc.).
            max_steps: Maximum number of training steps.
            read_options: Grain ``ReadOptions`` for prefetching.

        Returns:
            The trained model.

        Raises:
            DatasetExhausted: If the dataset runs out before ``max_steps``.
        """
        if read_options is None:
            read_options = grain.ReadOptions(num_threads=1, prefetch_buffer_size=2)

        self._data_iter = iter(dataset.to_iter_dataset(read_options))

        if self._restored_data_state is not None:
            self._data_iter.set_state(self._restored_data_state)
            self._restored_data_state = None

        self._run_callbacks("on_train_start", total_steps=max_steps)

        while self.step < max_steps:
            self.step += 1
            try:
                batch = next(self._data_iter)
            except StopIteration:
                raise DatasetExhausted(
                    f"Dataset exhausted at step {self.step} of {max_steps}. "
                    f"Call .repeat() on your dataset for multi-epoch training."
                )

            self.key, subkey = jax.random.split(self.key)

            self.model, self.opt_state, loss, grads = train_step(
                self.model,
                self.opt_state,
                batch,
                subkey,
                self.loss_module,
                self.optimizer,
            )

            self.logs.update({"loss": loss, "grads": grads})
            self._run_callbacks("on_step_end", step=self.step, logs=self.logs)

        self._run_callbacks("on_train_end")
        return self.model

    def load_checkpoint(self, ckpt_path: str, step: Optional[int] = None):
        """Restore model and optimizer state from an Orbax checkpoint.

        Args:
            ckpt_path: Directory containing Orbax checkpoints.
            step: Specific step to restore. If None, restores the latest.

        Returns:
            True if restoration succeeded, False otherwise.
        """
        ckpt_path = Path(ckpt_path).resolve()
        checkpointer = ocp.CheckpointManager(ckpt_path, item_names=("model", "optimizer", "metadata"))

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

        try:
            args = ocp.args.Composite(
                model=ocp.args.StandardRestore(model_params),
                optimizer=ocp.args.StandardRestore(self.opt_state),
                metadata=ocp.args.JsonRestore(),
            )

            restored = checkpointer.restore(step, args=args)

            static_model = eqx.filter(self.model, eqx.is_array, inverse=True)
            self.model = eqx.combine(restored.model, static_model)
            self.opt_state = restored.optimizer

            metadata = restored.metadata
            self.step = metadata["step"]
            self._restored_data_state = metadata.get("data_state")

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
