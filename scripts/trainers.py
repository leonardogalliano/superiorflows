"""
Trainers for Continuous Normalizing Flows.

This module provides trainer classes for CNF models. The base pattern separates
trainable (velocity_field) from non-trainable (base_distribution) components
using closures, which is JAX-friendly and efficient (no recompilation).
"""

from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .flow import Flow

__all__ = ["TrainState", "MaximumLikelihoodTrainer"]


class TrainState(NamedTuple):
    """Immutable training state container."""

    velocity_field: eqx.Module
    opt_state: optax.OptState
    step: int
    key: jax.Array


class BaseTrainer(ABC):
    """
    Abstract base class for CNF trainers.

    The key pattern for training CNFs with optax when using distrax distributions:
    - velocity_field (eqx.Module) is the traced/differentiable argument
    - base_distribution (distrax.Distribution) is closed over (static)
    - Flow is reconstructed inside loss_fn - this is just pytree construction,
      not JAX tracing, so it's free at runtime (no recompilation)

    Subclasses implement different training objectives:
    - MaximumLikelihoodTrainer: maximize log p(x) for samples from target
    - ReverseKLTrainer: minimize KL(q||p) [future]
    - FlowMatchingTrainer: flow matching objective [future]
    """

    def __init__(
        self,
        base_distribution,
        optimizer: optax.GradientTransformation,
        batch_size: int = 256,
    ):
        """
        Initialize trainer.

        Args:
            base_distribution: The base distribution for the flow (e.g., Gaussian)
            optimizer: Optax optimizer
            batch_size: Number of samples per training step
        """
        self.base_distribution = base_distribution
        self.optimizer = optimizer
        self.batch_size = batch_size
        self._train_step_fn = None

    @abstractmethod
    def loss_fn(self, flow: Flow, batch: jax.Array, key: jax.Array) -> jax.Array:
        """
        Compute the training loss.

        Args:
            flow: The flow model
            batch: Batch of training data
            key: PRNG key for stochastic operations

        Returns:
            Scalar loss value
        """
        pass

    @abstractmethod
    def sample_batch(self, key: jax.Array) -> jax.Array:
        """
        Sample a training batch.

        Args:
            key: PRNG key

        Returns:
            Batch of training samples
        """
        pass

    def init(self, velocity_field: eqx.Module, key: jax.Array) -> TrainState:
        """
        Initialize training state.

        Args:
            velocity_field: Initial velocity field model
            key: PRNG key

        Returns:
            Initial TrainState
        """
        opt_state = self.optimizer.init(eqx.filter(velocity_field, eqx.is_array))
        return TrainState(
            velocity_field=velocity_field,
            opt_state=opt_state,
            step=0,
            key=key,
        )

    def _build_train_step(self) -> Callable:
        """Build the JIT-compiled training step function."""
        base_dist = self.base_distribution
        optimizer = self.optimizer

        @eqx.filter_jit
        def train_step(state: TrainState, batch: jax.Array) -> tuple[TrainState, dict]:
            key, subkey = jax.random.split(state.key)

            @eqx.filter_value_and_grad
            def compute_loss(vf):
                # Flow reconstruction is NOT traced - just pytree construction
                flow = Flow(velocity_field=vf, base_distribution=base_dist)
                return self.loss_fn(flow, batch, subkey)

            loss, grads = compute_loss(state.velocity_field)
            updates, new_opt_state = optimizer.update(grads, state.opt_state, state.velocity_field)
            new_vf = eqx.apply_updates(state.velocity_field, updates)

            new_state = TrainState(
                velocity_field=new_vf,
                opt_state=new_opt_state,
                step=state.step + 1,
                key=key,
            )
            metrics = {"loss": loss, "step": state.step}
            return new_state, metrics

        return train_step

    def step(self, state: TrainState, batch: jax.Array) -> tuple[TrainState, dict]:
        """
        Perform a single training step.

        Args:
            state: Current training state
            batch: Batch of training data

        Returns:
            Tuple of (new_state, metrics_dict)
        """
        if self._train_step_fn is None:
            self._train_step_fn = self._build_train_step()
        return self._train_step_fn(state, batch)

    def train(
        self,
        state: TrainState,
        n_steps: int,
        log_every: int = 100,
        callback: Optional[Callable] = None,
    ) -> tuple[TrainState, list[dict]]:
        """
        Run training loop.

        Args:
            state: Initial training state
            n_steps: Number of training steps
            log_every: Print loss every N steps
            callback: Optional callback(state, metrics) called every step

        Returns:
            Tuple of (final_state, list_of_metrics)
        """
        all_metrics = []

        for i in range(n_steps):
            key, subkey = jax.random.split(state.key)
            state = state._replace(key=key)
            batch = self.sample_batch(subkey)
            state, metrics = self.step(state, batch)
            all_metrics.append(metrics)

            if callback is not None:
                callback(state, metrics)

            if log_every > 0 and (i + 1) % log_every == 0:
                print(f"Step {metrics['step']}: loss = {metrics['loss']:.4f}")

        return state, all_metrics

    def get_flow(self, state: TrainState) -> Flow:
        """
        Get the current flow model from training state.

        Args:
            state: Training state

        Returns:
            Flow model with current velocity_field
        """
        return Flow(
            velocity_field=state.velocity_field,
            base_distribution=self.base_distribution,
        )


class MaximumLikelihoodTrainer(BaseTrainer):
    """
    Maximum likelihood trainer for CNFs.

    Maximizes log p(x) for samples from a target distribution by minimizing
    the negative log-likelihood: L = -E_{x ~ target}[log p_flow(x)]
    """

    def __init__(
        self,
        base_distribution,
        target_distribution,
        optimizer: optax.GradientTransformation,
        batch_size: int = 256,
    ):
        """
        Initialize maximum likelihood trainer.

        Args:
            base_distribution: The base distribution for the flow
            target_distribution: The target distribution to learn
            optimizer: Optax optimizer
            batch_size: Number of samples per training step
        """
        super().__init__(base_distribution, optimizer, batch_size)
        self.target_distribution = target_distribution

    def loss_fn(self, flow: Flow, batch: jax.Array, key: jax.Array) -> jax.Array:
        """Negative log-likelihood loss."""
        return -jnp.mean(flow.log_prob(batch))

    def sample_batch(self, key: jax.Array) -> jax.Array:
        """Sample from target distribution."""
        return self.target_distribution.sample(seed=key, sample_shape=(self.batch_size,))

    def _build_train_step(self) -> Callable:
        """Build JIT-compiled training step with sampling included."""
        base_dist = self.base_distribution
        target_dist = self.target_distribution
        optimizer = self.optimizer
        batch_size = self.batch_size

        @eqx.filter_jit
        def train_step(state: TrainState) -> tuple[TrainState, dict]:
            key, sample_key, grad_key = jax.random.split(state.key, 3)

            # Sample batch INSIDE JIT for efficiency
            batch = target_dist.sample(seed=sample_key, sample_shape=(batch_size,))

            @eqx.filter_value_and_grad
            def compute_loss(vf):
                flow = Flow(velocity_field=vf, base_distribution=base_dist)
                return -jnp.mean(flow.log_prob(batch))

            loss, grads = compute_loss(state.velocity_field)
            updates, new_opt_state = optimizer.update(grads, state.opt_state, state.velocity_field)
            new_vf = eqx.apply_updates(state.velocity_field, updates)

            new_state = TrainState(
                velocity_field=new_vf,
                opt_state=new_opt_state,
                step=state.step + 1,
                key=key,
            )
            metrics = {"loss": loss, "step": state.step}
            return new_state, metrics

        return train_step

    def step(self, state: TrainState, batch: jax.Array = None) -> tuple[TrainState, dict]:
        """
        Perform a single training step.

        For MaximumLikelihoodTrainer, batch is ignored as sampling is done inside JIT.
        """
        if self._train_step_fn is None:
            self._train_step_fn = self._build_train_step()
        return self._train_step_fn(state)

    def train(
        self,
        state: TrainState,
        n_steps: int,
        log_every: int = 100,
        callback: Optional[Callable] = None,
    ) -> tuple[TrainState, list[dict]]:
        """Run training loop with optimized step function."""
        if self._train_step_fn is None:
            self._train_step_fn = self._build_train_step()

        all_metrics = []

        for i in range(n_steps):
            state, metrics = self._train_step_fn(state)
            all_metrics.append(metrics)

            if callback is not None:
                callback(state, metrics)

            if log_every > 0 and (i + 1) % log_every == 0:
                print(f"Step {metrics['step']}: loss = {metrics['loss']:.4f}")

        return state, all_metrics


# Future trainers to implement:
#
# class ReverseKLTrainer(BaseTrainer):
#     """
#     Reverse KL trainer: minimize KL(q_flow || p_target)
#
#     Uses samples from the flow and importance weighting.
#     """
#     pass
#
#
# class FlowMatchingTrainer(BaseTrainer):
#     """
#     Flow matching trainer.
#
#     Regresses velocity field to match optimal transport path.
#     """
#     pass
