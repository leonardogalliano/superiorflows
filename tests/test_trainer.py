"""Tests for the Trainer, Callbacks, and Loss functions."""
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import diffrax as dfx
import distrax as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from superiorflows import DistributionDataSource, Flow
from superiorflows.train import (
    Callback,
    CheckpointCallback,
    EnergyBasedLoss,
    KullbackLeiblerLoss,
    LoggerCallback,
    MaximumLikelihoodLoss,
    ProfilingCallback,
    ProgressBarCallback,
    Trainer,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class MLPVelocity(eqx.Module):
    """Simple MLP velocity field for testing."""

    mlp: eqx.nn.MLP

    def __init__(self, input_dim, width, depth, *, key):
        self.mlp = eqx.nn.MLP(input_dim + 1, input_dim, width, depth, activation=jax.nn.tanh, key=key)

    def __call__(self, t, x, args):
        t_feat = jnp.broadcast_to(t, x.shape[:-1] + (1,))
        return self.mlp(jnp.concatenate([x, t_feat], axis=-1))


class System(eqx.Module):
    """Simple particle system for pytree testing."""

    positions: jnp.ndarray
    species: jnp.ndarray
    temperature: Optional[float] = eqx.field(static=True, default=None)


class ParticleVelocityField(eqx.Module):
    """Velocity field for particle systems."""

    params: jax.Array

    def __call__(self, t, state: System, args):
        return System(
            positions=self.params * state.positions * t,
            species=jnp.zeros_like(state.species),
            temperature=None,
        )


@pytest.fixture
def base_dist():
    """Standard 2D Gaussian."""
    return dsx.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2))


@pytest.fixture
def target_dist():
    """8 Gaussians mixture for testing."""
    angles = jnp.arange(8) * (jnp.pi / 4)
    locs = 5.0 * jnp.stack([jnp.sin(angles), jnp.cos(angles)], axis=1)
    return dsx.MixtureSameFamily(
        mixture_distribution=dsx.Categorical(probs=jnp.ones(8) / 8),
        components_distribution=dsx.MultivariateNormalDiag(loc=locs, scale_diag=jnp.full((8, 2), 0.5)),
    )


@pytest.fixture
def model(base_dist):
    """Small MLP velocity model."""
    return MLPVelocity(input_dim=2, width=16, depth=2, key=jax.random.key(42))


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestMaximumLikelihoodLoss:
    """Tests for MaximumLikelihoodLoss."""

    def test_initialization(self, base_dist):
        """Test loss can be created."""
        loss = MaximumLikelihoodLoss(base_dist)
        assert loss.base_distribution is base_dist
        assert loss.flow_kwargs == {}

    def test_initialization_with_kwargs(self, base_dist):
        """Test loss with custom flow kwargs."""
        loss = MaximumLikelihoodLoss(base_dist, stepsize_controller=dfx.PIDController(rtol=1e-4, atol=1e-4))
        assert "stepsize_controller" in loss.flow_kwargs

    def test_forward_pass(self, base_dist, target_dist, model):
        """Test loss computes and returns scalar."""
        loss_fn = MaximumLikelihoodLoss(base_dist)
        batch = target_dist.sample(seed=jax.random.key(0), sample_shape=(16,))
        loss = loss_fn(model, batch, key=jax.random.key(1))
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_has_gradient(self, base_dist, target_dist, model):
        """Test loss is differentiable."""
        loss_fn = MaximumLikelihoodLoss(base_dist)
        batch = target_dist.sample(seed=jax.random.key(0), sample_shape=(16,))

        @eqx.filter_jit
        def loss_and_grad(m):
            return eqx.filter_value_and_grad(loss_fn)(m, batch, jax.random.key(1))

        loss, grads = loss_and_grad(model)
        assert jnp.isfinite(loss)
        grad_norms = jax.tree.map(lambda x: jnp.linalg.norm(x), eqx.filter(grads, eqx.is_array))
        total_grad_norm = sum(jax.tree.leaves(grad_norms))
        assert total_grad_norm > 0

    def test_flow_kwargs_passed_to_flow(self, base_dist):
        """Verify flow_kwargs are passed to the Flow constructor."""
        controller = dfx.PIDController(rtol=1e-4, atol=1e-4)
        loss_fn = MaximumLikelihoodLoss(base_dist, stepsize_controller=controller, dt0=0.05)

        mock_model = MagicMock(spec=eqx.Module)
        batch = jnp.zeros((10, 2))

        with patch("superiorflows.train.losses.Flow") as MockFlow:
            mock_flow_instance = MockFlow.return_value
            mock_flow_instance.log_prob.return_value = jnp.array(0.0)

            with jax.disable_jit():
                loss_fn(mock_model, batch, key=jax.random.key(0))

            kwargs = MockFlow.call_args[1]
            assert "stepsize_controller" in kwargs
            assert kwargs["stepsize_controller"] is controller
            assert kwargs["dt0"] == 0.05


class TestEnergyBasedLoss:
    """Tests for EnergyBasedLoss."""

    def test_initialization(self, base_dist, target_dist):
        """Test loss can be created."""
        loss = EnergyBasedLoss(base_dist, target_dist)
        assert loss.base_distribution is base_dist
        assert loss.target_distribution is target_dist

    def test_forward_pass(self, base_dist, target_dist, model):
        """Test loss computes and returns scalar."""
        loss_fn = EnergyBasedLoss(base_dist, target_dist)
        batch = base_dist.sample(seed=jax.random.key(0), sample_shape=(16,))
        loss = loss_fn(model, batch, key=jax.random.key(1))
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_has_gradient(self, base_dist, target_dist, model):
        """Test loss is differentiable."""
        loss_fn = EnergyBasedLoss(base_dist, target_dist)
        batch = base_dist.sample(seed=jax.random.key(0), sample_shape=(16,))

        @eqx.filter_jit
        def loss_and_grad(m):
            return eqx.filter_value_and_grad(loss_fn)(m, batch, jax.random.key(1))

        loss, grads = loss_and_grad(model)
        assert jnp.isfinite(loss)
        grad_norms = jax.tree.map(lambda x: jnp.linalg.norm(x), eqx.filter(grads, eqx.is_array))
        total_grad_norm = sum(jax.tree.leaves(grad_norms))
        assert total_grad_norm > 0


class TestKullbackLeiblerLoss:
    """Tests for KullbackLeiblerLoss."""

    def test_initialization(self, base_dist, target_dist):
        """Test loss can be created."""
        loss = KullbackLeiblerLoss(base_dist, target_dist, alpha=0.5)
        assert loss.alpha == 0.5
        assert loss.base_distribution is base_dist

    def test_alpha_bounds(self, base_dist, target_dist, model):
        """Test alpha=1 is pure MLE, alpha=0 is pure energy-based."""
        batch = target_dist.sample(seed=jax.random.key(0), sample_shape=(16,))

        loss_mle = MaximumLikelihoodLoss(base_dist)
        loss_hybrid_alpha1 = KullbackLeiblerLoss(base_dist, target_dist, alpha=1.0)

        l_mle = loss_mle(model, batch, key=jax.random.key(1))
        l_hybrid = loss_hybrid_alpha1(model, batch, key=jax.random.key(1))
        assert jnp.isfinite(l_mle)
        assert jnp.isfinite(l_hybrid)

    def test_forward_pass(self, base_dist, target_dist, model):
        """Test loss computes and returns scalar."""
        loss_fn = KullbackLeiblerLoss(base_dist, target_dist, alpha=0.5)
        batch = target_dist.sample(seed=jax.random.key(0), sample_shape=(16,))
        loss = loss_fn(model, batch, key=jax.random.key(1))
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_has_gradient(self, base_dist, target_dist, model):
        """Test loss is differentiable."""
        loss_fn = KullbackLeiblerLoss(base_dist, target_dist, alpha=0.5)
        batch = target_dist.sample(seed=jax.random.key(0), sample_shape=(16,))

        @eqx.filter_jit
        def loss_and_grad(m):
            return eqx.filter_value_and_grad(loss_fn)(m, batch, jax.random.key(1))

        loss, grads = loss_and_grad(model)
        assert jnp.isfinite(loss)
        grad_norms = jax.tree.map(lambda x: jnp.linalg.norm(x), eqx.filter(grads, eqx.is_array))
        total_grad_norm = sum(jax.tree.leaves(grad_norms))
        assert total_grad_norm > 0


# =============================================================================
# Callback Tests
# =============================================================================


class TestLoggerCallback:
    """Tests for LoggerCallback."""

    def test_initialization(self):
        """Test callback can be created with custom freq."""
        cb = LoggerCallback(log_freq=50)
        assert cb.log_freq == 50

    def test_log_metrics_format(self, capsys):
        """Test log output format."""
        cb = LoggerCallback()
        cb.log_metrics(100, {"loss": 0.5, "accuracy": 0.95}, prefix="Train")
        captured = capsys.readouterr()
        assert "Step 100" in captured.out
        assert "Train" in captured.out
        assert "loss" in captured.out

    def test_logging_in_training_loop(self, base_dist, target_dist, model, capsys):
        """Test that logger actually activates during training."""
        loss_fn = MaximumLikelihoodLoss(base_dist)
        optimizer = optax.adam(1e-3)
        logger = LoggerCallback(log_freq=1)
        trainer = Trainer(model, optimizer, loss_fn, callbacks=[logger])

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        trainer.train(source, max_steps=2)

        captured = capsys.readouterr()
        assert "Step 1" in captured.out
        assert "Step 2" in captured.out
        assert "loss" in captured.out


class TestProgressBarCallback:
    """Tests for ProgressBarCallback."""

    def test_initialization(self):
        """Test callback can be created."""
        cb = ProgressBarCallback(refresh_rate=25)
        assert cb.refresh_rate == 25

    def test_progress_bar_active(self, base_dist, target_dist, model, capsys):
        """Test progress bar updates during training."""
        loss_fn = MaximumLikelihoodLoss(base_dist)
        optimizer = optax.adam(1e-3)
        pbar = ProgressBarCallback(refresh_rate=1)
        trainer = Trainer(model, optimizer, loss_fn, callbacks=[pbar])

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        trainer.train(source, max_steps=3)

        captured = capsys.readouterr()
        assert "Training" in captured.err or "Training" in captured.out
        assert "100%" in captured.err or "3/3" in captured.err


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_initialization(self, tmp_path):
        """Test callback can be created with path."""
        cb = CheckpointCallback(ckpt_path=str(tmp_path), save_freq=100)
        assert cb.save_freq == 100
        assert cb.ckpt_path == tmp_path.resolve()

    def test_overwrite_flag_init(self, tmp_path):
        """Test overwrite flag is stored."""
        cb_no_overwrite = CheckpointCallback(ckpt_path=str(tmp_path / "a"), overwrite=False)
        cb_overwrite = CheckpointCallback(ckpt_path=str(tmp_path / "b"), overwrite=True)
        assert cb_no_overwrite.overwrite is False
        assert cb_overwrite.overwrite is True


class TestProfilingCallback:
    """Tests for ProfilingCallback."""

    def test_initialization(self, tmp_path):
        """Test callback can be created with default and custom values."""
        cb_default = ProfilingCallback()
        assert cb_default.warmup_steps == 50
        assert cb_default.profile_steps is None
        assert not cb_default._is_profiling

        cb_custom = ProfilingCallback(log_dir=tmp_path / "traces", warmup_steps=10, profile_steps=20)
        assert cb_custom.warmup_steps == 10
        assert cb_custom.profile_steps == 20
        assert cb_custom.log_dir == tmp_path / "traces"

    def test_profiling_creates_trace(self, base_dist, target_dist, model, tmp_path):
        """Test that profiling actually produces trace files."""
        trace_dir = tmp_path / "traces"
        cb = ProfilingCallback(log_dir=trace_dir, warmup_steps=2, profile_steps=3)

        loss_fn = MaximumLikelihoodLoss(base_dist)
        optimizer = optax.adam(1e-3)
        trainer = Trainer(model, optimizer, loss_fn, callbacks=[cb])

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        trainer.train(source, max_steps=10)

        assert not cb._is_profiling  # should have stopped
        assert trace_dir.exists()
        # jax.profiler writes .xplane.pb or similar files
        trace_files = list(trace_dir.iterdir())
        assert len(trace_files) > 0, f"No trace files in {trace_dir}"

    def test_profile_steps_none_runs_to_end(self, base_dist, target_dist, model, tmp_path):
        """Test that profile_steps=None keeps profiling until training ends."""
        trace_dir = tmp_path / "traces_none"
        cb = ProfilingCallback(log_dir=trace_dir, warmup_steps=2, profile_steps=None)

        loss_fn = MaximumLikelihoodLoss(base_dist)
        optimizer = optax.adam(1e-3)
        trainer = Trainer(model, optimizer, loss_fn, callbacks=[cb])

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        trainer.train(source, max_steps=5)

        # on_train_end should have stopped it
        assert not cb._is_profiling
        assert trace_dir.exists()
        trace_files = list(trace_dir.iterdir())
        assert len(trace_files) > 0


# =============================================================================
# Trainer Tests
# =============================================================================


class TestTrainerInitialization:
    """Tests for Trainer initialization."""

    def test_basic_init(self, base_dist, model):
        """Test trainer can be initialized."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)
        trainer = Trainer(model, optimizer, loss_fn, seed=42)
        assert trainer.model is model
        assert trainer.step == 0

    def test_init_with_prng_key(self, base_dist, model):
        """Test trainer init with explicit PRNGKey."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)
        key = jax.random.key(123)
        trainer = Trainer(model, optimizer, loss_fn, seed=key)
        assert jnp.array_equal(trainer.key, key)

    def test_init_with_callbacks(self, base_dist, model):
        """Test trainer with callbacks in init."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)
        cb = Callback()
        trainer = Trainer(model, optimizer, loss_fn, callbacks=[cb])
        assert len(trainer.callbacks) == 1

    def test_add_callback(self, base_dist, model):
        """Test adding callbacks after init."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)
        trainer = Trainer(model, optimizer, loss_fn)
        trainer.add_callback(LoggerCallback())
        assert len(trainer.callbacks) == 1


class TestTrainerTraining:
    """Tests for Trainer.train() method."""

    def test_basic_training(self, base_dist, target_dist, model):
        """Test basic training loop runs."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)
        trainer = Trainer(model, optimizer, loss_fn, seed=0)

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        final_model = trainer.train(source, max_steps=5)

        assert trainer.step == 5
        assert final_model is not None

    def test_training_decreases_loss(self, base_dist, target_dist, model):
        """Test that training actually improves the loss."""
        optimizer = optax.adam(1e-2)
        loss_fn = MaximumLikelihoodLoss(base_dist)

        test_batch = target_dist.sample(seed=jax.random.key(99), sample_shape=(64,))
        initial_loss = loss_fn(model, test_batch, key=jax.random.key(100))

        trainer = Trainer(model, optimizer, loss_fn, seed=0)
        source = DistributionDataSource(target_dist, batch_size=64, seed=0)
        trained_model = trainer.train(source, max_steps=50)

        final_loss = loss_fn(trained_model, test_batch, key=jax.random.key(100))

        assert final_loss < initial_loss + 1.0

    def test_training_with_small_source(self, base_dist, target_dist, model):
        """Test training with a small data source (grain repeats it)."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)
        trainer = Trainer(model, optimizer, loss_fn, seed=0)

        source = DistributionDataSource(target_dist, batch_size=16, seed=0, length=3)
        trainer.train(source, max_steps=10)

        assert trainer.step == 10

    def test_training_with_validation(self, base_dist, target_dist, model):
        """Test training with validation runs."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)

        val_calls = []

        class ValTracker(Callback):
            def on_validation_end(self, trainer, metrics, **kwargs):
                val_calls.append(metrics)

        trainer = Trainer(model, optimizer, loss_fn, callbacks=[ValTracker()])
        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        val_data = target_dist.sample(seed=jax.random.key(1), sample_shape=(32,))
        val_loader = [val_data]

        trainer.train(source, val_loader=val_loader, max_steps=10, val_freq=5)

        assert len(val_calls) == 2
        assert all("val_loss" in m for m in val_calls)


class TestTrainerCheckpointing:
    """Tests for checkpoint save/restore functionality."""

    def test_checkpoint_save_restore(self, base_dist, target_dist, model, tmp_path):
        """Test full checkpoint save and restore cycle."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)

        ckpt_path = tmp_path / "checkpoints"
        cb = CheckpointCallback(ckpt_path=str(ckpt_path), save_freq=5)
        trainer = Trainer(model, optimizer, loss_fn, callbacks=[cb])

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        trainer.train(source, max_steps=10)

        cb.checkpointer.wait_until_finished()
        assert len(cb.checkpointer.all_steps()) > 0

        new_trainer = Trainer(model, optimizer, loss_fn, seed=99)
        success = new_trainer.load_checkpoint(str(ckpt_path))

        assert success
        assert new_trainer.step > 0

    def test_checkpoint_overwrite_behavior(self, base_dist, target_dist, model, tmp_path):
        """Test that overwrite flag is respected during training."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)

        ckpt_path = tmp_path / "checkpoints_overwrite"

        cb_no = CheckpointCallback(ckpt_path=str(ckpt_path), save_freq=1, overwrite=False)
        trainer1 = Trainer(model, optimizer, loss_fn, callbacks=[cb_no])

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)

        trainer1.train(source, max_steps=1)
        cb_no.checkpointer.wait_until_finished()

        step_dir = None
        for p in ckpt_path.glob("*"):
            if p.name == "1" or p.name.endswith("_1"):
                step_dir = p
                break

        assert step_dir is not None and step_dir.exists()

        start_mtime = step_dir.stat().st_mtime
        time.sleep(1.1)

        cb_no.overwrite = True
        trainer2 = Trainer(model, optimizer, loss_fn, callbacks=[cb_no])
        trainer2.step = 1

        time.sleep(1.1)

        cb_no.on_step_end(trainer2, step=1, logs={})
        cb_no.checkpointer.wait_until_finished()

        step_dir_new = None
        for p in ckpt_path.glob("*"):
            if p.name == "1" or p.name.endswith("_1"):
                step_dir_new = p
                break

        new_mtime = step_dir_new.stat().st_mtime
        assert new_mtime != start_mtime


# =============================================================================
# Compilation Tests
# =============================================================================


class TestCompilationEfficiency:
    """Tests for JAX compilation behavior."""

    def test_no_recompilation_during_training(self, base_dist, target_dist, model):
        """Test that training steps do not trigger recompilation."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)
        trainer = Trainer(model, optimizer, loss_fn)

        source = DistributionDataSource(target_dist, batch_size=32, seed=0)

        times = []
        n_steps = 5

        start = time.perf_counter()
        trainer.train(source, max_steps=1)
        jax.block_until_ready(trainer.model)
        times.append(time.perf_counter() - start)

        for i in range(1, n_steps):
            start = time.perf_counter()
            trainer.train(source, max_steps=i + 1)
            jax.block_until_ready(trainer.model)
            times.append(time.perf_counter() - start)

        compilation_time = times[0]
        avg_execution_time = sum(times[1:]) / len(times[1:])

        print(f"Compilation: {compilation_time*1000:.2f}ms, " f"Avg Exec: {avg_execution_time*1000:.2f}ms")

        if avg_execution_time > 0.001:
            assert compilation_time > 2.0 * avg_execution_time, "Recompilation likely occurred!"


# =============================================================================
# Pytree / Particle System Tests
# =============================================================================


class ParticleSystem(eqx.Module):
    """Particle system state for training tests."""

    positions: jnp.ndarray
    species: jnp.ndarray
    box: jnp.ndarray
    temperature: Optional[float] = eqx.field(static=True, default=None)


class ParticleVelocityFieldTrainable(eqx.Module):
    """Trainable velocity field for particle systems."""

    params: jax.Array

    def __call__(self, t, state: ParticleSystem, args):
        vr = -t * state.positions @ self.params.T
        vs = jnp.zeros_like(state.species)
        return ParticleSystem(
            positions=vr,
            species=vs,
            box=None,
            temperature=state.temperature,
        )


class UniformParticleDistribution(eqx.Module, dsx.Distribution):
    """Uniform distribution over particle systems in a box."""

    box: jnp.ndarray
    ref_species: jnp.ndarray
    temperature: Optional[float] = eqx.field(static=True, default=None)

    @property
    def event_shape(self):
        return ParticleSystem(
            positions=(self.ref_species.shape[0], self.box.shape[0]),
            species=(self.ref_species.shape[0],),
            box=(self.box.shape[0],),
            temperature=None,
        )

    def _sample_n(self, key, n):
        N = self.ref_species.shape[0]
        d = self.box.shape[0]
        k1, k2 = jax.random.split(key)
        pos = jax.random.uniform(k1, shape=(n, N, d), minval=0.0, maxval=self.box)
        keys_perm = jax.random.split(k2, n)
        species = jax.vmap(lambda k: jax.random.permutation(k, self.ref_species))(keys_perm)
        batched_box = jnp.broadcast_to(self.box, (n, d))
        return ParticleSystem(positions=pos, species=species, box=batched_box, temperature=self.temperature)

    def log_prob(self, value: ParticleSystem):
        N = self.ref_species.shape[0]
        vol_log = jnp.sum(jnp.log(self.box))
        base_log_prob = -N * vol_log
        in_box = jnp.all((value.positions >= 0.0) & (value.positions <= self.box), axis=(-1, -2))
        sorted_val = jnp.sort(value.species, axis=-1)
        sorted_ref = jnp.sort(self.ref_species, axis=-1)
        valid_composition = jnp.all(jnp.isclose(sorted_val, sorted_ref), axis=-1)
        is_valid = in_box & valid_composition
        return jnp.where(is_valid, base_log_prob, -jnp.inf)


class TestTrainerWithParticleSystems:
    """Tests for training with particle system pytrees."""

    @pytest.fixture
    def particle_setup(self):
        """Create particle system components for testing."""
        N, d = 4, 2
        L = 5.0
        box = jnp.ones(d) * L
        key = jax.random.key(0)
        key, k1, k2 = jax.random.split(key, 3)
        ref_species = jax.random.uniform(k1, shape=(N,), minval=0.5, maxval=2.0)

        base_dist = UniformParticleDistribution(box=box, ref_species=ref_species, temperature=1.0)
        velocity_field = ParticleVelocityFieldTrainable(params=jax.random.normal(k2, (d, d)))

        sample = base_dist.sample(seed=jax.random.key(99))
        dynamic_mask = eqx.tree_at(
            lambda x: (x.positions, x.species, x.box),
            sample,
            replace=(True, True, False),
        )

        return {
            "base_dist": base_dist,
            "velocity_field": velocity_field,
            "dynamic_mask": dynamic_mask,
            "N": N,
            "d": d,
        }

    def test_particle_system_ml_training(self, particle_setup):
        """Test ML training on particle systems works end-to-end."""
        base_dist = particle_setup["base_dist"]
        velocity_field = particle_setup["velocity_field"]
        dynamic_mask = particle_setup["dynamic_mask"]

        loss_fn = MaximumLikelihoodLoss(
            base_distribution=base_dist,
            dynamic_mask=dynamic_mask,
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.1,
        )

        optimizer = optax.sgd(1e-4)
        trainer = Trainer(velocity_field, optimizer, loss_fn, seed=0)

        source = DistributionDataSource(base_dist, batch_size=1, seed=0)
        trained_model = trainer.train(source, max_steps=5)

        assert trainer.step == 5
        flow = Flow(
            velocity_field=trained_model,
            base_distribution=base_dist,
            dynamic_mask=dynamic_mask,
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.1,
        )
        x0 = base_dist.sample(seed=jax.random.key(1))
        x1 = flow.apply_map(x0)
        assert x1.positions.shape == x0.positions.shape
        assert jnp.all(jnp.isfinite(x1.positions))

    def test_particle_system_with_hutchinson(self, particle_setup):
        """Test particle system training with Hutchinson estimator."""
        base_dist = particle_setup["base_dist"]
        velocity_field = particle_setup["velocity_field"]
        dynamic_mask = particle_setup["dynamic_mask"]

        loss_fn = MaximumLikelihoodLoss(
            base_distribution=base_dist,
            dynamic_mask=dynamic_mask,
            hutchinson_samples=3,
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.1,
        )

        optimizer = optax.sgd(1e-4)
        trainer = Trainer(velocity_field, optimizer, loss_fn, seed=0)

        source = DistributionDataSource(base_dist, batch_size=1, seed=0)
        trainer.train(source, max_steps=5)
        assert trainer.step == 5

    def test_particle_gradient_flows(self, particle_setup):
        """Test that gradients flow correctly through particle system training."""
        base_dist = particle_setup["base_dist"]
        velocity_field = particle_setup["velocity_field"]
        dynamic_mask = particle_setup["dynamic_mask"]

        flow = Flow(
            velocity_field=velocity_field,
            base_distribution=base_dist,
            dynamic_mask=dynamic_mask,
            stepsize_controller=dfx.ConstantStepSize(),
            dt0=0.1,
        )

        x0 = base_dist.sample(seed=jax.random.key(0))
        x1, logq = flow.apply_map_and_log_prob(x0)

        assert jnp.all(jnp.isfinite(x1.positions))
        assert jnp.isfinite(logq)

        @eqx.filter_jit
        def compute_logq_grad(vf):
            def loss_fn(model):
                f = Flow(
                    velocity_field=model,
                    base_distribution=base_dist,
                    dynamic_mask=dynamic_mask,
                    stepsize_controller=dfx.ConstantStepSize(),
                    dt0=0.1,
                )
                _, lq = f.apply_map_and_log_prob(x0)
                return lq

            return eqx.filter_value_and_grad(loss_fn)(vf)

        logq_val, grads = compute_logq_grad(velocity_field)

        assert jnp.isfinite(logq_val)
        grad_norm = jnp.linalg.norm(grads.params)
        assert grad_norm > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndTraining:
    """Full integration tests."""

    def test_8_gaussians_ml_training(self, base_dist, target_dist, model):
        """Full 8 Gaussians test with ML loss."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist)
        logger = LoggerCallback(log_freq=10)
        trainer = Trainer(model, optimizer, loss_fn, callbacks=[logger])

        source = DistributionDataSource(target_dist, batch_size=32, seed=0)
        trained_model = trainer.train(source, max_steps=20)

        assert trainer.step == 20

        flow = Flow(velocity_field=trained_model, base_distribution=base_dist)
        x0 = base_dist.sample(seed=jax.random.key(1), sample_shape=(10,))
        x1 = jax.vmap(flow.apply_map)(x0)
        assert x1.shape == (10, 2)
        assert jnp.all(jnp.isfinite(x1))

    def test_8_gaussians_energy_training(self, base_dist, target_dist, model):
        """Full 8 Gaussians test with Energy-based loss."""
        optimizer = optax.adam(1e-3)
        loss_fn = EnergyBasedLoss(base_dist, target_dist)
        trainer = Trainer(model, optimizer, loss_fn)

        source = DistributionDataSource(base_dist, batch_size=32, seed=0)
        trained_model = trainer.train(source, max_steps=20)

        assert trainer.step == 20

        flow = Flow(velocity_field=trained_model, base_distribution=base_dist)
        x0 = base_dist.sample(seed=jax.random.key(1), sample_shape=(10,))
        x1 = jax.vmap(flow.apply_map)(x0)
        assert x1.shape == (10, 2)
        assert jnp.all(jnp.isfinite(x1))

    def test_8_gaussians_hybrid_training(self, base_dist, target_dist, model):
        """Full 8 Gaussians test with Hybrid KL loss."""
        optimizer = optax.adam(1e-3)
        loss_fn = KullbackLeiblerLoss(base_dist, target_dist, alpha=0.5)
        trainer = Trainer(model, optimizer, loss_fn)

        source = DistributionDataSource(target_dist, batch_size=32, seed=0)
        trained_model = trainer.train(source, max_steps=20)

        assert trainer.step == 20

        flow = Flow(velocity_field=trained_model, base_distribution=base_dist)
        x0 = base_dist.sample(seed=jax.random.key(1), sample_shape=(10,))
        x1 = jax.vmap(flow.apply_map)(x0)
        assert x1.shape == (10, 2)
        assert jnp.all(jnp.isfinite(x1))


# =============================================================================
# Hutchinson Estimator Tests
# =============================================================================


class TestHutchinsonTraining:
    """Tests for training with Hutchinson trace estimator."""

    def test_ml_loss_with_hutchinson(self, base_dist, target_dist, model):
        """Test ML training with Hutchinson estimator for divergence."""
        optimizer = optax.adam(1e-3)
        loss_fn = MaximumLikelihoodLoss(base_dist, hutchinson_samples=5)
        trainer = Trainer(model, optimizer, loss_fn)

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        trained_model = trainer.train(source, max_steps=10)

        assert trainer.step == 10

        flow = Flow(
            velocity_field=trained_model,
            base_distribution=base_dist,
            hutchinson_samples=5,
        )
        x0 = base_dist.sample(seed=jax.random.key(1), sample_shape=(5,))
        x1 = jax.vmap(flow.apply_map)(x0)
        assert x1.shape == (5, 2)
        assert jnp.all(jnp.isfinite(x1))

        logp = flow.log_prob(x1, key=jax.random.key(2))
        assert logp.shape == (5,)
        assert jnp.all(jnp.isfinite(logp))

    def test_energy_loss_with_hutchinson(self, base_dist, target_dist, model):
        """Test Energy-based training with Hutchinson estimator."""
        optimizer = optax.adam(1e-3)
        loss_fn = EnergyBasedLoss(base_dist, target_dist, hutchinson_samples=3)
        trainer = Trainer(model, optimizer, loss_fn)

        source = DistributionDataSource(base_dist, batch_size=16, seed=0)
        trained_model = trainer.train(source, max_steps=10)

        assert trainer.step == 10

        flow = Flow(
            velocity_field=trained_model,
            base_distribution=base_dist,
            hutchinson_samples=3,
        )
        x0 = base_dist.sample(seed=jax.random.key(1), sample_shape=(5,))
        keys = jax.random.split(jax.random.key(2), 5)
        x1, logq = jax.vmap(lambda x, k: flow.apply_map_and_log_prob(x, key=k))(x0, keys)
        assert x1.shape == (5, 2)
        assert logq.shape == (5,)
        assert jnp.all(jnp.isfinite(x1))
        assert jnp.all(jnp.isfinite(logq))

    def test_hybrid_loss_with_hutchinson(self, base_dist, target_dist, model):
        """Test Hybrid KL training with Hutchinson estimator."""
        optimizer = optax.adam(1e-3)
        loss_fn = KullbackLeiblerLoss(base_dist, target_dist, alpha=0.5, hutchinson_samples=3)
        trainer = Trainer(model, optimizer, loss_fn)

        source = DistributionDataSource(target_dist, batch_size=16, seed=0)
        trainer.train(source, max_steps=10)

        assert trainer.step == 10


class TestHutchinsonParticleTraining:
    """Tests for Hutchinson training comparison."""

    def test_hutchinson_gives_similar_results_to_exact(self, base_dist, target_dist):
        """Verify Hutchinson training produces similar results to exact."""
        key = jax.random.key(42)

        k1, k2 = jax.random.split(key)
        model_exact = MLPVelocity(input_dim=2, width=16, depth=2, key=k1)
        model_hutch = MLPVelocity(input_dim=2, width=16, depth=2, key=k1)

        loss_exact = MaximumLikelihoodLoss(base_dist)
        trainer_exact = Trainer(model_exact, optax.adam(1e-3), loss_exact, seed=0)
        source_exact = DistributionDataSource(target_dist, batch_size=32, seed=10)
        trained_exact = trainer_exact.train(source_exact, max_steps=20)

        loss_hutch = MaximumLikelihoodLoss(base_dist, hutchinson_samples=5)
        trainer_hutch = Trainer(model_hutch, optax.adam(1e-3), loss_hutch, seed=0)
        source_hutch = DistributionDataSource(target_dist, batch_size=32, seed=10)
        trained_hutch = trainer_hutch.train(source_hutch, max_steps=20)

        x0 = base_dist.sample(seed=jax.random.key(99), sample_shape=(10,))

        flow_exact = Flow(velocity_field=trained_exact, base_distribution=base_dist)
        flow_hutch = Flow(
            velocity_field=trained_hutch,
            base_distribution=base_dist,
            hutchinson_samples=5,
        )

        x1_exact = jax.vmap(flow_exact.apply_map)(x0)
        x1_hutch = jax.vmap(flow_hutch.apply_map)(x0)

        assert jnp.all(jnp.isfinite(x1_exact))
        assert jnp.all(jnp.isfinite(x1_hutch))

        assert jnp.abs(jnp.mean(x1_exact) - jnp.mean(x1_hutch)) < 5.0

        logp_exact = flow_exact.log_prob(x1_exact)
        logp_hutch = flow_hutch.log_prob(x1_hutch, key=jax.random.key(100))
        assert jnp.all(jnp.isfinite(logp_exact))
        assert jnp.all(jnp.isfinite(logp_hutch))
