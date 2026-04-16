import logging

import diffrax as dfx
import grain
import jax
import jax.numpy as jnp
import pytest
from superiorflows import Flow

from particle_systems.particle_system import ParticleSystem, TrajectoryDataSource, UniformParticles
from particle_systems.velocities import ParticlesMLPVelocity

# Disable logging from grain/TrajectoryDataSource for cleaner benchmark output
logging.getLogger("particle_systems.particle_system").setLevel(logging.WARNING)

DATA_PATH = "particle_systems/data/ss14_T0.1.xyz"

# Need to load the data source once to get N, d, L, composition
try:
    source = TrajectoryDataSource(DATA_PATH)
    N = source.N
    d = source.d
    L = float(source.box_size[0])
    # Assume binary 50/50 mixture to match training script default
    composition = (0.5, 0.5)
    n_species = len(jnp.unique(jnp.asarray(source[0].species)))
except FileNotFoundError:
    pytest.skip(f"Dataset not found at {DATA_PATH}, skipping benchmarks.", allow_module_level=True)

# MLP parameters from training script defaults
WIDTH = 64
DEPTH = 3


@pytest.fixture(scope="module")
def base_distribution():
    return UniformParticles(N=N, d=d, L=L, composition=composition)


@pytest.fixture(scope="module")
def mlp_velocity():
    key = jax.random.key(0)
    return ParticlesMLPVelocity(N=N, d=d, n_species=n_species, width=WIDTH, depth=DEPTH, key=key)


def get_flow(base_dist, velocity, solver_name, solver_param):
    if solver_name == "euler":
        steps = solver_param
        flow_kwargs = dict(
            dynamic_mask=ParticleSystem.get_dynamic_mask(),
            solver=dfx.Euler(),
            augmented_solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            augmented_stepsize_controller=dfx.ConstantStepSize(),
            dt0=1.0 / steps,
        )
    else:  # tsit5
        tol = solver_param
        slv = dfx.Tsit5()
        flow_kwargs = dict(
            dynamic_mask=ParticleSystem.get_dynamic_mask(),
            solver=slv,
            augmented_solver=slv,
            stepsize_controller=dfx.PIDController(rtol=tol, atol=tol),
            augmented_stepsize_controller=dfx.PIDController(rtol=tol, atol=tol),
        )

    return Flow(velocity_field=velocity, base_distribution=base_dist, **flow_kwargs)


SOLVER_CONFIGS = [
    ("tsit5", 1e-3),
    ("tsit5", 1e-5),
    ("euler", 10),
    ("euler", 100),
]
BATCH_SIZES = [32, 128]


@pytest.mark.parametrize("solver_name, solver_param", SOLVER_CONFIGS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_benchmark_sample(benchmark, base_distribution, mlp_velocity, solver_name, solver_param, batch_size):
    flow = get_flow(base_distribution, mlp_velocity, solver_name, solver_param)

    # JIT compile the sample function
    @jax.jit
    def sample_fn(key):
        return flow.sample(seed=key, sample_shape=(batch_size,))

    key = jax.random.key(42)
    # Warmup and compile
    _ = sample_fn(key)
    # Block until compilation and execution finishes
    jax.block_until_ready(_)

    def run_sample():
        new_key = jax.random.key(123)
        res = sample_fn(new_key)
        jax.block_until_ready(res)
        return res

    benchmark(run_sample)


@pytest.mark.parametrize("solver_name, solver_param", SOLVER_CONFIGS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_benchmark_sample_and_log_prob(
    benchmark, base_distribution, mlp_velocity, solver_name, solver_param, batch_size
):
    flow = get_flow(base_distribution, mlp_velocity, solver_name, solver_param)

    # JIT compile
    @jax.jit
    def sample_lp_fn(key):
        return flow.sample_and_log_prob(seed=key, sample_shape=(batch_size,))

    key = jax.random.key(42)
    # Warmup and compile
    samples, lps = sample_lp_fn(key)
    jax.block_until_ready(samples)
    jax.block_until_ready(lps)

    def run_sample_lp():
        new_key = jax.random.key(456)
        s, lp = sample_lp_fn(new_key)
        jax.block_until_ready(s)
        jax.block_until_ready(lp)
        return s, lp

    benchmark(run_sample_lp)


@pytest.mark.parametrize("solver_name, solver_param", SOLVER_CONFIGS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_benchmark_log_prob(benchmark, base_distribution, mlp_velocity, solver_name, solver_param, batch_size):
    flow = get_flow(base_distribution, mlp_velocity, solver_name, solver_param)

    # Load a single batch of dataset, following training_particles.py logic
    # We use small a buffer to avoid hanging if there's issue with workers in pytest
    ds = source.to_dataset(batch_size=batch_size, shuffle=True, seed=0)
    read_options = grain.ReadOptions(num_threads=1, prefetch_buffer_size=1)
    batch_iterator = iter(ds.to_iter_dataset(read_options))
    batch_pt = next(batch_iterator)

    # Extract elements since grain returns Pytrees, and Flow expects the system
    eval_batch = jax.tree_util.tree_map(jnp.asarray, batch_pt)

    @jax.jit
    def log_prob_fn(b):
        return flow.log_prob(b)

    # Warmup and compile
    lps = log_prob_fn(eval_batch)
    jax.block_until_ready(lps)

    def run_log_prob():
        res = log_prob_fn(eval_batch)
        jax.block_until_ready(res)
        return res

    benchmark(run_log_prob)


@pytest.mark.parametrize("solver_name, solver_param", [("tsit5", 1e-3)])
@pytest.mark.parametrize("batch_size", [128])
def test_profile_sample_and_log_prob(base_distribution, mlp_velocity, solver_name, solver_param, batch_size, tmp_path):
    import jax.profiler

    flow = get_flow(base_distribution, mlp_velocity, solver_name, solver_param)

    # JIT compile
    @jax.jit
    def sample_lp_fn(key):
        return flow.sample_and_log_prob(seed=key, sample_shape=(batch_size,))

    key = jax.random.key(42)
    # Warmup and compile
    samples, lps = sample_lp_fn(key)
    jax.block_until_ready(samples)
    jax.block_until_ready(lps)

    log_dir = tmp_path / "jax_profile"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProfile directory: {log_dir}")
    jax.profiler.start_trace(str(log_dir))

    for i in range(5):
        new_key = jax.random.key(100 + i)
        s, lp = sample_lp_fn(new_key)
        jax.block_until_ready(s)
        jax.block_until_ready(lp)

    jax.profiler.stop_trace()
    print("Tracing finished.")
