import logging
from pathlib import Path

import distrax as dsx
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class ParticleSystem(eqx.Module):
    positions: jnp.ndarray
    species: jnp.ndarray
    box: jnp.ndarray

    @property
    def d(self):
        return self.positions.shape[-1]

    @property
    def N(self):
        return self.positions.shape[-2]

    @classmethod
    def get_dynamic_mask(cls):
        return cls(positions=True, species=False, box=False)


class UniformParticles(eqx.Module, dsx.Distribution):
    N: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    L: float = eqx.field(static=True)
    ref_species: jnp.ndarray

    def __init__(self, N, d, L, composition):
        self.L = float(L)
        self.N = N
        self.d = d
        self.ref_species = jnp.concatenate([jnp.full(int(x * N), i, dtype=int) for i, x in enumerate(composition)])
        assert sum(composition) == 1.0
        assert jnp.all(self.ref_species.shape[0] == N)

    @property
    def event_shape(self):
        return ParticleSystem(
            positions=(self.N, self.d),
            species=(self.N,),
            box=(self.d,),
        )

    def _sample_n(self, key, n):
        k_pos, k_spec = jax.random.split(key)

        # 1. Sample Positions: Uniform(0, L)
        # Shape: (n, N, d)
        pos = jax.random.uniform(k_pos, shape=(n, self.N, self.d), minval=0.0, maxval=self.L)

        # 2. Sample Species: Random permutation of ref_species
        def _permute(k):
            return jax.random.permutation(k, self.ref_species)

        keys_perm = jax.random.split(k_spec, n)
        species = jax.vmap(_permute)(keys_perm)

        # 3. Box: Broadcast scalar L
        batched_box = jnp.full((n, self.d), self.L)

        return ParticleSystem(positions=pos, species=species, box=batched_box)

    def log_prob(self, value: ParticleSystem):
        # We assume configurations outside the box are valid
        # as they represent configurations
        # that would be projected back into the box.

        # 1. Base Log Prob: -N * log(Volume)
        base_log_prob = -self.N * (self.d * jnp.log(self.L))

        # 2. Correct Species Composition
        sorted_val_species = jnp.sort(value.species, axis=-1)
        sorted_ref_species = jnp.sort(self.ref_species, axis=-1)
        valid_composition = jnp.all(sorted_val_species == sorted_ref_species, axis=-1)

        # 3. Return
        return jnp.where(valid_composition, base_log_prob, -jnp.inf)


class BoltzmannDistribution(eqx.Module, dsx.Distribution):
    """Boltzmann distribution at temperature T for a particle system.

    This is the *target* distribution for CNF training.  It provides the
    unnormalised log-probability ``log p(x) = -U(x) / T`` via the ``log_prob``
    method.  **It cannot be sampled**.

    The energy function is built once at init time from a berni model dict
    using ``potentials.build_energy_fn``, so there is no per-call overhead.

    Args:
        N: Number of particles.
        d: Spatial dimensionality.
        L: Box side length (cubic box assumed).
        temperature: Temperature T.
        model: A berni model dictionary.
        composition: Tuple of species fractions (e.g. ``(0.5, 0.5)``).
    """

    N: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    L: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    model: dict = eqx.field(static=True)
    _energy_fn: object = eqx.field(static=True)

    def __init__(self, N, d, L, temperature, model, composition):
        from particle_systems.potentials import build_energy_fn

        self.N = N
        self.d = d
        self.L = float(L)
        self.temperature = float(temperature)
        self.model = model
        n_species = len(composition)
        box = jnp.ones(d) * self.L
        self._energy_fn = build_energy_fn(model, box, n_species)

    @property
    def event_shape(self):
        return ParticleSystem(
            positions=(self.N, self.d),
            species=(self.N,),
            box=(self.d,),
        )

    def _sample_n(self, key, n):
        raise NotImplementedError("Cannot sample from Boltzmann distribution.")

    def log_prob(self, value: ParticleSystem):
        """Unnormalised log-probability: -U(x) / T.

        Note:
            This is **not** normalised.

        Args:
            value: A ``ParticleSystem``, single or batched.

        Returns:
            Scalar or array of unnormalised log-probabilities.
        """
        positions = jnp.asarray(value.positions)
        species = jnp.asarray(value.species)

        def _single_log_prob(pos, sp):
            return -self._energy_fn(pos, sp) / self.temperature

        if positions.ndim == 2:  # single system: (N, d)
            return _single_log_prob(positions, species)
        # batched: (M, N, d)
        return jax.vmap(_single_log_prob)(positions, species)


class TrajectoryDataSource(grain.sources.RandomAccessDataSource):
    """Grain data source that serves ParticleSystem samples from trajectory files.

    Pre-loads all trajectory frames into contiguous numpy arrays at construction
    time, so ``__getitem__`` is a zero-overhead slice — no I/O during training.

    Supports two modes:
        - **Single file**: pass a path to an XYZ trajectory file.
        - **Directory scan**: pass a directory path and all files matching
          ``filename`` will be discovered recursively, sorted for
          reproducibility, and concatenated.

    Species strings from atooms (e.g. ``'1'``, ``'A'``) are converted to
    sorted integer labels (``0, 1, 2, ...``).

    Args:
        path: Path to a trajectory file or directory containing trajectories.
        filename: Filename pattern to search for when ``path`` is a directory.
        dtype: NumPy dtype for positions and box arrays.
    Example:
        >>> source = TrajectoryDataSource("/path/to/trajectories/")
        >>> dataset = source.to_dataset(batch_size=64, shuffle=True)
        >>> for batch in dataset:
        ...     print(batch.positions.shape)  # (64, N, d)
    """

    def __init__(self, path, filename="trajectory.xyz", dtype=np.float32):
        from atooms.trajectory import TrajectoryXYZ

        path = Path(path)

        # Discover trajectory files
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(path.rglob(filename))
            if not files:
                raise FileNotFoundError(f"No '{filename}' files found under {path}")
            logger.info(f"Found {len(files)} trajectory files under {path}")
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Load all frames into numpy arrays
        all_positions = []
        all_species = []
        all_box = []
        species_map = None  # str -> int mapping, built from first frame
        self._metadata = None

        for fpath in files:
            with TrajectoryXYZ(str(fpath)) as trj:
                if self._metadata is None:
                    self._metadata = dict(trj.metadata)

                for frame in trj:
                    pos = frame.view("position").astype(dtype)
                    spe_str = frame.view("species")
                    box = np.array(frame.cell.side, dtype=dtype)

                    # Build species mapping from first frame
                    if species_map is None:
                        unique_species = sorted(set(spe_str))
                        species_map = {s: i for i, s in enumerate(unique_species)}
                        logger.info(f"Species mapping: {species_map} " f"(N={pos.shape[0]}, d={pos.shape[1]})")

                    # Convert species strings to integer labels
                    spe_int = np.array([species_map[s] for s in spe_str], dtype=np.int32)

                    all_positions.append(pos)
                    all_species.append(spe_int)
                    all_box.append(box)

        # Stack into contiguous arrays
        self._positions = np.stack(all_positions)  # (n_frames, N, d)
        self._species = np.stack(all_species)  # (n_frames, N)
        self._box = np.stack(all_box)  # (n_frames, d)
        self._n_frames = len(all_positions)
        self._species_map = species_map
        self._n_files = len(files)

        logger.info(
            f"Loaded {self._n_frames} frames from {self._n_files} file(s): "
            f"positions {self._positions.shape}, "
            f"species {self._species.shape}, "
            f"box {self._box.shape}"
        )

    def __len__(self):
        return self._n_frames

    def __getitem__(self, idx):
        """Return a single ParticleSystem sample with numpy arrays."""
        return ParticleSystem(
            positions=self._positions[idx],
            species=self._species[idx],
            box=self._box[idx],
        )

    def __repr__(self):
        return (
            f"TrajectoryDataSource("
            f"n_frames={self._n_frames}, "
            f"n_files={self._n_files}, "
            f"shape=({self._positions.shape[1]}, {self._positions.shape[2]}), "
            f"species_map={self._species_map})"
        )

    def to_dataset(self, batch_size, shuffle=True, seed=0):
        """Build a grain pipeline: source → shuffle → batch → repeat.

        Returns a ``grain.MapDataset`` ready to be consumed via
        ``dataset.to_iter_dataset(read_options)``, which is what the
        ``Trainer`` expects.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle samples before batching.
            seed: Random seed for shuffling (ignored if shuffle=False).

        Returns:
            A ``grain.MapDataset`` yielding batched ``ParticleSystem`` pytrees.
        """
        ds = grain.MapDataset.source(self)
        if shuffle:
            ds = ds.shuffle(seed=seed)
        ds = ds.batch(batch_size=batch_size, drop_remainder=True)
        return ds

    @property
    def metadata(self):
        """Original atooms metadata dict from the first trajectory file."""
        return self._metadata

    @property
    def species_map(self):
        """Mapping from original species strings to integer labels."""
        return self._species_map

    @property
    def N(self):
        """Number of particles per frame."""
        return self._positions.shape[1]

    @property
    def d(self):
        """Spatial dimensionality."""
        return self._positions.shape[2]

    @property
    def box_size(self):
        """Box side lengths from the first frame."""
        return self._box[0]
