import logging
from itertools import permutations, product
from pathlib import Path

import distrax as dsx
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

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


def log_map(x, y, L):
    """Shortest displacement vector from x to y under PBC."""
    diff = y - x
    return diff - jnp.round(diff / L) * L


def exp_map(x, v, L):
    """Apply displacement v to position x, folding back into the box."""
    return jnp.remainder(x + v, L)


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

    def to_dataset(self, batch_size, shuffle=True, repeat=True, seed=0):
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
        if repeat:
            ds = ds.repeat()
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


def particle_geodesic_interpolant(t, x0, x1, L):
    """Geodesic interpolation from x0 to x1 at time t in [0, 1]."""
    v = log_map(x0.positions, x1.positions, L)
    xt = exp_map(x0.positions, t * v, L)
    return type(x0)(positions=xt, species=x0.species, box=x0.box)


class CoupleBaseSamples:
    """Stateless callable to generate the base distribution and couple it."""

    def __init__(self, base_dist, seed=0):
        self.base_dist = base_dist
        self.seed = seed

    def __call__(self, index, x1):
        batch_size = x1.positions.shape[0]

        with jax.default_device(jax.devices("cpu")[0]):
            key = jax.random.fold_in(jax.random.key(self.seed), index)
            x0 = self.base_dist.sample(seed=key, sample_shape=(batch_size,))

        return (x0, x1)


def generate_hyperoctahedral_group(d):
    """All elements of the hyperoctahedral group B_d as signed-permutation matrices.

    Returns:
        np.ndarray of shape ``(2^d * d!, d, d)``.
    """
    matrices = []
    for perm in permutations(range(d)):
        P = np.eye(d)[list(perm)]
        for signs in product([1, -1], repeat=d):
            matrices.append(np.diag(signs) @ P)
    return np.array(matrices)


def apply_box_symmetry(positions, g, L):
    """Apply a hyperoctahedral group element on the flat torus ``[0, L)^d``.

    Centres at ``L/2``, applies the linear map *g*, re-centres and wraps.
    """
    center = L / 2.0
    return ((positions - center) @ g.T + center) % L


def _solve_ot_single(pos_0, pos_1, species_0, species_1, box):
    """Solve species-wise OT for a single sample pair.

    Returns ``(aligned_positions, total_cost)``.
    """
    N, d = pos_0.shape
    aligned = np.empty_like(pos_0)
    total_cost = 0.0

    for s in np.unique(species_1):
        idx_0 = np.where(species_0 == s)[0]
        idx_1 = np.where(species_1 == s)[0]

        if len(idx_0) == 0 or len(idx_1) == 0:
            continue
        assert len(idx_0) == len(idx_1), f"Species counts mismatch for species {s}"

        p0_s = pos_0[idx_0]
        p1_s = pos_1[idx_1]

        delta = p0_s[:, None, :] - p1_s[None, :, :]
        delta -= box * np.round(delta / box)
        cost_matrix = np.sum(delta**2, axis=-1)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        aligned[idx_1[col_ind]] = p0_s[row_ind]
        total_cost += cost_matrix[row_ind, col_ind].sum()

    return aligned, total_cost


class EquivariantOptimalTransport:
    """Equivariant Optimal Transport for particle systems.

    Minimises the total squared geodesic distance between coupled samples
    ``(x0, x1)`` by exploiting symmetries of the base distribution:

    1. **Permutation within species** (always active): particles of the same
       species are interchangeable.
    2. **Hyperoctahedral group** (``use_box_symmetry=True``): the cubic
       periodic box is invariant under axis reflections and permutations
       (8 elements in 2D, 48 in 3D).

    Args:
        use_box_symmetry: Also optimise over the hyperoctahedral group B_d.
    """

    def __init__(self, use_box_symmetry=False):
        self.use_box_symmetry = use_box_symmetry

    def __call__(self, batch):
        x0, x1 = batch

        pos_0 = np.asarray(x0.positions)
        pos_1 = np.asarray(x1.positions)
        species_0 = np.asarray(x0.species)
        species_1 = np.asarray(x1.species)
        box = np.asarray(x1.box)

        B, N, d = pos_0.shape

        if self.use_box_symmetry:
            if not np.allclose(box[:, 0:1], box):
                raise ValueError("Box symmetry requires a cubic box, but box sides differ: " f"{box[0]}")
            group = generate_hyperoctahedral_group(d)
        else:
            group = np.eye(d)[np.newaxis]

        aligned_positions = np.empty_like(pos_0)

        for i in range(B):
            best_cost = np.inf
            best_aligned = None

            for g in group:
                p0_g = apply_box_symmetry(pos_0[i], g, box[i])
                aligned_i, cost_i = _solve_ot_single(
                    p0_g,
                    pos_1[i],
                    species_0[i],
                    species_1[i],
                    box[i],
                )
                if cost_i < best_cost:
                    best_cost = cost_i
                    best_aligned = aligned_i

            aligned_positions[i] = best_aligned

        x0_aligned = type(x0)(
            positions=aligned_positions,
            species=species_1,
            box=box,
        )
        return (x0_aligned, x1)


def batch_to_trajectory(particle_systems: ParticleSystem):
    """Convert a batched ParticleSystem into an in-memory atooms trajectory.

    Args:
        particle_systems: A batched ParticleSystem object (B, N, d).

    Returns:
        atooms.trajectory.TrajectoryRam holding an iterable list of Systems natively mapped.
    """
    from atooms.system import System
    from atooms.system.cell import Cell
    from atooms.system.particle import Particle
    from atooms.trajectory import TrajectoryRam

    pos = np.asarray(particle_systems.positions)
    species = np.asarray(particle_systems.species)
    boxes = np.asarray(particle_systems.box)

    if pos.ndim != 3:
        raise ValueError(f"Expected batched ParticleSystem (B, N, d), got shape {pos.shape}")

    B, N, d = pos.shape
    trj = TrajectoryRam()

    for i in range(B):
        sys = System()
        b = boxes if boxes.ndim == 1 else boxes[i]
        sys.cell = Cell(side=b)
        sys.particle = [Particle(position=p, species=int(s)) for p, s in zip(pos[i], species[i])]
        trj.append(sys)

    return trj
