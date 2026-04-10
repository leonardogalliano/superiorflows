"""Data sources for grain-based training pipelines."""

import grain
import jax

__all__ = ["DistributionDataSource", "CoupledDataSource"]


class DistributionDataSource(grain.sources.RandomAccessDataSource):
    """Wraps a distrax distribution as a grain-compatible data source.

    Each index deterministically maps to a batch via ``jax.random.fold_in``,
    so every epoch yields fresh samples while remaining fully reproducible
    from a checkpoint.

    Args:
        distribution: Any object with a ``.sample(seed=, sample_shape=)`` method
            (e.g. a ``distrax.Distribution``).
        batch_size: Number of samples per batch.
        seed: Base PRNG seed for deterministic sampling.
        length: Number of batches per epoch (only affects grain epoch boundaries).
    """

    def __init__(self, distribution, batch_size: int, *, seed: int = 0, length: int = 10_000):
        self._distribution = distribution
        self._batch_size = batch_size
        self._seed = seed
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        with jax.default_device(jax.devices("cpu")[0]):
            key = jax.random.fold_in(jax.random.key(self._seed), idx)
            return self._distribution.sample(seed=key, sample_shape=(self._batch_size,))


class CoupledDataSource(grain.sources.RandomAccessDataSource):
    """Composes two data sources into a source of paired samples.

    Each ``__getitem__`` call returns a tuple ``(source_0[idx], source_1[idx])``.
    The length is the minimum of the two sources. This is a generic coupling
    mechanism that works with any ``RandomAccessDataSource`` — distribution
    sources, trajectory sources, or anything else.

    Designed for stochastic interpolant training, where each step requires
    paired samples (e.g. from a base and target distribution).

    Args:
        source_0: First data source.
        source_1: Second data source.

    Example:
        >>> base_source = DistributionDataSource(base_dist, batch_size=32, seed=0)
        >>> target_source = DistributionDataSource(target_dist, batch_size=32, seed=1)
        >>> coupled = CoupledDataSource(base_source, target_source)
        >>> dataset = grain.MapDataset.source(coupled).repeat()
    """

    def __init__(self, source_0, source_1):
        self.source_0 = source_0
        self.source_1 = source_1

    def __len__(self):
        return min(len(self.source_0), len(self.source_1))

    def __getitem__(self, idx):
        return self.source_0[idx], self.source_1[idx]
