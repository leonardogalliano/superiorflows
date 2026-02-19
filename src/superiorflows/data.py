"""Data sources for grain-based training pipelines."""

import grain
import jax

__all__ = ["DistributionDataSource"]


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
