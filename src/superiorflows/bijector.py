"""Abstract bijector interface for invertible transformations.

Provides a lightweight base class for bijectors (invertible, differentiable
maps) that operate on arbitrary pytree states. Concrete subclasses must
implement `forward_and_log_det` and `inverse_and_log_det`; all other
methods have default implementations that delegate to these two.

All methods accept ``**kwargs`` so that concrete implementations may receive
additional arguments without modifying the abstract interface — for example,
PRNG keys for stochastic log-det estimators.
"""

import equinox as eqx

__all__ = ["AbstractBijector", "DistreqxBijectorWrapper"]


class AbstractBijector(eqx.Module):
    """Abstract invertible transformation on arbitrary pytree states.

    Subclasses must implement :meth:`forward_and_log_det` and
    :meth:`inverse_and_log_det`.  The remaining four methods are provided
    automatically but may be overridden for efficiency.

    Sign convention: ``forward_and_log_det`` returns
    ``(y, log|det J_f(x)|)`` and ``inverse_and_log_det`` returns
    ``(x, log|det J_{f^{-1}}(y)|)``.  These satisfy
    ``log|det J_f(x)| = -log|det J_{f^{-1}}(f(x))|``.
    """

    def forward(self, x, **kwargs):
        """Compute y = f(x)."""
        y, _ = self.forward_and_log_det(x, **kwargs)
        return y

    def inverse(self, y, **kwargs):
        """Compute x = f⁻¹(y)."""
        x, _ = self.inverse_and_log_det(y, **kwargs)
        return x

    def forward_and_log_det(self, x, **kwargs):
        """Compute (y, log|det J_f(x)|)."""
        raise NotImplementedError

    def inverse_and_log_det(self, y, **kwargs):
        """Compute (x, log|det J_{f⁻¹}(y)|)."""
        raise NotImplementedError

    def forward_log_det_jacobian(self, x, **kwargs):
        """Compute log|det J_f(x)| without returning y."""
        _, logdet = self.forward_and_log_det(x, **kwargs)
        return logdet

    def inverse_log_det_jacobian(self, y, **kwargs):
        """Compute log|det J_{f⁻¹}(y)| without returning x."""
        _, logdet = self.inverse_and_log_det(y, **kwargs)
        return logdet


class DistreqxBijectorWrapper(AbstractBijector):
    """Adapt a distreqx bijector to the superiorflows interface.

    Wraps any object that exposes the distreqx bijector protocol
    (``forward``, ``inverse``, ``forward_and_log_det``,
    ``inverse_and_log_det``) so that it can be used wherever a
    :class:`AbstractBijector` is expected.  Extra ``**kwargs`` are
    silently ignored since distreqx bijectors do not accept them.
    """

    _bijector: eqx.Module

    def forward(self, x, **kwargs):
        return self._bijector.forward(x)

    def inverse(self, y, **kwargs):
        return self._bijector.inverse(y)

    def forward_and_log_det(self, x, **kwargs):
        return self._bijector.forward_and_log_det(x)

    def inverse_and_log_det(self, y, **kwargs):
        return self._bijector.inverse_and_log_det(y)
