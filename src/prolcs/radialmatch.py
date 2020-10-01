from typing import *
import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore


class RadialMatch():
    def __init__(self, ranges: np.ndarray):
        """
        Based on [PDF p. 256] but slightly different:

        * We allow multi-dimensional inputs (he only does ``D_X = 1``).
        * We do not introduce helper parameters (for now).
        * We only roughly try to mimic the ranges of mutation used by him.
        * We clip both sigma and mu values to the respective dimension's range.

        This is basically what (Butz, 2005, “Kernel-based, ellipsoidal
        conditions in the real-valued XCS classifier system”) but with soft
        boundaries (i.e. every classifier matches everywhere, even if only
        slightly), that is, without a threshold parameter. We evolve
        ``sigma**2`` directly.

        :param ranges: The value ranges of the problem considered
        """
        D_X, _ = ranges.shape
        assert _ == 2
        self._ranges = ranges

        self.mu = np.random.random(size=D_X)
        # Drugowitsch restricts ``sigma**2`` values to ``[10**(-50), 1)``, I
        # don't think we need that?
        self.sigma_2 = np.random.random(size=(D_X, D_X))

    def mutate(self):
        self.mu = np.clip(
            np.random.normal(
                loc=self.mu,
                # For each dimension, we set the normal's scale to ``0.1 * (u -
                # l)``.
                # TODO This was chosen to be similar to [PDF p. 228] but isn't
                # probably
                scale=0.1 * np.sum(self._ranges * np.array([-1, 1]), 1),
                size=self.mu.shape),
            # clip to each dimension's range
            self._ranges[:, [0]],
            self._ranges[:, [1]])
        self.sigma_2 = np.random.normal(
            loc=self.sigma_2,
            # For each dimension, we set the normal's scale to ``0.05 * (u -
            # l)``.
            # TODO This was chosen relatively arbitrary (but motivated by [PDF
            # p. 228])
            scale=0.05 * np.sum(self._ranges * np.array([-1, 1]), 1),
            size=self.sigma_2.shape)

    def match(self, X: np.ndarray):
        """
        We vectorize the following (i.e. feed the whole input through at once)::

            for x in X:
                np.exp(-0.5 * (x - self.mu)[:, np.newaxis] @ self.sigma_2 @ (x - self.mu))``

        :param X: input matrix (N × D_X)
        :returns: matching vector (N) of this matching function (i.e. of this
            classifier)
        """

        # TODO Acc. to Wikipedia, sigma has to be positive definite?
        # https://en.wikipedia.org/wiki/Gaussian_function#Multi-dimensional_Gaussian_function
        #
        # We vectorize the following:
        # np.exp(-0.5 * (x - self.mu)[:, np.newaxis] @ self.sigma_2 @ (x - self.mu))
        mu_ = np.broadcast_to(self.mu, X.shape)
        delta = X - mu_
        return np.exp(-0.5 * np.sum(delta.T * (self.sigma_2 @ delta.T), 0))
