from typing import *
import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore


class RadialMatch():
    def __init__(self,
                 mu: np.ndarray,
                 lambd_2: np.ndarray,
                 ranges: np.ndarray = None):
        """
        Note: The covariance matrix has to be positive definite (cf. e.g.
        `Wikipedia
        <https://en.wikipedia.org/wiki/Gaussian_function#Multi-dimensional_Gaussian_function>`_),
        thus we simply require its inverse right away (also, this way we don't
        have to invert it which is costly and the model structure optimizer
        can very well work on the inverse directly anyway).

        :param mu: Position of the Gaussian
        :param lambd_2: Squared precision matrix (squared inverse covariance
            matrix)
        :param ranges: The value ranges of the problem considered. If ``None``,
            use ``[-inf, inf]`` for each dimension.
        """
        assert mu.shape[0] == lambd_2.shape[0]
        assert mu.shape[0] == lambd_2.shape[1]
        self.mu = mu
        self.lambd_2 = lambd_2
        if ranges is not None:
            assert ranges.shape == (mu.shape[0], 2)
            self.ranges = ranges
        else:
            self.ranges = np.repeat([-np.inf, np.inf], len(mu)).reshape(
                (mu.shape[0], 2))

    @classmethod
    def random(cls, ranges: np.ndarray):
        """
        Based on [PDF p. 256] but slightly different:

        * We allow multi-dimensional inputs (he only does ``D_X = 1``).
        * We do not introduce helper parameters (for now).
        * We only roughly try to mimic the ranges of mutation used by him.
        * We clip both lambd_2 and mu values to the respective dimension's range.

        This is basically what (Butz, 2005, “Kernel-based, ellipsoidal
        conditions in the real-valued XCS classifier system”) but with soft
        boundaries (i.e. every classifier matches everywhere, even if only
        slightly), that is, without a threshold parameter. We evolve
        ``lambd_2**2`` directly.

        :param ranges: The value ranges of the problem considered
        """
        D_X, _ = ranges.shape
        assert _ == 2
        return RadialMatch(
            ranges=ranges,
            mu=np.random.random(size=D_X),
            # Drugowitsch restricts ``lambd_2**2`` values to ``[10**(-50), 1)``, I
            # don't think we need that?
            lambd_2=np.random.random(size=(D_X, D_X)))

    def mutate(self):
        self.mu = np.clip(
            np.random.normal(
                loc=self.mu,
                # For each dimension, we set the normal's scale to ``0.1 * (u -
                # l)``.
                # TODO This was chosen to be similar to [PDF p. 228] but isn't
                # probably
                scale=0.1 * np.sum(self.ranges * np.array([-1, 1]), 1)),
            # clip to each dimension's range
            self.ranges[:, [0]].reshape((-1)),
            self.ranges[:, [1]].reshape((-1)))
        self.lambd_2 = np.random.normal(
            loc=self.lambd_2,
            # For each dimension, we set the normal's scale to ``0.05 * (u -
            # l)``.
            # TODO This was chosen relatively arbitrary (but motivated by [PDF
            # p. 228])
            scale=0.05 * np.sum(self.ranges * np.array([-1, 1]), 1),
            size=self.lambd_2.shape)
        return self

    def match(self, X: np.ndarray):
        """
        We vectorize the following (i.e. feed the whole input through at once)::

            for n in range(len(X)):
                M[n] = np.exp(-0.5 * (x - mu) @ lambd_2 @ (x - mu))

        :param X: input matrix (N × D_X)
        :returns: matching vector (N) of this matching function (i.e. of this
            classifier)
        """
        mu_ = np.broadcast_to(self.mu, X.shape)
        delta = X - mu_
        # We have to clip this so we don't return 0 here (0 should never be
        # returned because every match function matches everywhere at least a
        # little bit).
        x = np.clip(-0.5 * np.sum(delta.T * (self.lambd_2 @ delta.T), 0),
                    np.log(np.finfo(None).tiny), np.log(np.finfo(None).max))
        return np.exp(x)[:, np.newaxis]
