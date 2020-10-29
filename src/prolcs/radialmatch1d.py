from typing import *
import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore


class RadialMatch1D():
    def __init__(self,
                 a: float=None,
                 b: float=None,
                 mu: float=None,
                 sigma_2: float=None,
                 ranges: Tuple[float, float] = (-np.inf, np.inf),
                 # TODO Have to use random_state b/c sklearn
                 rng: np.random.Generator = np.random.default_rng()):
        """
        Exactly one of ``a`` and ``mu`` has to be given (the other one can be
        inferred); the same goes for ``b`` and ``sigma_2``.

        :param a: Evolving parameter from which the position of the Gaussian is
            inferred.
        :param b: Evolving parameter from which the standard deviation of the
            Gaussian is inferred.
        :param mu: Position of the Gaussian.
        :param sigma_2: Standard deviation.
        :param ranges: The value ranges of the problem considered. If ``None``,
            use ``[-inf, inf]`` for each dimension.
        """
        self.ranges = ranges

        if a is not None and mu is None:
            self.a = a
        elif a is None and mu is not None:
            assert np.isfinite(ranges).all(), "If specifying mu, ranges need to be finite"
            l, u = self.ranges
            self.a = 100 * (mu - l) / (u - l)
        else:
            raise ValueError("Exactly one of a and mu has to be given")

        if b is not None and sigma_2 is None:
            self.b = b
        elif b is None and sigma_2 is not None:
            self.b = -10 * np.log10(sigma_2)
        else:
            raise ValueError("Exactly one of b and sigma_2 has to be given")

        print(self.mu())
        print(self.sigma_2())

        self.rng = rng

    def mu(self):
        l, u = self.ranges
        return l + (u - l) * self.a / 100

    def sigma_2(self):
        return 10**(-self.b / 10)

    def __repr__(self):
        return f"RadialMatch1D({self.mu}, {self.sigma}, {self.ranges})"

    @classmethod
    def random(cls,
               ranges: Tuple[float, float],
               rng: np.random.Generator = np.random.default_rng()):
        """
        [PDF p. 256]

        :param ranges: The input values' range
        """
        return RadialMatch1D(a=rng.uniform(0, 100),
                             b=rng.uniform(0, 50),
                             ranges=ranges,
                             rng=rng)

    def mutate(self):
        """
        [PDF p. 256]
        """
        self.a = np.clip(self.rng.normal(loc=self.a, scale=10), 0, 100)
        self.b = np.clip(self.rng.normal(loc=self.b, scale=5), 0, 50)
        return self

    def match(self, X: np.ndarray):
        """
        We vectorize the following (i.e. feed the whole input through at once)::

            for n in range(len(X)):
                M[n] = np.exp(-0.5 * (x - mu) @ lambd_2 @ (x - mu))

        :param X: input matrix ``(N × D_X)`` with ``D_X == 1``
        :returns: matching vector ``(N)`` of this matching function (i.e. of
            this classifier)
        """
        # We have to clip this so we don't return 0 here (0 should never be
        # returned because every match function matches everywhere at least a
        # little bit). Also, we clip from above such that this never returns a
        # value larger than 1 (it's a probability, after all).
        m_min = np.log(np.finfo(None).tiny)
        # TODO The maximum negative number might be different than simply the
        # negation of the minimum positive number.
        m_max = -np.finfo(None).tiny
        m = np.clip(-1 / (2 * self.sigma_2()) * (X - self.mu())**2, m_min,
                    m_max)
        return np.exp(m)

    def plot(self, ax, **kwargs):
        l = self.ranges[0]
        h = self.ranges[1]
        X = np.arange(l, h, 0.01)[:, np.newaxis]
        M = self.match(X)
        ax.plot(X, M, **kwargs)
        ax.axvline(self.mu(), color=kwargs["color"])
