import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


class SoftInterval1D():
    def __init__(self, l: float, u: float, has_bias=True):
        """
        ``self.match`` is a soft interval–based matching function as defined in
        Drugowitsch's book [PDF p. 260].

        “When specifying the interval for classifier k by its lower bound l_k
        and upper bound u_k, we want exactly one standard deviation of the
        Gaussian to lie inside this interval, and additionally require 95% of
        the area underneath the matching function to be inside this interval.”

        Data is assumed to lie within ``[-1, 1]``.

        :param a: Evolving parameter from which the position of the Gaussian is
            inferred (``0 <= a <= 100``). [PDF p. 256]

            Exactly one of ``a`` and ``mu`` has to be given (the other one can
            be inferred); the same goes for ``b`` and ``sigma_2``.
        :param b: Evolving parameter from which the standard deviation of the
            Gaussian is inferred (``0 <= b <= 50``). See ``a``.
        :param mu: Position of the Gaussian. See ``a``.
        :param sigma_2: Standard deviation. See ``a``.
        :param has_bias: Whether to expect 2D data where we always match the
            first dimension (e.g. because it is all ones as a bias to implicitly
            fit the intercept).
        """
        self.has_bias = has_bias

        # Data is assumed to lie within [-1, 1]
        self._l, self._u = -1, 1

        # Unordered bound representation, we swap if necessary. [PDF p. 261]
        self.l, self.u = tuple(sorted([l, u]))

    def __repr__(self):
        return (f"SoftInterval1D(l={self.l},u={self.u},"
                f"has_bias={self.has_bias})")

    def sigma2(self):
        return (0.0662 * (self.u - self.l))**2

    @classmethod
    def random(cls, random_state: np.random.RandomState):
        """
        [PDF p. 260]
        """
        random_state = check_random_state(random_state)
        bounds = random_state.uniform(-1, 1, size=2)
        return SoftInterval1D(*bounds)

    def mutate(self, random_state: np.random.RandomState):
        """
        Note: Unordered bound representation, we swap if necessary.

        [PDF p. 261]
        """
        self.l, self.u = tuple(
            sorted(
                np.clip(
                    random_state.normal(loc=(self.l, self.u),
                                        scale=(self._u - self._l) / 10),
                    self._l, self._u)))
        return self

    def match(self, X: np.ndarray):
        """
        Compute matching vector for given input. Depending on whether the input
        is expected to have a bias column (see attribute ``self.has_bias``),
        remove that beforehand.

        Parameters
        ----------
        X : array of shape ``(N, 1)`` or ``(N, 2)`` if ``self.has_bias``
            Input matrix.

        Returns
        -------
        array of shape ``(N)``
            Matching vector of this matching function for the given input.
        """

        if self.has_bias:
            assert X.shape[
                1] == 2, f"X should have 2 columns but has {X.shape[1]}"
            X = X.T[1:].T

        return self._match_wo_bias(X)

    def _match_wo_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Compute matching vector for given input assuming that the input doesn't
        have a bias column.

        Parameters
        ----------
        X : input matrix of shape ``(N, D_X)`` with ``D_X == 1``

        Returns
        -------
        array of shape ``(N)``
            Matching vector of this matching function for the given input.
        """
        sigma2 = self.sigma2()

        # The interval may be trivial.
        if sigma2 == 0:
            return np.where(X == self.u, 1, np.finfo(None).tiny)
        else:
            # We have to clip this so we don't return 0 here (0 should never be
            # returned because every match function matches everywhere at least a
            # little bit).
            conds = [
                X < self.l,
                X > self.u,
            ]
            cases = [
                np.exp(-1 / (2 * sigma2) * (X - self.l)**2),
                np.exp(-1 / (2 * sigma2) * (X - self.u)**2),
            ]
            default = 1
            m = np.select(conds, cases, default=default)
            m_min = np.finfo(None).tiny
            m_max = 1
            return np.clip(m, m_min, m_max)

    def plot(self, l, u, ax, **kwargs):
        X = np.arange(l, u, 0.01)[:, np.newaxis]
        M = self._match_wo_bias(X)
        ax.plot(X, M, **kwargs)
        ax.axvline(self.mu(), color=kwargs["color"])
