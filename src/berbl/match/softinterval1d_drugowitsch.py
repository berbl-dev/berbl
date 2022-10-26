import numpy as np  # type: ignore


class SoftInterval1D():
    def __init__(self, l: float, u: float, has_bias=True, input_bounds=None):
        """
        ``self.match`` is a soft interval–based matching function as defined in
        Drugowitsch's book [PDF p. 260].

        “When specifying the interval for [rule] k by its lower bound l_k
        and upper bound u_k, we want exactly one standard deviation of the
        Gaussian to lie inside this interval, and additionally require 95% of
        the area underneath the matching function to be inside this interval.”

        Parameters
        ==========
        input_bounds : pair of two floats or None
            If ``None`` (the default), input is assumed to be standardized.
            Otherwise, input is assumed to lie within the interval described by
            the two floats. Note that inputs *should* be standardized for
            everything else to work properly.
        """
        self.has_bias = has_bias

        if input_bounds is not None:
            self._l, self._u = input_bounds
            print("Warning: Changed matching function input bounds "
                  f"to {input_bounds}")
        else:
            # Since inputs are standardized, there a low probability of an input
            # lying outside [-5, 5]. If inputs were distributed normally, that
            # probability would be 1e-6. If inputs were distributed uniformly,
            # that probability would be 0.
            self._l, self._u = -5, 5

        # Unordered bound representation, we swap if necessary. [PDF p. 261]
        self.l, self.u = tuple(sorted([l, u]))

    def __repr__(self):
        return (f"SoftInterval1D(l={self.l},u={self.u},"
                f"has_bias={self.has_bias})")

    def sigma2(self):
        return (0.0662 * (self.u - self.l))**2

    @classmethod
    def random(cls, random_state: np.random.RandomState, input_bounds=None):
        """
        [PDF p. 260]

        Parameters
        ==========
        input_bounds : pair of two floats or None
            See constructor documentation.
        """
        if input_bounds is not None:
            _l, _u = input_bounds
        else:
            # Since inputs are standardized, there a low probability of an input
            # lying outside [-5, 5]. If inputs were distributed normally, that
            # probability would be 1e-6. If inputs were distributed uniformly,
            # that probability would be 0.
            _l, _u = -5, 5

        l, u = tuple(random_state.uniform(_l, _u, size=2))
        return SoftInterval1D(l, u, input_bounds=input_bounds)

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
        X : input matrix of shape ``(N, DX)`` with ``DX == 1``

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
