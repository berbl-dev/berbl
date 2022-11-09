import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


class RadialMatch1D():

    def __init__(
            self,
            *,
            a: float = None,
            b: float = None,
            mu: float = None,
            sigma_2: float = None,
            # TODO Detect input_bounds automatedly
            input_bounds=None):
        """
        [`self.match`][berbl.match.radial1d_drugowitsch.RadialMatch1D.match]
        is a radial basis function–based matching function as
        defined in [Drugowitsch's book](/) [PDF p. 256].

        Parameters
        ----------
        a : float
            Evolving parameter from which the position of the Gaussian is
            inferred (`0 <= a <= 100`). [PDF p. 256]

            Exactly one of `a` and `mu` has to be given (the other one can
            be inferred); the same goes for `b` and `sigma_2`.
        b : float
            Evolving parameter from which the standard deviation of the
            Gaussian is inferred (`0 <= b <= 50`). See `a`.
        mu : float
            Position of the Gaussian. See `a`.
        sigma_2 : float
            Standard deviation. See `a`.
        input_bounds : pair of two floats or None
            The expected range of the inputs. If `None` (the default), this is
            calibrated for standardized uniformly distributed inputs (i.e. an
            input range of [-2, 2] is assumed which is [`-np.sqrt(3)`,
            `np.sqrt(3)`] with a little bit of wiggle room).
        """
        if input_bounds is not None:
            self._l, self._u = input_bounds
            print("Warning: Changed matching function input bounds "
                  f"to {input_bounds}")
        else:
            # Since inputs are standardized, there is a low probability of an
            # input lying outside [-2, 2] (especially when assuming inputs
            # to be distributed uniformly).
            self._l, self._u = -2, 2

        if a is not None and mu is None:
            self.a = a
        elif a is None and mu is not None:
            self.a = 100 * (mu - self._l) / (self._u - self._l)
        else:
            raise ValueError("Exactly one of a and mu has to be given")

        if b is not None and sigma_2 is None:
            self.b = b
        elif b is None and sigma_2 is not None:
            self.b = -10 * np.log10(sigma_2)
            if not (0 <= self.b <= 50):
                raise ValueError(
                    "sigma_2 is too small (i.e. probably too close to zero)")
        else:
            raise ValueError("Exactly one of b and sigma_2 has to be given")

    def __repr__(self):
        return (f"RadialMatch1D(mu={self.mu()},sigma_2={self.sigma_2()})")

    def mu(self):
        return self._l + (self._u - self._l) * self.a / 100

    def sigma_2(self):
        return 10**(-self.b / 10)

    @classmethod
    def random(cls,
               random_state: np.random.RandomState,
               DX: int = 1,
               input_bounds=None):
        """
        [PDF p. 256]

        Parameters
        ----------
        DX : int
            Dimensionality of inputs.
        input_bounds : pair of two floats or None
            See constructor documentation.
        """
        if DX != 1:
            raise ValueError(
                "RadialMatch1D only supports 1-dimensional inputs")

        random_state = check_random_state(random_state)
        return RadialMatch1D(a=random_state.uniform(0, 100),
                             b=random_state.uniform(0, 50),
                             input_bounds=input_bounds)

    def mutate(self, random_state: np.random.RandomState):
        """
        [PDF p. 256]
        """
        # NOTE LCSBookCode puts int(...) around the normal.
        self.a = np.clip(random_state.normal(loc=self.a, scale=10), 0, 100)
        self.b = np.clip(random_state.normal(loc=self.b, scale=5), 0, 50)
        return self

    # TODO Implement __call__ instead
    def match(self, X: np.ndarray) -> np.ndarray:
        """
        Compute matching vector for given input.

        We vectorize the following (i.e. feed the whole input through at once):

            for n in range(len(X)):
                M[n] = np.exp(-0.5 / sigma_2 * (x - mu)**2)

        Parameters
        ----------
        X : array of shape (N, D_X)
            Input matrix `(N × D_X)` with `D_X == 1`.

        Returns
        -------
        array of shape (N, 1)
            Matching vector of this matching function for the given input.
        """
        # We have to clip this so we don't return 0 here (0 should never be
        # returned because every match function matches everywhere at least a
        # little bit). Also, we clip from above such that this function never
        # returns a value larger than 1 (it's a probability, after all), meaning
        # that m should not be larger than 0.
        m_min = np.log(np.finfo(None).tiny)
        m_max = 0
        # NOTE If ``self.sigma_2()`` is very close to 0 then the next line may
        # result in ``nan`` due to ``-inf * 0 = nan``. However, if we use ``b``
        # to set ``self.sigma_2()`` this problem will not occur.
        m = np.clip(-0.5 / self.sigma_2() * (X - self.mu())**2, m_min, m_max)
        return np.exp(m)

    def plot(self, l, u, ax, **kwargs):
        X = np.arange(l, u, 0.01)[:, np.newaxis]
        M = self.match(X)
        ax.plot(X, M, **kwargs)
        ax.axvline(self.mu(), color=kwargs["color"])
