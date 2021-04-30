# TODO Check for standard deviation (sigma) vs variance (sigma**2) vs inverse of
# those

import numpy as np  # type: ignore
import scipy.stats as st  # type: ignore
import scipy.special as sp  # type: ignore
from ..utils import radius_for_ci
from sklearn.utils import check_random_state  # type: ignore


# TODO Add support for initially centering means on training data
def random_balls(n, **kwargs):
    """
    Parameters
    ----------
    n : positive int
        How many random ``RadialMatch`` instances to generate.
    **kwargs
        Passed through to ``RadialMatch.random_ball``. The only exception is
        ``random_state`` which is expected as a parameter by the returned
        function.

    Returns
    -------
    callable expecting a ``RandomState``
        A distribution over ``n``-length lists of ``RadialMatch.random_ball``s.
    """
    def p(random_state):
        return [
            RadialMatch.random_ball(random_state=random_state, **kwargs)
            for _ in range(n)
        ]

    return p


def _check_dimensions(D_X):
    assert D_X > 1, f"Dimensionality {D_X} not suitable for RadialMatch"


class RadialMatch():
    """
    Radial basis function–based matching for dimensions greater than 1.

    Important: The very first column is always matched as we expect it to be a
    bias column.
    """
    def __init__(self,
                 mean: np.ndarray,
                 eigvals: np.ndarray,
                 eigvecs: np.ndarray,
                 has_bias=True):
        """
        Parameters
        ----------

        mean : array
             Position of the Gaussian.
        eigvals : array
            Eigenvalues of the Gaussian's precision matrix.
        eigvecs : array
            Eigenvectors of the Gaussian's precision matrix.
        """
        self.D_X = mean.shape[0]
        _check_dimensions(self.D_X)

        assert mean.shape[0] == eigvals.shape[0]
        assert mean.shape[0] == eigvecs.shape[0]
        self.mean = mean

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        self.has_bias = has_bias

    def __repr__(self):
        return f"RadialMatch({self.mean}, {self.eigvals}, {self.eigvecs})"

    @classmethod
    def random_ball(cls,
                    ranges: np.ndarray,
                    has_bias=True,
                    cover_confidence=0.5,
                    coverage=0.2,
                    random_state=None):
        """
        A randomly positioned (fixed size) ball-shaped (i.e. not a general
        ellipsoid) matching function covering a given fraction of the input
        space.

        Parameters
        ----------
        ranges : array of shape ``(X_D, 2)``
            A value range pair per input dimension. We assume that ranges never
            contains an entry for a bias column.
        has_bias : bool
            Whether a bias column is included in the input. For matching, this
            means that we ignore the first column (as it is assumed to be the
            bias column and that is assumed to always be matched). Note that if
            ``has_bias``, then ``ranges.shape[0] = X.shape[1] - 1`` as ranges
            never contains an entry for a bias column.
        cover_confidence : float in ``(0, 1)``
            The amount of probability mass around the mean of our Gaussian
            matching distribution that we see as being covered by the matching
            function.
        coverage : float in ``(0, 1)``
            Fraction of the input space volume that is to be covered by the
            matching function. (See also: ``cover_confidence``.)
        """
        D_X, _ = ranges.shape
        assert _ == 2

        _check_dimensions(D_X)

        random_state = check_random_state(random_state)

        mean = random_state.uniform(low=ranges[:, [0]].reshape((-1)),
                                  high=ranges[:, [1]].reshape((-1)),
                                  size=D_X)

        # Input space volume.
        V = np.prod(np.diff(ranges))

        r = radius_for_ci(n=D_X, ci=cover_confidence)

        # sigma^n.
        sigma_n = coverage * V * sp.gamma(D_X / 2 + 1) / (np.pi**(D_X / 2)
                                                          * r**D_X)
        # Draw nth root to get sigma.
        sigma = sigma_n**(1. / D_X)

        # Eigenvalues are the squares of the sigmas.
        eigvals = np.repeat(sigma**2, D_X)

        # Due to the equal extent of all eigenvalues, the value of the
        # eigenvectors doesn't play a role at first. However, it *does* play a
        # role where we started when we begin to apply evolutionary operators on
        # these and the eigenvalues!
        eigvecs = st.special_ortho_group.rvs(dim=D_X,
                                             random_state=random_state)

        return RadialMatch(mean=mean, eigvals=eigvals, eigvecs=eigvecs)

    def match(self, X: np.ndarray) -> np.ndarray:
        """
        Compute matching vector for the given input.

        If ``self.has_bias``, we expect inputs to contain a bias column (which
        is always matched) and thus remove the first column beforehand.

        Parameters
        ----------
        X : array of shape ``(N, D_X)``
            Input matrix.

        Returns
        -------
        array of shape ``(N)``
            Matching vector of this matching function for the given input.
        """
        if self.has_bias:
            X = X.T[1:].T

        return self._match_wo_bias(X)

    def _covariance(self):
        """
        This matching function's covariance matrix.
        """
        return self.eigvecs @ np.diag(self.eigvals) @ np.linalg.inv(
            self.eigvecs)

    def _match_wo_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Matching function but assume the bias column to not be part of the
        input.

        Parameters
        ----------
        X : array of shape ``(N, D_X)``
            Input matrix.

        Returns
        -------
        array of shape ``(N,)``
            Matching vector
        """
        # NOTE I could work directly on Lambda instead of on Sigma (and then use
        # det(Sigma) = 1 / det(Lambda)). But is that more efficient than SciPy?
        # I doubt it.
        #
        # Lambda = self._covariance()
        # det_Sigma = 1 / np.linalg.det(Lambda)
        # X_mu = X - self.mean
        # # The ``np.sum`` is a vectorization of ``(X_mu[n].T @ Lambda @
        # # X_mu[n])`` for all ``n``.
        # m = np.exp(-0.5 * np.sum((X_mu @ Lambda) * X_mu, axis=1))
        # m = m / (np.sqrt(2 * np.pi)**self.D_X * det_Sigma)

        # Construct covariance matrix.
        Sigma = self._covariance()

        # I'm pretty certain that using SciPy is more efficient than writing it
        # in Python.
        m = st.multivariate_normal(mean=self.mean, cov=Sigma).pdf(X)

        # SciPy is too smart. If ``X`` only contains one example, then
        # ``st.multivariate_normal`` returns a float (instead of an array).
        if len(X) == 1:
            m = np.array([m])

        # ``m`` can be zero (when it shouldn't be ever) due to floating point
        # problems.
        m = np.clip(m, a_min=np.finfo(None).tiny, a_max=1)

        return m[:, np.newaxis]
