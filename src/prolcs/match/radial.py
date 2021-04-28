# TODO Check for standard deviation (sigma) vs variance (sigma**2) vs inverse of
# those

import numpy as np  # type: ignore
import scipy.stats as st  # type: ignore
from ..utils import radius_for_ci
from sklearn.utils import check_random_state  # type: ignore


def _check_dimensions(D_X):
    assert D_X > 1, f"Dimensionality {D_X} not suitable for RadialMatch"


class RadialMatch():
    """
    Radial basis function–based matching for dimensions greater than 1.

    Important: The very first column is always matched as we expect it to be a
    bias column.
    """
    def __init__(self, mu: np.ndarray, eigvals: np.ndarray,
                 eigvecs: np.ndarray, has_bias=True):
        """
        Parameters
        ----------

        mu : array
             Position of the Gaussian.
        eigvals : array
            Eigenvalues of the Gaussian's precision matrix.
        eigvecs : array
            Eigenvectors of the Gaussian's precision matrix.
        """
        self.D_X = mu.shape[0]
        _check_dimensions(self.D_X)

        assert mu.shape[0] == eigvals.shape[0]
        assert mu.shape[0] == eigvecs.shape[0]
        self.mu = mu

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        self.has_bias = has_bias

    def __repr__(self):
        return f"RadialMatch({self.mu}, {self.eigvals}, {self.eigvecs})"

    @classmethod
    def random(cls, ranges: np.ndarray, has_bias=True, ci=0.2, random_state=None):
        """
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
        ci : float in `(0, 1)`
            Ratio of samples expected to fall into one standard deviation (i.e.
            how wide the resulting ellipsoid is).
        """
        D_X, _ = ranges.shape
        assert _ == 2

        _check_dimensions(D_X)

        random_state = check_random_state(random_state)

        mu = random_state.uniform(low=ranges[:, [0]].reshape((-1)),
                                  high=ranges[:, [1]].reshape((-1)),
                                  size=D_X)

        # If this were a ball, then ``r`` would be its radius.
        r = radius_for_ci(n=D_X, ci=ci)

        # The index of the eigenvalue that is used to balance the eigenvalue
        # product in the end.
        i0 = random_state.randint(0, D_X)

        # Draw ``D_X - 1`` eigenvalues, each from a uniform distribution with
        # expected value ``r``.
        rs = random_state.uniform(r - r / 2, r + r / 2, size=D_X - 1)
        # Balance eigenvalues such that, in the end, ``r**D_X =
        # np.prod(eigvals)`` (i.e. the ellipsoid has the same volume as a ball
        # with radius ``r`` had).
        r0 = r**D_X / np.prod(rs)
        rs = np.insert(rs, i0, r0)
        # Eigenvalues are the square of the radii.
        eigvals = rs**2

        # TODO Unsure: Do I need to use the special orthogonal group here (i.e.
        # enforce det = +1)?
        # TODO Yes, I think we indeed need this since this way the length of the
        # vectors is 1??
        eigvecs = st.special_ortho_group.rvs(dim=D_X, random_state=None)
        # eigvecs = st.ortho_group.rvs(dim=D_X, random_state=None)

        # TODO May need to use range somehow (eigenvalues seem pretty large for
        # [-1, 1] currently).

        return RadialMatch(mu=mu, eigvals=eigvals, eigvecs=eigvecs)

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
        # X_mu = X - self.mu
        # # The ``np.sum`` is a vectorization of ``(X_mu[n].T @ Lambda @
        # # X_mu[n])`` for all ``n``.
        # m = np.exp(-0.5 * np.sum((X_mu @ Lambda) * X_mu, axis=1))
        # m = m / (np.sqrt(2 * np.pi)**self.D_X * det_Sigma)

        # Construct covariance matrix.
        Sigma = self._covariance()

        # I'm pretty certain that using SciPy is more efficient than writing it
        # in Python.
        m = st.multivariate_normal(mean=self.mu, cov=Sigma).pdf(X)

        # SciPy is too smart. If ``X`` only contains one example, then
        # ``st.multivariate_normal`` returns a float (instead of an array).
        if len(X) == 1:
            m = np.array([m])

        # ``m`` can be zero (when it shouldn't be ever) due to floating point
        # problems.
        m = np.clip(m, a_min=np.finfo(None).tiny, a_max=1)

        return m[:,np.newaxis]
