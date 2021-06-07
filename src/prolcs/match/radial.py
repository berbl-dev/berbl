# TODO Check for standard deviation (sigma) vs variance (sigma**2) vs inverse of
# those

from typing import List

import numpy as np  # type: ignore
import scipy.special as sp  # type: ignore
import scipy.stats as sst  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from ..utils import ellipsoid_vol, radius_for_ci, ranges_vol


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


def _check_input_dim(mean, has_bias):
    """
    Checks the given vector and `has_bias` flag for being suitable for
    `RadialMatch`.

    As of now, if the resulting covariance matrices etc. had dimensionality 1,
    `RadialMatch` cannot be used.

    Returns
    -------
    int
        the real input dimensionality ``D_X`` (i.e. the expected ``X.shape[1]``
        including bias columns).
    """
    D_X = mean.shape[0]
    if has_bias:
        D_X += 1
    assert ((not has_bias and D_X > 1)
            or (has_bias and
                D_X > 2)), f"Dimensionality {D_X} not suitable for RadialMatch"
    return D_X


class RadialMatch():
    """
    Radial basis function–based matching for dimensions greater than 1.
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
        has_bias : bool
            Whether the input data's first column is expected to be an all-ones
            bias column that is always matched. If this is the case, then input
            data's dimensionality is expected to be ``mean.shape[0] + 1`` (i.e.
            ``mean``, ``eigvals`` and ``eigvecs`` don't have entries regarding
            the bias columns).
        """
        self.D_X = _check_input_dim(mean, has_bias)

        # Bias columns do not take part in matching so we adjust by subtracting
        # 1 if a bias column is expected to be part of the input.
        self.D_X_adj = self.D_X - has_bias

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
        ranges : array of shape ``(D_X, 2)``
            A value range pair per input dimension. We assume that ``ranges``
            never contains an entry for a bias column (see ``has_bias``).
        has_bias : bool
            Whether a bias column is included in the input. For matching, this
            means that we ignore the first column (as it is assumed to be the
            bias column and that is assumed to always be matched). Note that if
            ``has_bias``, then ``ranges.shape[0] = X.shape[1] - 1 = D_X - 1`` as
            ranges never contains an entry for a bias column.
        cover_confidence : float in ``(0, 1)``
            The amount of probability mass around the mean of our Gaussian
            matching distribution that we see as being covered by the matching
            function.
        coverage : float in ``(0, 1)``
            Fraction of the input space volume that is to be covered by the
            matching function. (See also: ``cover_confidence``.)
        """
        assert ranges.shape[1] == 2

        D_X = _check_input_dim(ranges[:, [0]], has_bias)

        # Bias columns do not take part in matching so we adjust by subtracting
        # 1 if a bias column is expected to be part of the input.
        D_X_adj = D_X - has_bias

        random_state = check_random_state(random_state)

        mean = random_state.uniform(low=ranges[:, [0]].reshape((-1)),
                                    high=ranges[:, [1]].reshape((-1)),
                                    size=D_X_adj)

        # Input space volume.
        V = ranges_vol(ranges)

        # Radius.
        r = (coverage * V * sp.gamma(D_X_adj / 2 + 1) /
             (np.pi**(D_X_adj / 2)))**(1 / D_X_adj)

        # Ellipsoid matrix factor.
        lambd = r**2 / sst.chi2.ppf(cover_confidence, D_X_adj)

        # Eigenvalues are simply the ellipsoid matrix factors.
        eigvals = np.repeat(lambd, D_X_adj)

        # Due to the equal extent of all eigenvalues, the value of the
        # eigenvectors doesn't play a role at first. However, it *does* play a
        # role where we started when we begin to apply evolutionary operators on
        # these and the eigenvalues!
        eigvecs = sst.special_ortho_group.rvs(dim=D_X_adj,
                                              random_state=random_state)

        # TODO Since I restrict input space I could (or, maybe, must?)
        # normalize matching distribution function regarding that?
        return RadialMatch(mean=mean,
                           eigvals=eigvals,
                           eigvecs=eigvecs,
                           has_bias=has_bias)

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
        # The inverse of the eigenvector matrix is actually always its transpose
        # due to orthonormality.
        return self.eigvecs @ np.diag(self.eigvals) @ self.eigvecs.T

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
        # NOTE The following is a faster version (factor 5 for D_X = 30) but may
        # cause numerical issues as no care at all regarding those is taken.
        # Therefore we'll stick with the slower but hopefully safer version.
        #
        # Construct inverse covariance matrix.
        # Lambda = self._inv_covariance()
        # det_Sigma = 1 / np.linalg.det(Lambda)
        # X_mu = X - self.mean
        # # The ``np.sum`` is a vectorization of ``(X_mu[n].T @ Lambda @
        # # X_mu[n])`` for all ``n``.
        # m = np.exp(-0.5 * np.sum((X_mu @ Lambda) * X_mu, axis=1))
        # # The usual normalization factor.
        # m = m / (np.sqrt(2 * np.pi)**self.D_X * det_Sigma)
        # # ``m`` can be zero (when it shouldn't be ever) due to floating point
        # # problems.
        # m = np.clip(m, a_min=np.finfo(None).tiny, a_max=1)
        # return m[:, np.newaxis]

        # TODO Performance: May be better if we used one of the private
        # functions of multivariate_normal (and then have this object not store
        # the covariance but the precision matrix to get around the costly
        # inversion).
        # TODO Performance: Or use the logpdf formula directly from
        # https://github.com/scipy/scipy/blob/v1.6.3/scipy/stats/_multivariate.py#L452
        # TODO Performance: Or implement myself, see previous paragraph
        Sigma = self._covariance()
        m = sst.multivariate_normal(mean=self.mean, cov=Sigma).pdf(X)

        # SciPy is too smart. If ``X`` only contains one example, then
        # ``sst.multivariate_normal`` returns a float (instead of an array).
        if len(X) == 1:
            m = np.array([m])

        # ``m`` can be zero (when it shouldn't be ever) due to floating point
        # problems.
        m = np.clip(m, a_min=np.finfo(None).tiny, a_max=1)

        return m[:, np.newaxis]

    def covered_vol(self, cover_confidence=0.5):
        """
        The volume covered by this matching function.

        Parameters
        ----------
        cover_confidence : float in ``(0, 1)``
            The amount of probability mass around the mean of our Gaussian
            matching distribution that we see as being covered by the matching
            function.
        """
        rs = np.sqrt(
            radius_for_ci(ci=cover_confidence, n=self.D_X_adj) * self.eigvals)
        return ellipsoid_vol(rs=rs, n=self.D_X_adj)


# TODO Extract this to search.ga module
def mutate_list(mupb=None):
    """
    Create a mutation operator based on a probability of mutating each allel.

    [PDF p. 256]

    Parameters
    ----------
    mupb : float in [0, 1]
        Amount of alleles to mutate (alleles chosen randomly). If ``None``,
        mutate one allele on average.


    Returns
    -------
    RadialMatch
        The input object which has been modified in-place.
    """
    def f(matchs: List[RadialMatch], random_state: np.random.RandomState):
        random_state = check_random_state(random_state)

        if mupb is None:
            p = 1 / len(matchs)
        else:
            p = mupb

        for match in matchs:
            if random_state.random() < p:
                mutate(match, random_state)

        return matchs

    return f


# TODO Extract this to search.ga module
def mutate(match: RadialMatch, random_state: np.random.RandomState):
    """
    Mutates the given RadialMatch in-place.

    [PDF p. 256]

    Returns
    -------
    RadialMatch
        The input object which has been modified in-place.
    """
    random_state = check_random_state(random_state)

    match.eigvals += random_state.random(size=match.eigvals.shape)

    # Choose plane regarding which to rotate.
    i1, i2 = tuple(
        random_state.choice(list(range(len(match.eigvecs))),
                            size=2,
                            replace=False))

    # Angle in degrees, then get radian.
    raise NotImplementedError("Angle is fixed right now, need distribution")
    angle = 10

    match.eigvecs = _rotate(match.eigvecs, angle)

    return match


def _rotate(eigvecs: np.ndarray, angle: float, i1, i2):
    """
    Rotates the given array of orthonormal eigenvectors by the given angle.

    Parameters
    ----------
    eigvecs : array
        Array of eigenvectors to rotate.
    angle : float
        Angle (in degree) to rotate by.
    i1, i2 : int
        Indices of the eigenvectors which create the plane regarding which to
        rotate.
    """
    D_X_adj = len(eigvecs)

    v1, v2 = eigvecs[i1], eigvecs[i2]

    # Angle is in degrees, thus get radian.
    angle = angle * 2 * np.pi / 360

    # Calculate rotation matrix.
    V = np.outer(v1, v1) + np.outer(v2, v2)
    W = np.outer(v1, v2) - np.outer(v2, v1)
    R = np.identity(D_X_adj) + (np.cos(angle) - 1) * V + np.sin(angle) * W

    # Rotate eigenvectors.
    eigvecs = R @ eigvecs

    return eigvecs
