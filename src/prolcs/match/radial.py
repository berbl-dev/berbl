# TODO Check for standard deviation (sigma) vs variance (sigma**2) vs inverse of
# those

from typing import List

import numpy as np  # type: ignore
import scipy.special as sp  # type: ignore
import scipy.stats as sst  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from ..utils import ellipsoid_vol, radius_for_ci, space_vol


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


def _check_input_dim(dX: int, has_bias: bool):
    """
    Checks the given vector and `has_bias` flag for being suitable for
    `RadialMatch`.

    As of now, if the resulting covariance matrices etc. had dimensionality 1,
    `RadialMatch` cannot be used.

    Parameters
    ----------
    dX : int
        Dimensionality to check.
    has_bias : bool
        Whether a bias column is expected (and thus the first column is to be
        ignored during matching).

    Returns
    -------
    int
        the adjusted input dimensionality ``dX - has_bias`` (i.e. the expected
        ``X.shape[1]`` excluding bias columns).
    """
    dX = dX - has_bias
    assert dX > 1, f"Dimensionality {dX} not suitable for RadialMatch"
    return dX


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

            *Careful!* This differs from ``random_ball`` where the ``D_X``
            argument *includes* the bias column.
        """
        # This is not *that* nice, summing with ``has_bias`` here, but what can
        # you do.
        self.dX = _check_input_dim(mean.shape[0] + has_bias, has_bias)

        assert mean.shape[0] == eigvals.shape[0]
        assert mean.shape[0] == eigvecs.shape[0]
        self.mean = mean

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        self.has_bias = has_bias

    def __repr__(self):
        return (
            f"RadialMatch({tostring(self.mean)}, {tostring(self.eigvals)}, "
            f"{tostring(self.eigvecs)}, {repr(self.has_bias)})")

    @classmethod
    def random_ball(cls,
                    dX: int,
                    cover_confidence: float = 0.5,
                    coverage: float = 0.2,
                    has_bias: bool = True,
                    random_state=None):
        """
        A randomly positioned (fixed size) ball-shaped (i.e. not a general
        ellipsoid) matching function covering a given fraction of the input
        space. Input space is assumed to be ``[-1, 1]^dX`` (i.e. normalized).

        Parameters
        ----------
        dX : int ``> 1``
            Dimensionality of the input expected by this matching function
            (*not including* the bias column).
        cover_confidence : float in ``(0, 1)``
            The amount of probability mass around the mean of our Gaussian
            matching distribution that we see as being covered by the matching
            function.
        coverage : float in ``(0, 1)``
            Fraction of the input space volume that is to be covered by the
            matching function. (See also: ``cover_confidence``.)
        has_bias : bool
            Whether a bias column is included in the input. For matching, this
            means that we ignore the first column (as it is assumed to be the
            bias column and that is assumed to always be matched). Note that the
            default is to internally add a bias column which means that this
            should probably be left true.
        """
        if not has_bias:
            print("Warning: has_bias is False when it should probably be True")

        _check_input_dim(dX + has_bias, has_bias)

        random_state = check_random_state(random_state)

        high = np.ones(dX)
        low = high - 2
        mean = random_state.uniform(low=low, high=high, size=dX)

        # Input space volume. Assumes input space to be ``[-1, 1]^DdX.e.
        # normalized).
        V = space_vol(dX)

        # Radius.
        r = (coverage * V * sp.gamma(dX / 2 + 1) / (np.pi**(dX / 2)))**(1 / dX)

        # Ellipsoid matrix factor.
        lambd = r**2 / sst.chi2.ppf(cover_confidence, dX)

        # Eigenvalues are simply the ellipsoid matrix factors.
        eigvals = np.repeat(lambd, dX)

        # Due to the equal extent of all eigenvalues, the value of the
        # eigenvectors doesn't play a role at first. However, it *does* play a
        # role where we started when we begin to apply evolutionary operators on
        # these and the eigenvalues!
        eigvecs = sst.special_ortho_group.rvs(dim=dX,
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
        X : array of shape ``(N, self.dX + self.has_bias)``
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
        X : array of shape ``(N, self.dX)``
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
        return _covered_vol(self.eigvals, cover_confidence=cover_confidence)


def _covered_vol(eigvals, cover_confidence=0.5):
    """
    The volume covered by an ellipsoid based on the given eigenvalues.

    Parameters
    ----------
    cover_confidence : float in ``(0, 1)``
        The amount of mass around the ellipsoid center we see as being covered.
    """
    # The ellipsoids radii.
    rs = np.sqrt(sst.chi2.ppf(cover_confidence, len(eigvals)) * eigvals)
    return ellipsoid_vol(rs=rs, n=len(eigvals))


# TODO Extract this to search.ga module
def mutate_list(mupb=None, volstd=0.01, cover_confidence=0.5, pscale=0.1):
    """
    Create a mutation operator based on a probability of mutating each allel.

    [PDF p. 256]

    Parameters
    ----------
    mupb : float in [0, 1]
        Amount of alleles to mutate (alleles chosen randomly). If ``None``,
        mutate one allele on average.
    cover_confidence : float in ``(0, 1)``
        Used to compute the volume to change (see ``RadialMatch.covered_vol``),
        passed through to ``mutate``.
    pscale : float
        Passed through to ``mutate``.

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
                mutate(match=match,
                       random_state=random_state,
                       cover_confidence=cover_confidence,
                       volstd=volstd,
                       pscale=pscale)

        return matchs

    return f


def mutate(match: RadialMatch,
           random_state: np.random.RandomState,
           cover_confidence: float,
           volstd: float,
           pscale: float = 0.1):
    """
    Mutates the given ``RadialMatch`` in-place (undirected).

    Parameters
    ----------
    match : ``RadialMatch`` object
        The ``RadialMatch`` object to mutate.
    cover_confidence : float in ``(0, 1)``
        The amount of mass around the ellipsoid center we see as being covered.
    volstd : float
        Standard deviation for altering the volume using a normal distribution
        (mean is the current volume).
    pscale : float
        How strongly the extent of each of the ellipsoid's principal axes should
        be altered in expectation, in percent of their current value (i.e. the
        product of this with their respective current extent is used as the
        standard deviation for a normal distribution around their current
        extent).

    Returns
    -------
    RadialMatch
        The input object which has been modified in-place.
    """
    random_state = check_random_state(random_state)

    vol_old = match.covered_vol(cover_confidence)
    vol_new = np.clip(random_state.normal(loc=vol_old, scale=volstd), a_min=0, a_max=np.inf)
    vol_diff = vol_old - vol_new

    match.eigvals = _stretch(match.eigvals,
                             voldiff=vol_diff,
                             scale=pscale,
                             cover_confidence=cover_confidence,
                             random_state=random_state)

    # Choose plane regarding which to rotate.
    #
    # First, rank all eigenvalues, largest rank to the lowest one and make ranks
    # non-zero (eigenvectors and eigenvalues are in the same order).
    ranks = np.argsort(-match.eigvals) + 1
    # TODO Maybe use a more elaborate scheme rather than using the ranks 1:1 for
    # the weights. E.g. p*(1 - p)**rank or similar.
    # Normalize distribution.
    weights = ranks / np.sum(ranks)
    i1, i2 = tuple(
        random_state.choice(list(range(len(match.eigvecs))),
                            size=2,
                            replace=False,
                            p=weights))

    # Note: There is a little bit of thought behind the average angle of 18 °;
    # it's not entirely arbitrary (although almost entirely).
    # TODO Provide proper formal background for average rotation angle
    angle = random_state.normal(loc=18, scale=18)
    sign = 1 if random_state.random() < 0.5 else -1
    angle = sign * angle

    match.eigvecs = _rotate(match.eigvecs, angle, i1, i2)

    return match


def _stretch(eigvals: np.ndarray, voldiff: float, scale: float,
             cover_confidence: float, random_state: np.random.RandomState):
    r"""
    Parameters
    ----------
    eigvals : array of shape ``(dX,)``
        The eigenvalues to stretch.
    voldiff : float
        The covered volume we want to add (may be negative to remove volume).
        Note that no checks on the size of this volume are performed.
    scale : float
        All but one eigenvalue is assigned a new value based on the old value
        using a normal distribution:

        .. math:: \lambda_\text{new} = \mathcal{N}(\lambda, \lambda * \text{scale})

    cover_confidence : float in ``(0, 1)``
        The amount of mass around the ellipsoid center we see as being covered.
    """
    n = eigvals.shape[0]

    # The index of the eigenvalue that is used to balance the eigenvalue
    # product in the end.
    i0 = random_state.randint(0, n)

    # Draw ``n`` eigenvalues (one too many for simplicity's sake).
    # TODO We need something better than ``eigvals * scale``, see notes.
    eigvals_ = random_state.normal(loc=eigvals, scale=eigvals * scale)

    # “Delete” entry we want to replace in the next steps (1 is neutral wrt
    # multiplication).
    eigvals_[i0] = 1

    # Balance eigenvalues such that, in the end, the volume changed by the
    # specified amount.
    n = len(eigvals)
    eigval_0 = np.prod(np.sqrt(eigvals))
    raise NotImplementedError("This may not work yet, see comments")
    # test_volume_after__stretch fails due to eigvals_ containing negative
    # values. My current guess is that this is due to voldiff not being used as
    # an absolute volume but as a percentage in _stretch (but being set up as an
    # absolute volume in the corresponding test). Check Zettels for whether its
    # semantics are as I think they are.
    eigval_0 += (
        voldiff * sp.gamma(n / 2 + 1) /
        (np.pi**(n / 2) * np.sqrt(sst.chi2.ppf(cover_confidence, n))**n))
    # assert eigval_0_ == eigval_0
    eigval_0 **= 2
    eigval_0 *= 1 / np.prod(eigvals_)

    eigvals_[i0] = eigval_0

    return eigvals_


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
    dX = len(eigvecs)

    v1, v2 = eigvecs[i1], eigvecs[i2]

    # Angle is in degrees, thus get radian.
    angle = angle * 2 * np.pi / 360

    # Calculate rotation matrix.
    V = np.outer(v1, v1) + np.outer(v2, v2)
    W = np.outer(v1, v2) - np.outer(v2, v1)
    R = np.identity(dX) + (np.cos(angle) - 1) * V + np.sin(angle) * W

    # Rotate eigenvectors.
    eigvecs = R @ eigvecs

    return eigvecs


def tostring(a):
    return ("np.array(" + np.array2string(
        a, separator=",", max_line_width=np.inf, threshold=None) + ")")
