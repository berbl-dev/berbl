from functools import wraps
from time import asctime, localtime, time

import numpy as np  # type: ignore
import scipy.stats as st  # type: ignore
import scipy.special as sp  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
import scipy.optimize as so  # type: ignore


def logstartstop(f):
    """
    Simple decorator for adding stdout prints when the given callable is called
    and when it returns.
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        print(f"Start {f.__name__} at {asctime(localtime(ts))}")
        r = f(*args, **kw)
        te = time()
        print(f"Stop {f.__name__} after %2.4f s" % (te - ts))
        return r

    return wrap


def get_ranges(X: np.ndarray):
    """
    Computes the value range for each dimension.

    :param X: input data as an ``(N, D_X)`` matrix

    :returns: a ``(D_X, 2)`` matrix where each row consists the minimum and
        maximum in the respective dimension
    """
    return np.vstack([np.min(X, axis=0), np.max(X, axis=0)]).T


def add_bias(X: np.ndarray):
    """
    Prefixes each input vector (i.e. row) in the given input matrix with 1 for
    fitting the intercept.

    :param X: input data as an ``(N, D_X)`` matrix

    :returns: a ``(N, D_X + 1)`` matrix where each row is the corresponding
        original matrix's row prefixed with 1
    """
    N, D_X = X.shape
    return np.hstack([np.ones((N, 1)), X])


def pr_in_sd1(r=1):
    """
    Expected percentage of examples falling within one standard deviation of a
    one-dimensional Gaussian distribution. See ``pr_in_sd``.

    Parameters
    ----------
    r : float
        Radius (in multiples of standard deviation).
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erf.html
    return sp.erf(r / np.sqrt(2))


def pr_in_sd2(r=1):
    """
    Expected percentage of examples falling within one standard deviation of a
    two-dimensional Gaussian distribution. See ``pr_in_sd``.

    Parameters
    ----------
    r : float
        Radius (in multiples of standard deviation).
    """
    return 1 - np.exp(-(r**2) / 2)


def pr_in_sd(n=3, r=1):
    """
    Expected percentage of examples falling within multiples of a standard
    deviation of a multivariate Gaussian distribution.

    Reference for the used formulae:
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118537 .

    Parameters
    ----------
    n : positive int
        Dimensionality of the Gaussian.
    r : float greater than 1
        Factor for standard deviation radius. Numerical issues(?) if radius too
        close to zero. We probably never want require r < 1, though, so this is
        probably fine.
    """
    if r < 1:
        raise ValueError(f"r = {r} < 1 may result in numerical issues")

    if n == 1:
        return pr_in_sd1(r=r)
    elif n == 2:
        return pr_in_sd2(r=r)
    elif n >= 3:
        ci = pr_in_sd(n=n - 2, r=r) - (r / np.sqrt(2))**(n - 2) * np.exp(
            -(r**2) / 2) / sp.gamma(n / 2)
        if ci < 0 and np.isclose(ci, 0):
            return 0
        else:
            return ci
    else:
        raise ValueError("n must be positive")


pr_in_sd_ = np.vectorize(pr_in_sd)


def radius_for_ci(n=3, ci=0.5):
    """
    Calculate how many standard deviations are required to fulfill the given
    confidence interval for a multivariate Gaussian of the given dimensionality.
    """
    # Other than in this German Wikipedia article we actually need to use
    # SciPy's inverse of the *lower* incomplete gamma function:
    # https://de.wikipedia.org/wiki/Mehrdimensionale_Normalverteilung
    return np.sqrt(2 * sp.gammaincinv(n / 2, ci))


radius_for_ci_ = np.vectorize(radius_for_ci)
