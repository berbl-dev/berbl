"""
Tools for initializing lists of matching functions (i.e. model structures).
"""

import numpy as np  # type: ignore


def binomial_init(n, p, allel_init, kmin=None, kmax=None, **kwargs):
    """
    Creates a distribution over lists of `RadialMatch` based on a binomial
    distribution over the lists' lengths.

    List lengths are drawn from the distribution `np.clip(binomial(n, p,
    size=size), kmin, kmax)`.

    (Drugowitsch problem-dependently samples individual sizes from such
    distribution as well [PDF p. 221, 3rd paragraph]).

    Parameters
    ----------
    n : int
        n-parameter of the underlying binomial distribution.
    p : float
        p-parameter of the underlying binomial distribution.
    allel_init : callable
        Callable for initializing a single random allele (e.g.
        `RadialMatch.random_ball`), receives.
    kmin : positive int
        Minimum value for individual lengths. If `None` (the default), assume
        `kmin = 1`.
    kmax : int
        Maximum value for individual lengths.  If `None` (the default), assume
        `kmax = 10 * n`.
    **kwargs
        Passed through to `func` unchanged.

    Returns
    -------
    callable receiving a `np.random.RandomState`
        A distribution over lists of random `RadialMatch` objects.
    """
    if kmin is None:
        kmin = 1
    if kmax is None:
        kmax = 10 * n

    # TODO Consider adding a parameter size (passed through to binomial etc.) to
    # be more efficient
    def init(random_state):
        # Draw a solution size.
        K = np.clip(random_state.binomial(n, p), a_min=kmin, a_max=kmax)
        # Initialize a solution.
        return [
            allel_init(random_state=random_state, **kwargs) for k in range(K)
        ]

    return init
