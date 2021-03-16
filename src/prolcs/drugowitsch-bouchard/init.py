import numpy as np
from prolcs.common import phi_standard
from prolcs.drugowitsch.model import Model
from prolcs.drugowitsch.radialmatch1d import RadialMatch1D
from prolcs.utils import get_ranges


def _individual(k: int, ranges: np.ndarray,
               random_state: np.random.RandomState):
    """
    Individuals are simply lists of RadialMatch1D matching functions (the length
    of the list is the number of classifiers, the matching functions specify
    their localization).
    """
    return Model([
        RadialMatch1D.random(ranges, random_state=random_state)
        for i in range(k)
    ],
                 phi=phi_standard)


def make_init(n, p, size, kmin=1, kmax=100):
    """
    Creates a generator ``init`` for a population of 1D individuals whose sizes
    are drawn from the distribution ``np.clip(binomial(n, p, size=size), kmin,
    kmax)`` (Drugowitsch problem-dependently samples individual sizes from such
    distribution as well [PDF p. 221, 3rd paragraph]). The signature of ``init``
    is as follows::

        def init(X, Y, random_state)

    The range of values in ``X`` (required for ``RadialMatch1D`` individuals) is
    computed using ``get_ranges``.
    """
    def init(X, Y, random_state):
        Ks = np.clip(random_state.binomial(n, p, size=size), 1, 100)
        ranges = get_ranges(X)
        ranges = tuple(ranges[0])
        return [_individual(k, ranges, random_state) for k in Ks]

    return init
