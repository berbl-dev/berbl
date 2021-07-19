from typing import *

import numpy as np  # type: ignore
from deap import tools


def phi_standard(X: np.ndarray):
    """
    The input data's mixing feature matrix usually employed by LCSs, i.e. a
    mixing feature vector of ``phi(x) = 1`` for each sample ``x``.

    :param X: input data as an ``(N, D_X)`` matrix

    :returns: all-ones mixing feature matrix ``(N, 1)``
    """
    N, D_X = X.shape
    return np.ones((N, 1))


def matching_matrix(matchs: List, X: np.ndarray):
    """
    :param ind: an individual for which the matching matrix is returned
    :param X: input matrix (N × D_X)

    :returns: matching matrix (N × K)
    """
    # TODO Can we maybe vectorize this?
    return np.hstack([m.match(X) for m in matchs])


def initRepeat_binom(container, func, n, p, random_state, kmin=1, kmax=100):
    """
    Alternative to `deap.tools.initRepeat` that samples individual sizes from a
    binomial distribution B(n, p).
    """
    size = np.clip(random_state.binomial(n, p), kmin, kmax)
    return tools.initRepeat(container, func, size)
