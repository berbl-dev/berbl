from typing import *

import numpy as np  # type: ignore
from deap import tools


# TODO Move all of this to utils


def check_phi(phi, X: np.ndarray):
    """
    Given a mixing feature mapping ``phi``, compute the mixing feature matrix
    ``Phi``.

    If ``phi`` is ``None``, use the default LCS mixing feature mapping, i.e. a
    mixing feature vector of ``phi(x) = 1`` for each data point ``x``.

    Parameters
    ----------
    phi : callable receiving ``X`` or ``None``
        Mixing feature extractor (N × D_X → N × D_V); if ``None`` uses the
        default LCS mixing feature matrix based on ``phi(x) = 1``.
    X : array of shape (N, D_X)
        Input matrix.

    Returns
    -------
    Phi : array of shape (N, D_V)
        Mixing feature matrix.
    """
    # NOTE This is named like this in order to stay close to sklearn's naming
    # scheme (e.g. check_random_state etc.).

    N, _ = X.shape

    if phi is None:
        Phi = np.ones((N, 1))
    else:
        Phi = phi(X)

    return Phi


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
