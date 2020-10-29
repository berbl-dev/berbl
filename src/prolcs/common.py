from typing import *

import numpy as np  # type: ignore


def phi_standard(X: np.ndarray):
    """
    The mixing feature extractor usually employed by LCSs, i.e. ``phi(x) = 1``
    for each sample ``x``.

    :param X: input data as an ``(N, D_X)`` matrix

    :returns: a ``(D_X, 2)`` matrix where each row consists the minimum and
        maximum in the respective dimension
    """
    N, D_X = X.shape
    return np.ones((N, 1))


def matching_matrix(ind: List, X: np.ndarray):
    """
    :param ind: an individual for which the matching matrix is returned
    :param X: input matrix (N × D_X)

    :returns: matching matrix (N × K)
    """
    # TODO Can we maybe vectorize this?
    return np.hstack(list(map(lambda m: m.match(X), ind)))
