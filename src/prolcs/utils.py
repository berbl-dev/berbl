from functools import wraps
from time import asctime, localtime, time
import numpy as np  # type: ignore


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
