from functools import wraps
from time import time


def logstartstop(f):
    """
    Simple decorator for adding stdout prints when the given callable is called
    and when it returns.
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        print(f"Start {f.__name__} at {ts}")
        r = f(*args, **kw)
        te = time()
        print(f"Stop {f.__name__} after %2.4f s" % (te - ts))
        return r

    return wrap
