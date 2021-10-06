import numpy as np  # type: ignore


class AllMatch:
    """
    ``self.match`` is a matching function that matches all inputs.
    """
    def __init__(self):
        pass

    def match(self, X: np.ndarray):
        """
        Since this matching function matches all inputs, this always returns an
        all-ones (N × 1) matrix (with each entry corresponding to one of the
        rows of the input matrix).

        :param X: input matrix ``(N × D_X)`` with ``D_X == 1``
        :returns: matching vector ``(N)`` of this matching function
        """
        return np.ones((len(X), 1))
