import numpy as np  # type: ignore


class AllMatch:
    """
    [``self.match``][berbl.match.allmatch.AllMatch.match] is a matching 
    function that matches all inputs.
    """
    def __init__(self):
        pass

    # TODO Use __call__ here instead
    def match(self, X: np.ndarray):
        """
        Since this matching function matches all inputs, this always returns an
        all-ones matrix of shape `(N, 1)` (with each entry corresponding to one
        of the rows of the input matrix).

        Parameters
        ----------
        X : array of shape `(N, DX)`
            Input matrix.

        Returns
        -------
        array
            Matching vector ``(N)`` of this matching function
        """
        return np.ones((len(X), 1))
