import numpy as np  # type: ignore


class NoMatch:
    """
    [``self.match``][berbl.match.nomatch.NoMatch.match] is a matching function 
    that doesn't match any of the inputs given to it. `Not matching` meaning 
    here that the smallest positive non-zero number is returned (i.e. not 
    matching in a fuzzy matching sense).
    """
    def __init__(self):
        pass

    def match(self, X: np.ndarray):
        """
        Since this matching function matches no inputs, this always returns an
        all-ones (N × 1) matrix (with each entry corresponding to one of the
        rows of the input matrix).

        Parameters
        ----------
        X : array of shape (N, D_X)
            Input matrix ``(N × D_X)`` with ``D_X == 1``

        Returns
        -------
        array of shape (N)
            Matching vector of this matching function
        """
        return np.repeat(np.finfo(None).tiny, len(X))[:, np.newaxis]
