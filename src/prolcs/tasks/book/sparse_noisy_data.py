import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

def f(x,
      noise_var: float = 0.2,
      random_state: np.random.RandomState = 0):
    """
    Parameters
    ----------
    x : array_like, ``0 <= x <= 4``
        Location to get the function value at.
    noise_var : float
        Noise variance (``0.2`` acc. to [PDF p. 262]).
    """
    random_state = check_random_state(random_state)

    # NOTE LCSBookCode's first factor is 4.26 which differs from the book's
    # 4.25.
    return 4.25 * (np.exp(-x) - 4 * np.exp(-2 * x)
                   + 3 * np.exp(-3 * x)) + random_state.normal(
                       0, np.sqrt(noise_var), size=x.shape)


def generate(n: int = 200, random_state: np.random.RandomState = 0):
    """
    Creates a sample from the second benchmark function in (Drugowitsch, 2007;
    [PDF p. 262]). This sample has considerably more noise and is sparser than
    samples from the first benchmark function.

    Note that the input range is [0, 4] (i.e. standardization may be necessary).

    Parameters
    ----------
    n : int
        The size of the sample to generate. Supplying ``X`` overrides this.

    Returns
    -------
    array of shape (N, 1)
        input matrix X
    array of shape (N, 1)
        output matrices y
    """
    random_state = check_random_state(random_state)

    X = random_state.uniform(low=0, high=4, size=(n, 1))
    Y = f(X, random_state=random_state)

    return X, Y
