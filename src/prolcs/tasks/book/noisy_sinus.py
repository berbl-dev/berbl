import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


def f(x, noise_var=0.15, random_state: np.random.RandomState = 0):
    random_state = check_random_state(random_state)
    return np.sin(2 * np.pi * x) + random_state.normal(
        0, np.sqrt(noise_var), size=x.shape)


def generate(n: int = 300, random_state: np.random.RandomState = 0):
    """
    Creates a sample from the fourth benchmark function in (Drugowitsch, 2007;
    [PDF p. 265]).

    Parameters
    ----------
    n : int
        The number of samples to generate. Supplying ``X`` overrides this.
    noise : bool
        Whether to generate noisy data (the default) or not. The latter may be
        useful for visualization purposes.
    X : Sample the function at these exact input points (instead of generating
        ``n`` input points randomly).

    Returns
    -------
    array of shape (N, 1)
        input matrix X
    array of shape (N, 1)
        output matrices y
    """
    random_state = check_random_state(random_state)

    X = random_state.uniform(low=-1, high=1, size=(n, 1))
    y = f(X, random_state=random_state)

    return X, y
