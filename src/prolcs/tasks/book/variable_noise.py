import numpy as np  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


def f(X, noise_vars=(0.6, 0.1), random_state: np.random.RandomState = 0):
    """
    Function underlying the variable noise samples.

    Parameters
    ----------
    X : array-like
        The input point(s) at which to retrieve the function values.
    noise_vars : pair of floats
        The two different noise variances for input values below ``0`` and above
        ``0``.
    """
    random_state = check_random_state(random_state)
    return np.where(
        X < 0, -1 - 2 * X
        + random_state.normal(0, np.sqrt(noise_vars[0]), size=X.shape), -1
        + 2 * X + random_state.normal(0, np.sqrt(noise_vars[1]), size=X.shape))


def generate(n: int = 200, random_state: np.random.RandomState = 0):
    """
    [PDF p. 262]

    :param n: the number of samples to generate

    :returns: input and output matrices X (N × 1) and y (N × 1)
    """
    random_state = check_random_state(random_state)

    X = random_state.uniform(low=-1, high=1, size=(n, 1))
    y = f(X, random_state=random_state)

    return X, y
