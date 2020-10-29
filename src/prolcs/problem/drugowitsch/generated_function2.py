# TODO It sometimes seems to hang somewhere (i.e. not finishing the generation)
# TODO Why is variance increasing that much on the right side?!
import numpy as np  # type: ignore
from prolcs.common import matching_matrix, phi_standard
from prolcs.drugowitsch import mixing
from prolcs.drugowitsch.ga1d import DrugowitschGA1D
from prolcs.radialmatch1d import RadialMatch1D
from prolcs.utils import get_ranges
from sklearn.utils import check_random_state  # type: ignore

# The individual used in function generation.
ms = [
    RadialMatch1D(mu=0.2, sigma_2=0.05, ranges=(0, 1)),
    RadialMatch1D(mu=0.5, sigma_2=0.01, ranges=(0, 1)),
    RadialMatch1D(mu=0.8, sigma_2=0.05, ranges=(0, 1)),
]


def generate(n: int = 300, random_state: np.random.RandomState = 0):
    """
    [PDF p. 260]

    :param n: the number of samples to generate

    :returns: input and output matrices X (N × 1) and Y (N × 1)
    """
    random_state = check_random_state(random_state)

    X = random_state.random((n, 1))

    M = matching_matrix(ms, X)
    Phi = phi_standard(X)

    W = [
        np.array([0.05, 0.5]),
        np.array([2, -4]),
        np.array([-1.5, 2.5]),
    ]
    Lambda_1 = [
        np.array([0.1]),
        np.array([0.1]),
        np.array([0.1]),
    ]
    V = np.array([0.5, 1.0, 0.4]).reshape(1, 3)

    G = mixing(M, Phi, V)

    # After matching, augment samples by prepending 1 to enable non-zero
    # intercepts.
    X_ = np.vstack([np.ones(n), X.reshape((n))]).T
    Y = np.zeros(X.shape)
    for n in range(len(X)):
        y = 0
        for k in range(len(ms)):
            # sample the three classifiers
            y += random_state.normal(loc=G[n][k] * (W[k] @ X_[n]), scale=Lambda_1[k])
        Y[n] = y

    # We return the non-augmented samples (because our algorithm augments them
    # itself).
    return X, Y


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    import mlflow

    LOGGING = "mlflow"

    seed = 0

    mlflow.set_experiment("generated_function")
    with mlflow.start_run() as run:
        mlflow.log_param("seed", seed)
        random_state = check_random_state(seed)

        X, Y = generate()

        ranges = (0, 1)


        # TODO Use random_state
        def individual(k: int):
            """
            Individuals are simply lists of matching functions (the length of
            the list is the number of classifiers, the matching functions
            specify their localization).
            """
            return [RadialMatch1D.random(ranges, random_state=random_state) for i in range(k)]

        # [PDF p. 221, 3rd paragraph]
        # Drugowitsch samples individual sizes from a certain problem-dependent
        # Binomial distribution.
        def init(X, Y):
            Ks = np.clip(random_state.binomial(8, 0.5, size=20), 1, 100)
            ranges = get_ranges(X)
            return [individual(k) for k in Ks]

        estimator = DrugowitschGA1D(n_iter=250, init=init, random_state=random_state)
        estimator = estimator.fit(X, Y)
        # W, Lambda_1, a_tau, b_tau, V = get_params(params_elitist)
        # print(elitist)

