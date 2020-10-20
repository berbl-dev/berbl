# TODO Why is variance increasing that much on the right side?!
from prolcs.radialmatch1d import RadialMatch1D
from prolcs.radialmatch import RadialMatch
from prolcs import *
import numpy as np  # type: ignore

# The individual used in function generation.
ms = [
    # We invert the sigma parameters because our RadialMatch expects squared
    # inverse covariance matrices.
    RadialMatch(mu=np.array([0.2]), lambd_2=np.array([[(1 / 0.05)**2]])),
    RadialMatch(mu=np.array([0.5]), lambd_2=np.array([[(1 / 0.01)**2]])),
    RadialMatch(mu=np.array([0.8]), lambd_2=np.array([[(1 / 0.05)**2]])),
]

def generate(n: int = 300, rng: Generator = None):
    """
    [PDF p. 260]

    :param n: the number of samples to generate

    :returns: input and output matrices X (N × 1) and Y (N × 1)
    """
    if rng == None:
        rng = np.random.default_rng()

    X = np.random.random((n, 1))

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
            y += np.random.normal(loc=G[n][k] * (W[k] @ X_[n]),
                                  scale=Lambda_1[k])
        Y[n] = y

    # We return the non-augmented samples (because our algorithm augments them
    # itself).
    return X, Y


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    seed = 1

    X, Y = generate(1000, rng=np.random.default_rng(seed))
    plt.plot(X.reshape((-1)), Y.reshape((-1)), "r+")


    # [PDF p. 221, 3rd paragraph]
    # Drugowitsch samples individual sizes from a certain problem-dependent
    # Binomial distribution.
    def init(X, Y):
        Ks = np.clip(rng.binomial(8, 0.5, size=20), 1, 100)
        ranges = get_ranges(X)
        return [individual(ranges, k, rng=rng) for k in Ks]

    phi, elitist, p_M_D_elitist, params_elitist = ga(X, Y, iter=50, init=init)
    W, Lambda_1, a_tau, b_tau, V = get_params(params_elitist)
    print(elitist)

    X_test, Y_test_true = generate(1000, rng=np.random.default_rng(seed + 1))

    Y_test, var = np.zeros(Y_test_true.shape), np.zeros(Y_test_true.shape)
    for i in range(len(X_test)):
        Y_test[i], var[i] = predict1(X_test[i],
                                     elitist,
                                     W,
                                     Lambda_1,
                                     a_tau,
                                     b_tau,
                                     V,
                                     phi=phi)

    plt.errorbar(X_test.reshape((-1)),
                 Y_test.reshape((-1)),
                 var.reshape((-1)),
                 color="navy",
                 ecolor="gray",
                 fmt="v")

    plt.show()
