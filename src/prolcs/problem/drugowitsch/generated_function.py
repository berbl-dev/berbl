from prolcs.radialmatch1d import RadialMatch1D
from prolcs.radialmatch import RadialMatch
from prolcs import *
import numpy as np  # type: ignore


def generate(n: int = 300, rng: Generator = None):
    """
    [PDF p. 260]

    :param n: the number of samples to generate

    :returns: input and output matrices X (N × 1) and Y (N × 1)
    """
    if rng == None:
        rng = np.random.default_rng()

    X = np.random.random((n, 1))
    ms = [
        # We invert the sigma parameters because our RadialMatch expects squared
        # inverse covariance matrices.
        RadialMatch(mu=np.array([0.2]), lambd_2=np.array([[(1 / 0.05)**2]])),
        RadialMatch(mu=np.array([0.5]), lambd_2=np.array([[(1 / 0.01)**2]])),
        RadialMatch(mu=np.array([0.8]), lambd_2=np.array([[(1 / 0.05)**2]])),
    ]
    M = matching_matrix(ms, X)
    # After matching, prepend 1 to each sample
    X_ = np.vstack([np.ones(n), X.reshape((n))]).T
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

    Phi = phi_standard(X_)
    G = mixing(M, Phi, V)

    Y = np.zeros(X.shape)
    for n in range(len(X)):
        y = 0
        for k in range(len(ms)):
            # sample the three classifiers
            y += np.random.normal(loc=G[n][k] * (W[k] @ X_[n]),
                                  scale=Lambda_1[k])
        Y[n] = y

    return X, Y


if __name__ == "__main__":
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    X, Y = generate(1000)
    data = pd.DataFrame(np.hstack([X, Y]), columns=["X", "Y"])

    sns.relplot(x="X", y="Y", data=data)
    plt.show()
