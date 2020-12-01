# TODO It sometimes seems to hang somewhere (i.e. not finishing the generation)
# TODO Why is variance increasing that much on the right side?!
import numpy as np  # type: ignore
from prolcs.common import matching_matrix, phi_standard
from prolcs.drugowitsch import mixing
from prolcs.drugowitsch.ga1d import DrugowitschGA1D
from prolcs.drugowitsch.model import Model
from prolcs.radialmatch1d import RadialMatch1D
from prolcs.utils import get_ranges
from sklearn import preprocessing  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
import joblib as jl
import click

# The individual used in function generation.
ms = [
    RadialMatch1D(mu=0.2, sigma_2=0.05, ranges=(0, 1)),
    RadialMatch1D(mu=0.5, sigma_2=0.01, ranges=(0, 1)),
    RadialMatch1D(mu=0.8, sigma_2=0.05, ranges=(0, 1)),
]

np.seterr(all="warn")


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
            y += random_state.normal(loc=G[n][k] * (W[k] @ X_[n]),
                                     scale=Lambda_1[k])
        Y[n] = y

    # We return the non-augmented samples (because our algorithm augments them
    # itself).
    return X, Y


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--show/--no-show", type=bool, default=False)
def run_experiment(n_iter, seed, show):
    import matplotlib.pyplot as plt

    import mlflow

    LOGGING = "mlflow"

    mlflow.set_experiment("generated_function")
    with mlflow.start_run() as run:
        mlflow.log_param("seed", seed)
        random_state = check_random_state(seed)

        X, Y = generate()

        # TODO Is this what Drugowitsch means by “standardized by a linear
        # transformation”?
        x_scaler = preprocessing.StandardScaler().fit(X)
        X = x_scaler.transform(X)

        y_scaler = preprocessing.StandardScaler().fit(Y)
        Y = y_scaler.transform(Y)

        ranges = (0, 1)

        def individual(k: int):
            """
            Individuals are simply lists of matching functions (the length of
            the list is the number of classifiers, the matching functions
            specify their localization).
            """
            return Model([
                RadialMatch1D.random(ranges, random_state=random_state)
                for i in range(k)
            ],
                         phi=phi_standard)

        # [PDF p. 221, 3rd paragraph]
        # Drugowitsch samples individual sizes from a certain
        # problem-dependent Binomial distribution.
        def init(X, Y):
            Ks = np.clip(random_state.binomial(8, 0.5, size=20), 1, 100)
            ranges = get_ranges(X)
            return [individual(k) for k in Ks]

        estimator = DrugowitschGA1D(n_iter=n_iter,
                                    init=init,
                                    random_state=random_state)
        estimator = estimator.fit(X, Y)

        # store the model, you never know when you need it
        model_file = f"Model {seed}.joblib"
        jl.dump(estimator, model_file)
        mlflow.log_artifact(model_file)

        # generate test data
        X_test, Y_test_true = generate(1000, random_state=12345)
        X_test = x_scaler.transform(X_test)
        Y_test_true = y_scaler.transform(Y_test_true)

        # make predictions for test data
        Y_test, var = np.zeros(Y_test_true.shape), np.zeros(Y_test_true.shape)
        for i in range(len(X_test)):
            Y_test[i], var[i] = estimator.predict1_elitist_mean_var(X_test[i])

        fig, ax = plt.subplots()

        # plot input data
        ax.plot(X.ravel(), Y.ravel(), "r+")

        # plot test data
        ax.errorbar(X_test.ravel(),
                    Y_test.ravel(),
                    var.ravel(),
                    color="navy",
                    ecolor="gray",
                    fmt="v")

        # plot elitist's classifiers
        W = estimator.elitist_.W
        X_test_ = np.hstack([np.ones((len(X_test), 1)), X_test])
        # save approximation so we don't need to run it over and over again
        for k in range(len(W)):
            ax.plot(X_test.ravel(),
                    np.sum(W[k] * X_test_, axis=1),
                    c="C" + str(k),
                    zorder=10)

        ax.set(title=f"K = {len(W)}")

        # store the figure (e.g. so we can run headless)
        fig_file = f"Final approximation {seed}.pdf"
        fig.savefig(fig_file)
        mlflow.log_artifact(fig_file)

        if show:
            plt.show()


if __name__ == "__main__":
    run_experiment()
