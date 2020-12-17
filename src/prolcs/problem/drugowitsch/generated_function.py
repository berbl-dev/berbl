# TODO Why is variance increasing that much on the right side?!
# TODO Why do I only get a fitness of ~52 instead of Drugowitschs >100?
import os

import click
import joblib as jl
import numpy as np  # type: ignore
from prolcs.common import matching_matrix, phi_standard
from prolcs.drugowitsch import mixing
from prolcs.drugowitsch.ga1d import DrugowitschGA1D
from prolcs.drugowitsch.hyperparams import HParams
from prolcs.drugowitsch.state import State
from prolcs.logging import log_
from prolcs.problem.drugowitsch.init import make_init
from prolcs.radialmatch1d import RadialMatch1D
from prolcs.utils import add_intercept
from sklearn.utils import check_random_state  # type: ignore

from sklearn import metrics  # type: ignore

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
    X_ = add_intercept(X)
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
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=300)
def run_experiment(n_iter, seed, show, sample_size):
    # We import these packages here so the generate function can be used without
    # installing them.
    import matplotlib.pyplot as plt
    import mlflow

    mlflow.set_experiment("generated_function")
    with mlflow.start_run() as run:
        mlflow.log_params(HParams().__dict__)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)

        X, Y = generate(sample_size)

        init = make_init(8, 0.5, size=20, kmin=1, kmax=100)

        estimator = DrugowitschGA1D(n_iter=n_iter,
                                    init=init,
                                    random_state=seed)
        estimator = estimator.fit(X, Y)
        log_("random_state.random", State().random_state.random(), n_iter)
        log_("algorithm.oscillations.count", State().oscillation_count, n_iter)

        # store the model, you never know when you need it
        model_file = f"Model {seed}.joblib"
        jl.dump(estimator, model_file)
        mlflow.log_artifact(model_file)

        # generate test data
        X_test, Y_test_true = generate(1000, random_state=12345)

        # make predictions for test data
        Y_test, var = np.zeros(Y_test_true.shape), np.zeros(Y_test_true.shape)
        for i in range(len(X_test)):
            Y_test[i], var[i] = estimator.predict1_elitist_mean_var(X_test[i])

        mse = metrics.mean_squared_error(Y_test_true, Y_test)
        r2 = metrics.r2_score(Y_test_true, Y_test)
        log_("elitist.mse", mse, n_iter)
        log_("elitist.r2-score", r2, n_iter)

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
        elitist = estimator.elitist_
        W = elitist.W
        X_test_ = np.hstack([np.ones((len(X_test), 1)), X_test])
        # save approximation so we don't need to run it over and over again
        for k in range(len(W)):
            ax.plot(X_test.ravel(),
                    np.sum(W[k] * X_test_, axis=1),
                    c="C" + str(k),
                    zorder=10)

        ax.set(
            title=
            f"K = {len(W)}, p(M|D) = {elitist.p_M_D:.2}, mse = {mse:.2}, r2 = {r2:.2}"
        )

        # store the figure (e.g. so we can run headless)
        fig_folder = "latest-final-approximations"
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
        fig_file = f"{fig_folder}/Final approximation {seed}.pdf"
        fig.savefig(fig_file)
        mlflow.log_artifact(fig_file)

        if show:
            plt.show()


if __name__ == "__main__":
    run_experiment()
