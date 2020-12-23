import os

import click
import joblib as jl
import numpy as np  # type: ignore
from prolcs.drugowitsch.ga1d import DrugowitschGA1D
from prolcs.drugowitsch.hyperparams import HParams
from prolcs.drugowitsch.state import State
from prolcs.logging import log_
from prolcs.problem.drugowitsch.init import make_init
from sklearn import metrics  # type: ignore
from sklearn.utils import check_random_state  # type: ignore


def f(x, noise_vars=(0.6, 0.1), random_state: np.random.RandomState = 0):
    random_state = check_random_state(random_state)
    return np.where(
        x < 0, -1 - 2 * x
        + random_state.normal(0, np.sqrt(noise_vars[0]), size=x.shape), -1
        + 2 * x + random_state.normal(0, np.sqrt(noise_vars[1]), size=x.shape))


def generate(n: int = 200, random_state: np.random.RandomState = 0):
    """
    [PDF p. 262]

    :param n: the number of samples to generate

    :returns: input and output matrices X (N × 1) and Y (N × 1)
    """
    random_state = check_random_state(random_state)

    X = random_state.uniform(low=-1, high=1, size=(n, 1))
    Y = f(X, random_state=random_state)

    return X, Y


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=200)
def run_experiment(n_iter, seed, show, sample_size):
    # We import these packages here so the generate function can be used without
    # installing them.
    import matplotlib.pyplot as plt
    import mlflow

    mlflow.set_experiment("variable_noise")
    with mlflow.start_run() as run:
        mlflow.log_params(HParams().__dict__)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)

        X, Y = generate(sample_size)

        # De-noised data for visual reference
        X_denoised = np.arange(-1, 1, 0.01)
        Y_denoised = f(X_denoised, noise_vars=(0, 0))

        # TODO Drugowitsch uses soft-interval matching here, I haven't that
        # implemented as of now
        init = make_init(n=8, p=0.5, size=20, kmin=1, kmax=100)

        estimator = DrugowitschGA1D(n_iter=n_iter,
                                    init=init,
                                    random_state=seed)
        estimator = estimator.fit(X, Y)
        log_("random_state.random", State().random_state.random(), n_iter)

        X_test, Y_test_true = generate(1000, random_state=54321)

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

        # plot denoised input data for visual reference
        ax.plot(X_denoised.reshape((-1)), Y_denoised.reshape((-1)), "k--")

        # plot test data
        X_test_ = X_test.ravel()
        perm = np.argsort(X_test_)
        X_test_ = X_test_[perm]
        Y_test_ = Y_test.ravel()[perm]
        var_ = var.ravel()[perm]
        ax.plot(X_test_, Y_test_, "b-")
        ax.plot(X_test_, Y_test_ - var_, "b--", linewidth=0.5)
        ax.plot(X_test_, Y_test_ + var_, "b--", linewidth=0.5)
        ax.fill_between(X_test_, Y_test_ - var_, Y_test_ + var_, alpha=0.2)

        # plot elitist's classifiers
        elitist = estimator.elitist_
        W = elitist.W
        X_test_ = np.hstack([np.ones((len(X_test), 1)), X_test])
        for k in range(len(W)):
            ax.plot(X_test.ravel(),
                    np.sum(W[k] * X_test_, axis=1),
                    c="grey",
                    linestyle="-",
                    linewidth=0.5,
                    alpha=0.7,
                    zorder=10)

        # add metadata to plot for ease of use
        ax.set(
            title=
            f"K = {len(W)}, p(M|D) = {elitist.p_M_D:.2}, mse = {mse:.2}, r2 = {r2:.2}"
        )

        # store the figure (e.g. so we can run headless)
        fig_folder = "latest-final-approximations"
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
        fig_file = f"{fig_folder}/Final approximation {seed}.pdf"
        print(f"Storing final approximation figure in {fig_file}")
        fig.savefig(fig_file)
        mlflow.log_artifact(fig_file)

        if show:
            plt.show()


if __name__ == "__main__":
    run_experiment()
