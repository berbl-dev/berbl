# TODO Why do I only get a fitness of ~52 instead of Drugowitschs >100?
import os

import click
import joblib as jl
import numpy as np  # type: ignore
from deap import base, creator, tools
from prolcs.common import matching_matrix, check_phi, initRepeat_binom
from prolcs.literal import mixing
from prolcs.literal.mixture import Mixture
from prolcs.literal.hyperparams import HParams
from prolcs.literal.state import State
from prolcs.logging import log_
from prolcs.match.radial1d_drugowitsch import RadialMatch1D
from prolcs.search.ga.drugowitsch import GADrugowitsch
from prolcs.search.operators.drugowitsch import Toolbox
from prolcs.utils import add_bias
from sklearn import metrics  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

# The individual used in function generation.
ms = [
    RadialMatch1D(mu=0.2, sigma_2=0.05, has_bias=False),
    RadialMatch1D(mu=0.5, sigma_2=0.01, has_bias=False),
    RadialMatch1D(mu=0.8, sigma_2=0.05, has_bias=False),
]

np.seterr(all="warn")


def generate(n: int = 300,
             noise=True,
             X=None,
             random_state: np.random.RandomState = 0):
    """
    [PDF p. 260]

    :param n: The number of samples to generate. Supplying ``X`` overrides this.
    :param noise: Whether to generate noisy data (the default) or not. The
        latter may be useful for visualization purposes.
    :param X: Sample the function at these exact input points (instead of
        generating ``n`` input points randomly).

    :returns: input and output matrices X (N × 1) and Y (N × 1)
    """
    random_state = check_random_state(random_state)

    if X is None:
        X = random_state.random((n, 1))

    M = matching_matrix(ms, X)
    Phi = check_phi(None, X)

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
    X_ = add_bias(X)
    Y = np.zeros(X.shape)
    for n in range(len(X)):
        y = 0
        for k in range(len(ms)):
            # sample the three classifiers
            if noise:
                y += random_state.normal(loc=G[n][k] * (W[k] @ X_[n]),
                                         scale=Lambda_1[k])
            else:
                y += G[n][k] * (W[k] @ X_[n])
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

    mlflow.set_experiment("literal.generated_function")
    with mlflow.start_run() as run:
        mlflow.log_params(HParams().__dict__)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)

        X, y = generate(sample_size)

        # generate denoised data as well (for visual reference)
        X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
        _, Y_denoised = generate(1000, noise=False, X=X_denoised)

        # TODO Normalize X

        toolbox = Toolbox()

        random_state = check_random_state(seed)

        toolbox.register("gene",
                         RadialMatch1D.random,
                         random_state=random_state)

        toolbox.register("genotype",
                         initRepeat_binom,
                         creator.Genotype,
                         toolbox.gene,
                         n=8,
                         p=0.5,
                         random_state=random_state)

        toolbox.register("population", tools.initRepeat, list,
                         toolbox.genotype)

        def _evaluate(genotype, X, y):
            phenotype = Mixture(matchs=genotype, random_state=random_state)
            phenotype.fit(X, y)
            genotype.phenotype = phenotype
            return (phenotype.p_M_D_, )

        toolbox.register("evaluate", _evaluate)

        estimator = GADrugowitsch(toolbox,
                                  n_iter=n_iter,
                                  random_state=random_state)
        estimator = estimator.fit(X, y)

        log_("random_state.random", State().random_state.random(), n_iter)
        log_("algorithm.oscillations.count", State().oscillation_count, n_iter)

        # TODO Store the model, you never know when you need it
        # model_file = f"models/Model {seed}.joblib"
        # jl.dump(estimator, model_file)
        # mlflow.log_artifact(model_file)

        # generate test data
        X_test, Y_test_true = generate(1000, random_state=12345)

        # make predictions for test data
        Y_test, var = estimator.predict_mean_var(X_test)

        mse = metrics.mean_squared_error(Y_test_true, Y_test)
        r2 = metrics.r2_score(Y_test_true, Y_test)
        log_("elitist.mse", mse, n_iter)
        log_("elitist.r2-score", r2, n_iter)

        fig, ax = plt.subplots()

        # plot input data
        ax.plot(X.ravel(), y.ravel(), "r+")

        # plot denoised input data for visual reference
        ax.plot(X_denoised.ravel(), Y_denoised.ravel(), "k--")

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
        elitist = estimator.elitist_[0].phenotype
        W = elitist.W_
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
            f"K = {len(W)}, p(M|D) = {elitist.p_M_D_:.2}, mse = {mse:.2}, r2 = {r2:.2}"
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
