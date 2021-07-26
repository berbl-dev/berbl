# TODO Why do I only get a fitness of ~52 instead of Drugowitschs >100?
import os

import click
import joblib as jl
import matplotlib.pyplot as plt
import mlflow
import numpy as np  # type: ignore
from deap import creator, tools
from prolcs.common import initRepeat_binom
from prolcs.literal.hyperparams import HParams
from prolcs.literal.model import Model
from prolcs.literal.state import State
from prolcs.logging import log_
from prolcs.match.radial1d_drugowitsch import RadialMatch1D
from prolcs.search.ga.drugowitsch import GADrugowitsch
from prolcs.search.operators.drugowitsch import Toolbox
from prolcs.tasks.book.generated_function import generate
from sklearn import metrics  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

np.seterr(all="warn")


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=300)
def run_experiment(n_iter, seed, show, sample_size):

    mlflow.set_experiment("book.generated_function.literal")
    with mlflow.start_run() as run:
        mlflow.log_params(HParams().__dict__)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)

        X, y = generate(sample_size)

        # generate denoised data as well (only for visual reference)
        X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
        _, y_denoised = generate(1000, noise=False, X=X_denoised)

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
            genotype.phenotype = Model(matchs=genotype,
                                       random_state=random_state).fit(X, y)
            return (genotype.phenotype.p_M_D_, )

        toolbox.register("evaluate", _evaluate)

        estimator = GADrugowitsch(toolbox,
                                  n_iter=n_iter,
                                  random_state=random_state)
        estimator = estimator.fit(X, y)

        log_("random_state.random", State().random_state.random(), n_iter)

        # store the model, you never know when you need it
        model_file = f"models/Model {seed}.joblib"
        jl.dump(estimator.frozen(), model_file)
        mlflow.log_artifact(model_file)

        # generate test data
        X_test, y_test_true = generate(1000, random_state=12345)

        # make predictions for test data
        y_test, var = estimator.predict_mean_var(X_test)

        # two additional statistics to better gauge solution performance
        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true, y_test)
        log_("elitist.mse", mse, n_iter)
        log_("elitist.r2-score", r2, n_iter)

        fig, ax = plt.subplots()

        # plot input data
        ax.plot(X.ravel(), y.ravel(), "r+")

        # plot denoised input data for visual reference
        ax.plot(X_denoised.ravel(), y_denoised.ravel(), "k--")

        # plot test data
        X_test_ = X_test.ravel()
        perm = np.argsort(X_test_)
        X_test_ = X_test_[perm]
        y_test_ = y_test.ravel()[perm]
        var_ = var.ravel()[perm]
        ax.plot(X_test_, y_test_, "b-")
        ax.plot(X_test_, y_test_ - var_, "b--", linewidth=0.5)
        ax.plot(X_test_, y_test_ + var_, "b--", linewidth=0.5)
        ax.fill_between(X_test_, y_test_ - var_, y_test_ + var_, alpha=0.2)

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
