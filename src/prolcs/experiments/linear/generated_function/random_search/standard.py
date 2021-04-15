import os

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np  # type: ignore
from prolcs.experiments.drugowitsch.generated_function import generate
from prolcs.linear.mixture import Mixture
from prolcs.logging import log_
from prolcs.radialmatch1d import RadialMatch1D
from prolcs.utils import add_bias
from sklearn import metrics  # type: ignore
from prolcs.linear.search.random import RandomSearch

np.seterr(all="warn")


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=300)
def run_experiment(n_iter, seed, show, sample_size):
    mlflow.set_experiment("linear-generated_function")
    with mlflow.start_run() as run:
        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)

        random_state = np.random.RandomState(seed)

        X, y = generate(sample_size)
        X_augmented = add_bias(X)

        # generate denoised data as well (for visual reference)
        X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
        _, y_denoised = generate(1000, noise=False, X=X_denoised)

        K = 5
        model = RandomSearch(n_cls=K,
                             n_iter=250,
                             fit_mixing="bouchard",
                             random_state=seed)
        model.fit(X_augmented, y)

        # generate test data
        X_test, y_test_true = generate(1000, random_state=12345)
        X_test_augmented = add_bias(X_test)

        # make predictions for test data
        y_test, var = model.predict_mean_var(X_test_augmented)

        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true, y_test)
        log_("mse", mse, n_iter)
        log_("r2-score", r2, n_iter)

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

        if True:  # plot matching and classifiers
            for m in model.mixture_.matchs_:
                m.plot(ax, color="grey")

            # plot classifier models
            models = model.predicts(X_test_augmented)
            for y in models:
                ax.plot(X_test.ravel(),
                        y,
                        c="grey",
                        linestyle="-",
                        linewidth=0.7,
                        alpha=0.7,
                        zorder=10)

        # add metadata to plot for ease of use
        ax.set(title=f"K = {K}, p(M|D) = {model.mixture_.p_M_D_:.2}")

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