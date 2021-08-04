# TODO Why do I only get a fitness of ~52 instead of Drugowitschs >100?

import click
import joblib as jl
import mlflow
import numpy as np  # type: ignore
from prolcs.literal.hyperparams import HParams
from prolcs.logging import log_
from prolcs.search.ga.drugowitsch import GADrugowitsch
from prolcs.tasks.book.generated_function import generate
from sklearn.preprocessing import StandardScaler
from sklearn import metrics  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from .plot import *
from .toolbox import Toolbox

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
        X_test, y_test_true = generate(1000, random_state=12345)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        X_test = scaler_X.transform(X_test)
        y = scaler_y.fit_transform(y)
        y_test_true = scaler_y.transform(y_test_true)

        random_state = check_random_state(seed)

        estimator = GADrugowitsch(Toolbox(random_state=random_state),
                                  n_iter=n_iter,
                                  random_state=random_state)
        estimator = estimator.fit(X, y)

        # make predictions for test data
        y_test, var = estimator.predict_mean_var(X_test)

        # two additional statistics to maybe better gauge solution performance
        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true, y_test)
        log_("elitist.mse", mse, n_iter)
        log_("elitist.r2-score", r2, n_iter)

        X = scaler_X.inverse_transform(X)
        X_test = scaler_X.inverse_transform(X_test)
        y = scaler_y.inverse_transform(y)
        y_test = scaler_y.inverse_transform(y_test)
        var = scaler_y.scale_**2 * var

        # generate equidistant, denoised data as well (only for visual
        # reference); note that this doesn't need to be transformed back and
        # forth
        X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
        _, y_denoised = generate(1000, noise=False, X=X_denoised)

        # store the model, you never know when you need it
        model_file = f"models/Model {seed}.joblib"
        jl.dump(estimator.frozen(), model_file)
        mlflow.log_artifact(model_file)

        fig, ax = plot_prediction(X=X,
                                  y=y,
                                  X_test=X_test,
                                  y_test=y_test,
                                  var=var,
                                  X_denoised=X_denoised,
                                  y_denoised=y_denoised)

        plot_cls(estimator, X_test, ax=ax)
        save_plot(fig, seed)
        add_title(ax, estimator.size_[0], estimator.p_M_D_[0], mse, r2)

        if show:
            plt.show()


if __name__ == "__main__":
    run_experiment()
