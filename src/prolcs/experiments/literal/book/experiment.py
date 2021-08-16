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


def experiment(gaparams, X, y, X_test, y_test_true, X_denoised, y_denoised, n_iter,
               seed, show, sample_size):
    mlflow.set_experiment("book.generated_function.literal")
    with mlflow.start_run() as run:
        mlflow.log_params(HParams().__dict__)
        mlflow.log_param("seed", seed)
        mlflow.log_param("train.size", sample_size)

        # scaler_X = StandardScaler()
        # scaler_y = StandardScaler()
        # X = scaler_X.fit_transform(X)
        # X_test = scaler_X.transform(X_test)
        # y = scaler_y.fit_transform(y)
        # y_test_true = scaler_y.transform(y_test_true)

        random_state = check_random_state(seed)

        estimator = GADrugowitsch(Toolbox(gaparams, random_state=random_state),
                                  n_iter=n_iter,
                                  random_state=random_state)
        estimator = estimator.fit(X, y)

        # make predictions for test data
        y_test, var = estimator.predict_mean_var(X_test)

        # get unmixed classifier predictions
        y_cls = estimator.predicts(X)

        # X = scaler_X.inverse_transform(X)
        # X_test = scaler_X.inverse_transform(X_test)
        # y = scaler_y.inverse_transform(y)
        # y_test = scaler_y.inverse_transform(y_test)
        # var = scaler_y.scale_**2 * var
        # y_cls = scaler_y.inverse_transform(y_cls)

        # two additional statistics to maybe better gauge solution performance
        mse = metrics.mean_squared_error(y_test_true, y_test)
        r2 = metrics.r2_score(y_test_true, y_test)
        log_("elitist.mse", mse, n_iter)
        log_("elitist.r2-score", r2, n_iter)

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

        plot_cls(X=X, y=y_cls, ax=ax)
        add_title(ax, estimator.size_[0], estimator.p_M_D_[0], mse, r2)
        save_plot(fig, seed)

        if show:
            plt.show()
