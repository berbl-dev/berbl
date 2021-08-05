import os

import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import numpy as np  # type: ignore


def plot_prediction(X,
                    y,
                    X_test,
                    y_test,
                    var,
                    X_denoised=None,
                    y_denoised=None):
    fig, ax = plt.subplots()

    # plot input data
    ax.plot(X.ravel(), y.ravel(), "r+")

    if X_denoised is not None and y_denoised is not None:
        # plot denoised input data for visual reference
        ax.plot(X_denoised.ravel(), y_denoised.ravel(), "k--")

    # plot test data
    X_test_ = X_test.ravel()
    perm = np.argsort(X_test_)
    X_test_ = X_test_[perm]
    y_test_ = y_test.ravel()[perm]
    var_ = var.ravel()[perm]
    std = np.sqrt(var_)
    ax.plot(X_test_, y_test_, "b-")
    ax.plot(X_test_, y_test_ - std, "b--", linewidth=0.5)
    ax.plot(X_test_, y_test_ + std, "b--", linewidth=0.5)
    ax.fill_between(X_test_, y_test_ - std, y_test_ + std, alpha=0.2)

    return fig, ax


def plot_cls(X, y, ax=None):
    """
    Parameters
    ----------
    X : array of shape (N, 1)
        Points for which the classifiers made predictions.
    y : array of shape (K, N, Dy)
        Predictions of the ``K`` classifiers.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    for k in range(len(y)):
        ax.plot(X.ravel(),
                y[k],
                c="grey",
                linestyle="-",
                linewidth=0.5,
                alpha=0.7,
                zorder=10)

    return fig, ax


def add_title(ax, K, p_M_D, mse, r2):
    # add metadata to plot for ease of use
    ax.set(title=(f"K = {K}, "
                  f"p(M|D) = {(p_M_D):.2}, "
                  f"mse = {mse:.2}, "
                  f"r2 = {r2:.2}"))


def save_plot(fig, seed):
    # store the figure (e.g. so we can run headless)
    fig_folder = "latest-final-approximations"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig_file = f"{fig_folder}/Final approximation {seed}.pdf"
    print(f"Storing final approximation figure in {fig_file}")
    fig.savefig(fig_file)
    mlflow.log_artifact(fig_file)
