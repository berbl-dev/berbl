import time

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.ecial as sp  # type: ignore
import scipy.stats as st  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

random_state = check_random_state(1)


def gen_input():
    D_X = 30
    mean = random_state.random(size=D_X)
    eigvals = random_state.random(size=(D_X, )) * 10
    eigvecs = st.special_ortho_group.rvs(dim=D_X, random_state=random_state)
    Lambda = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)

    n_examples = 100
    X = random_state.random(size=(n_examples, D_X))
    return mean, Lambda, X


def normal_by_hand(mean, Lambda, X):
    det_Sigma = 1 / np.linalg.det(Lambda)
    X_mu = X - mean
    # The ``np.sum`` is a vectorization of ``(X_mu[n].T @ Lambda @
    # X_mu[n])`` for all ``n``.
    m = np.exp(-0.5 * np.sum((X_mu @ Lambda) * X_mu, axis=1))
    return m / (np.sqrt(2 * np.pi)**X.shape[1] * det_Sigma)


def normal_by_hand_inv(mean, Sigma, X):
    _, D_X = X.shape
    det_Sigma = np.linalg.det(Sigma)
    X_mu = X - mean
    # The ``np.sum`` is a vectorization of ``(X_mu[n].T @ Lambda @
    # X_mu[n])`` for all ``n``.
    m = np.exp(-0.5 * np.sum((X_mu @ np.linalg.inv(Sigma)) * X_mu, axis=1))
    return m / (np.sqrt(2 * np.pi)**X.shape[1] * det_Sigma)


def normal_scipy(mean, Sigma, X):
    return st.multivariate_normal(mean=mean, cov=Sigma).pdf(X)


def timethem(reps=100):
    functions = [normal_by_hand, normal_by_hand_inv, normal_scipy]
    times = pd.DataFrame(np.zeros((reps, 3)))
    times.columns = [f.__name__ for f in functions]
    for i in range(reps):
        inp = gen_input()
        for f in [normal_by_hand, normal_by_hand_inv, normal_scipy]:
            # In “fractional seconds”.
            start = time.process_time()
            for j in range(1000):
                f(*inp)
            end = time.process_time()
            times[f.__name__][i] = end - start

    return times
