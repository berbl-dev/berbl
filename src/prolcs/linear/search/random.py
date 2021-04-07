import numpy as np  # type: ignore
from prolcs.linear.mixture import Mixture
from prolcs.logging import log_


class RandomSearch:
    def __init__(self, n_iter=250, **kwargs):
        self.n_iter = n_iter
        self.__kwargs = kwargs
        self.mixture = None

    def fit(self, X: np.ndarray, y: np.ndarray, random_state=None):
        self.mixture = None
        for iter in range(self.n_iter):
            ranges = (X.min(), X.max())
            mixture = Mixture(ranges=ranges, **self.__kwargs)
            mixture.fit(X, y, random_state=random_state)

            if self.mixture is None or mixture.p_M_D > self.mixture.p_M_D:
                self.mixture = mixture

            log_("p_M_D", self.mixture.p_M_D, iter)

            print(f"Trained random mixture {iter}, "
                  f"current best has ln p(M | D) = {self.mixture.p_M_D:.2}")

        self.mixture = mixture

    def predict(self, X):
        return self.mixture.predict(X)

    def predict_mean_var(self, X):
        return self.mixture.predict_mean_var(X)

    def predicts(self, X):
        return self.mixture.predicts(X)
