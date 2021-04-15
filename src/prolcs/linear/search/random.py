import numpy as np  # type: ignore
from prolcs.linear.mixture import Mixture
from prolcs.logging import log_
from sklearn.utils import check_random_state  # type: ignore


class RandomSearch:
    def __init__(self, n_iter=250, random_state=None, **kwargs):
        self.n_iter = n_iter
        self.random_state = random_state
        self.__kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray):
        random_state = check_random_state(self.random_state)

        self.mixture_ = None

        for iter in range(self.n_iter):
            # Highest possible seed is `2**32 - 1` for NumPy legacy generators.
            seed = random_state.randint(2**32 - 1)
            ranges = (X.min(), X.max())
            mixture = Mixture(ranges=ranges,
                              random_state=seed,
                              **self.__kwargs)
            mixture.fit(X, y)

            if self.mixture_ is None or mixture.p_M_D_ > self.mixture_.p_M_D_:
                self.mixture_ = mixture

            log_("p_M_D", self.mixture_.p_M_D_, iter)

            print(f"Trained random mixture {iter}, "
                  f"current best has ln p(M | D) = {self.mixture_.p_M_D_:.2}")

        return self

    def predict(self, X):
        return self.mixture_.predict(X)

    def predict_mean_var(self, X):
        return self.mixture_.predict_mean_var(X)

    def predicts(self, X):
        return self.mixture_.predicts(X)
