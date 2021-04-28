import numpy as np  # type: ignore
from prolcs.linear.mixture import Mixture
from prolcs.match.radial1d import RadialMatch1D
from prolcs.match.radial import RadialMatch
from prolcs.logging import log_
from ...utils import get_ranges
from sklearn.utils import check_random_state  # type: ignore


def uniform_interval(low, high):
    """
    A uniform distribution over the number of classifiers in a solution.

    The arguments get passed through to ``random_state.randint(low, high)``.

    Parameters
    ----------
    low : int
        Minimum number of classifiers.
    high : int
        Maximum number of classifiers.
    """
    def f(random_state):
        random_state = check_random_state(random_state)
        return random_state.randint(low, high)

    return f


class RandomSearch:
    """
    Generates random solutions (i.e. random sets of match functions), trains the
    mixture model for each and keeps the best of those models.

    Note that different matching functions are used for one- and
    multi-dimensional input.
    """
    def __init__(self, n_iter=250, n_cls=5, random_state=None, **kwargs):
        """
        Parameters
        ----------
        n_iter : int, optional
            The number of random solutions to generate.
        n_cls : int or prob distribution over ints expecting a random_state
            The number of match functions/classifiers each solution should have.
            Either fixed or a probability distribution over this variable.
        random_state
            The usual `random_state` argument to probabilistic modules.
        **kwargs
            Any other keyword parameters are passed through to `Mixture`,
            `Classifier` and `Mixing`.
        """
        self.n_iter = n_iter
        self.n_cls = n_cls
        self.random_state = random_state
        self.__kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray):
        random_state = check_random_state(self.random_state)

        if X.shape[1] <= 2:
            match_class = RadialMatch1D
        else:
            match_class = RadialMatch

        self.mixture_ = None

        for iter in range(self.n_iter):
            if type(self.n_cls) is int:
                n_cls_ = self.n_cls
            else:
                n_cls_ = self.n_cls(random_state)

            # Highest possible seed is `2**32 - 1` for NumPy legacy generators.
            seed = random_state.randint(2**32 - 1)
            ranges = get_ranges(X)
            mixture = Mixture(ranges=ranges,
                              match_class=match_class,
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
