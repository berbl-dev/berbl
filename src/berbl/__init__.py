from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from .utils import randseed
from .search.ga.drugowitsch import GADrugowitsch
from .search.operators.drugowitsch import DefaultToolbox

search_methods = {"drugowitsch": GADrugowitsch}


class BERBL(BaseEstimator, RegressorMixin):
    """
    An implementation of a Bayesian evolutionary rule-based learning system.
    Based on (but also extending) the book ‘Design and Analysis of Learning
    Classifier Systems – A Probabilistic Approach’ by Jan Drugowitsch.
    """
    def __init__(self,
                 toolbox=DefaultToolbox(random_state=None),
                 search="drugowitsch",
                 n_iter=100):
        """
        Parameters
        ----------
        toolbox : Toolbox object
            A DEAP ``Toolbox`` object that specifies all the operators required
            by the selected search algorithm (``search`` parameter).
        search : str
            Which search algorithm to use to perform model selection. Also see
            ``toolbox`` parameter.
        n_iter : int
            Number of iterations to run the search.
        """
        self.toolbox = toolbox
        self.search = search
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Notes
        -----
        `X` and `y` are assumed to be standardized. In the default
        configuration, matching functions are initialized in [-5, 5] which means
        that a large part of the inputs `X` should lie in this region (which is
        the case for standardized uniformly distributed inputs as well as
        standardized normally distributed inputs).
        """

        # See SLEP010.
        X, y = self._validate_data(X, y, multi_output=True)

        searchcls = search_methods[self.search]
        self.search_ = searchcls(self.toolbox,
                                 n_iter=self.n_iter,
                                 random_state=randseed(
                                     self.toolbox.random_state))

        self.search_ = self.search_.fit(X, y)

        return self

    def predict(self, X):
        # See SLEP010.
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        return self.search_.predict(X)

    def predict_mean_var(self, X):
        # See SLEP010.
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        return self.search_.predict_mean_var(X)

    def predicts(self, X):
        # See SLEP010.
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        return self.search_.predicts(X)

    def predict_distribution(self, x):
        """
        The distribution over all outputs for the given input.

        Parameters
        ----------
        x : array of shape (DX,)

        Returns
        -------
        pdf : callable
            A callable that, when given a possible output value ``y`` (an array
            of shape ``(Dy, )``) returns the value of the predictive
            distribution at that point.
        """
        # TODO Properly validate input here
        check_is_fitted(self)
        return self.search_.predict_distribution(x)
