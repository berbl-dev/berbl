import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from .search.ga.drugowitsch import GADrugowitsch
from .search.operators.drugowitsch import DefaultToolbox
from .utils import randseed

search_methods = {"drugowitsch": GADrugowitsch}


class BERBL(BaseEstimator, RegressorMixin):
    """
    An implementation of the Bayesian Learning Classifier System.

    Based on (but also extending) the book ‘Design and Analysis of Learning
    Classifier Systems – A Probabilistic Approach’ by Jan Drugowitsch.

    Follows the [scikit-learn estimator
    pattern](https://scikit-learn.org/stable/developers/develop.html#estimators).
    """
    def __init__(self,
                 toolbox=DefaultToolbox(random_state=None),
                 search="drugowitsch",
                 searchparams: dict= {}):
        """
        Parameters
        ----------
        toolbox : object
            A DEAP `Toolbox` object that specifies all the operators required
            by the selected search algorithm (`search` parameter). By default,
            [`DefaultToolbox(random_state=None)`](search/operators/#berbl.search.operators.drugowitsch.DefaultToolbox).
        search : str
            Which search algorithm to use to perform model selection. Also see
            `toolbox` parameter. For now, only `'drugowitsch'` (denoting the
            simplistic genetic algorithm from [Drugowitsch's book](/)) is
            supported.
        kwargs : kwargs
            Passed through to the constructor of the search.
        """
        self.toolbox = toolbox
        self.search = search
        self.searchparams = searchparams

    def fit(self, X, y):
        """
        Fit BERBL to the data.

        Parameters
        ----------
        X : array of shape (N, DX)
            Training data.
        y : array of shape (N, Dy)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        `X` and `y` are assumed to be standardized. In the default
        configuration, matching functions are initialized uniformly at random in
        [-2, 2] which means that a large part of the inputs `X` should lie in
        this region (which should be the case for standardized uniformly
        distributed inputs as well as standardized normally distributed inputs).
        """
        # TODO Link to sklearn Sphinx-generated inventory

        # See SLEP010.
        X, y = self._validate_data(X, y, multi_output=True)

        # Let's remind the user to standardize data beforehand.
        for i, feature in enumerate(X.T):
            std = feature.std()
            mean = feature.mean()
            if not (np.isclose(mean, 0) and np.isclose(std, 1)):
                warnings.warn("Inputs and outputs should be "
                              f"standardized but training data input "
                              f"feature {i}'s "
                              f"mean is {mean} and "
                              f"std is {std}")

        std = y.std()
        mean = y.mean()
        if not (np.isclose(mean, 0) and np.isclose(std, 1)):
            warnings.warn("Inputs and outputs should be "
                          "standardized but training data output "
                          f"mean is {mean} and "
                          f"std is {std}")

        searchcls = search_methods[self.search]
        self.search_ = searchcls(self.toolbox,
                                 random_state=randseed(
                                     self.toolbox.random_state),
                                 **self.searchparams)

        self.search_ = self.search_.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict using BERBL.

        Parameters
        ----------
        X : array of shape (n, DX)
            Inputs to make predictions for.

        Returns
        -------
        y : array of shape (n, Dy)
            Predictions for the inputs (i.e. BERBL's predicitive distributions'
            means).
        """

        # See SLEP010.
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        return self.search_.predict(X)

    def predict_mean_var(self, X):
        """
        Get the mean and variance of BERBL's predictive density for each of the
        provided data points.

        “As the mixture of Student’s t distributions might be multimodal, there
        exists no clear definition for the 95% confidence intervals, but a
        mixture density-related study that deals with this problem can be found
        in [118].  Here, we take the variance as a sufficient indicator of the
        prediction’s confidence.”[^1]

        [^1]: Jan Drugowitsch. 2008. Design and Analysis of Learning Classifier
        Systems - A Probabilistic Approach.

        Parameters
        ----------
        X : array of shape (n, DX)
            Inputs to make predictions for.

        Returns
        -------
        y : array of shape (N, Dy)
        y_var : array of shape (N, Dy)
            Means and variances.
        """

        # See SLEP010.
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        return self.search_.predict_mean_var(X)

    def predicts(self, X):
        """
        Get each submodel's predictions, one by one, without mixing them.

        Parameters
        ----------
        X : array of shape (n, DX)
            Inputs to make predictions for.

        Returns
        -------
        array of shape (K, N, Dy)
            Mean output vectors of each submodel.
        """
        # See SLEP010.
        X = self._validate_data(X, reset=False)
        check_is_fitted(self)
        return self.search_.predicts(X)

    def predict_distribution(self, x):
        """
        The predictive distribution for the given input (i.e. the distribution
        over all outputs for that input).

        Parameters
        ----------
        x : array of shape (DX,)
            The input to compute the predictive distribution for.

        Returns
        -------
        pdf : callable
            A callable that, when given a possible output value `y` (an array
            of shape `(Dy, )`) returns the value of the predictive
            distribution at that point.
        """
        # TODO Properly validate input here
        check_is_fitted(self)
        return self.search_.predict_distribution(x)
