from typing import *

import numpy as np  # type: ignore
from sklearn.utils import check_consistent_length  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from ..utils import add_bias
from ..common import check_phi, matching_matrix
from . import mixing, model_probability


class Model():
    def __init__(self,
                 matchs: List,
                 add_bias=True,
                 phi=None,
                 random_state=None):
        """
        A model based on mixing linear classifiers using the given model
        structure.

        Parameters
        ----------
        matchs
            A list of matching functions (i.e. objects implementing a ``match``
            attribute) defining the structure of this mixture. If given,
            ``n_cls`` and ``match_class`` are not used to generate classifiers
            randomly.
        add_bias : bool
            Whether to add an all-ones bias column to the input data.
        phi
            mixing feature extractor (N × D_X → N × D_V); if ``None`` uses the
            default LCS mixing feature matrix based on ``phi(x) = 1``
        random_state
            See ``n_cls``.
        """
        self.matchs = matchs
        self.add_bias = add_bias
        self.phi = phi
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        check_consistent_length(X, y)

        if self.add_bias:
            X = add_bias(X)

        random_state = check_random_state(self.random_state)

        self.K_ = len(self.matchs)
        _, self.D_X_ = X.shape
        _, self.D_y_ = y.shape

        Phi = check_phi(self.phi, X)

        self.metrics_, self.params_ = model_probability(matchs=self.matchs,
                                                        X=X,
                                                        Y=y,
                                                        Phi=Phi)

        self.L_q_ = self.metrics_["L_q"]
        self.ln_p_M_ = self.metrics_["ln_p_M"]
        self.L_k_q_ = self.metrics_["L_k_q"]
        self.L_M_q_ = self.metrics_["L_M_q"]

        self.W_ = self.params_["W"]
        self.Lambda_1_ = self.params_["Lambda_1"]
        self.a_tau_ = self.params_["a_tau"]
        self.b_tau_ = self.params_["b_tau"]
        self.a_alpha_ = self.params_["a_alpha"]
        self.b_alpha_ = self.params_["b_alpha"]
        self.V_ = self.params_["V"]
        self.Lambda_V_1_ = self.params_["Lambda_V_1"]
        self.a_beta_ = self.params_["a_beta"]
        self.b_beta_ = self.params_["b_beta"]

        self.K_ = len(self.W_)

        return self

    def predict(self, X: np.ndarray):

        return self.predict_mean_var(X)[0]

    def predict_mean_var(self, X: np.ndarray):
        """
        [PDF p. 224]

        The mean and variance of the predictive density described by this model
        for each of the provided data points.

        “As the mixture of Student’s t distributions might be multimodal, there
        exists no clear definition for the 95% confidence intervals, but a
        mixture density-related study that deals with this problem can be found
        in [118].  Here, we take the variance as a sufficient indicator of the
        prediction’s confidence.” [PDF p. 224]

        :param X: input vector (N × D_X)

        :returns: mean output vector (N × D_y), variance of output (N × D_y)
        """
        check_is_fitted(self)

        N, _ = X.shape
        D_y, D_X = self.W_[0].shape

        Phi = check_phi(self.phi, X)

        if self.add_bias:
            X = add_bias(X)

        # Collect the independent predictions and variances of each classifier.
        # We use the definitions of those that do neither perform input checking
        # nor bias adding to save some time.
        y = self._predicts(X)
        y_var = self._predict_vars(X)

        # Next, mix the predictions.
        M = matching_matrix(self.matchs, X)
        G_ = mixing(M, Phi, self.V_)

        # For each classifier's prediction, we weigh every dimension of the
        # output vector by the same amount, thus we simply repeat the G values
        # over D_y. TODO Am I sure about this?
        G = G_.reshape(y.shape).repeat(D_y, axis=2)  # K × N × D_y

        y = np.sum(G * y, axis=0)

        y_var = np.sum(G * (y_var + y**2), axis=0) - y**2

        # TODO Re-check this for correctness (should(?) probably be the same as
        # the following loop but is not?)
        # var = np.zeros((N, D_y))
        # for n in range(N):
        #     x_ = X[n]
        #     g = G_.T[n]
        #     for j in range(D_y):
        #         for k in range(self.K_):
        #             cl = self.classifiers[k]
        #             var[n][j] += g[k] * (2 * cl.b_tau / (cl.a_tau - 1) *
        #                                  (1 + x_ @ cl.Lambda_1 @ x_) +
        #                                  (cl.W[j] @ x_)**2)
        #         var[n][j] -= y[j]**2
        # assert np.all(np.isclose(
        #     y_var, var)), (y_var - var)[np.where(~(np.isclose(y_var - var,
        #     0)))]

        return y, y_var

    def predicts(self, X):
        """
        Returns this model's classifiers' predictions, one by one, without
        mixing them.

        Returns
        -------
        array of shape (K, N, D_y)
            Mean output vectors of each classifier.
        """
        check_is_fitted(self)

        if self.add_bias:
            X = add_bias(X)

        return self._predicts(X)


    def _predicts(self, X):
        """
        No bias is added and no fitted check is performed.
        """
        N = len(X)

        y = np.zeros((self.K_, N, self.D_y_))
        for k in range(self.K_):
            # A classifier's prediction.
            y[k] = X @ self.W_[k].T
        return y

    def predict_vars(self, X):
        """
        Returns this model's classifiers' prediction variance, one by one,
        without mixing them.

        Returns
        -------
        array of shape (K, N)
            Prediction variances of each classifier.
        """
        check_is_fitted(self)

        if self.add_bias:
            X = add_bias(X)

        return self._predict_vars(X)

    def _predict_vars(self, X):
        """
        No bias is added and no fitted check is performed.
        """
        N = len(X)

        y_var = np.zeros((self.K_, N, self.D_y_))
        for k in range(self.K_):
            # A classifier's prediction variance.
            var = 2 * self.b_tau_[k] / (self.a_tau_[k] - 1) * (
                1 + np.sum(X * X @ self.Lambda_1_[k], 1))
            y_var[k] = var.reshape((len(X), self.D_y_)).repeat(self.D_y_,
                                                               axis=1)

        return y_var
