from typing import *

import numpy as np  # type: ignore
from prolcs.common import matching_matrix, phi_standard, initRepeat_binom
from sklearn.utils import check_consistent_length  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from ..utils import add_bias
from . import model_probability, mixing


class Mixture():
    def __init__(
            self,
            matchs: List,
            # TODO Shouldn't ranges have been removed already?!
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
        fit_mixing
            either of "bouchard" or "laplace"
        random_state
            See ``n_cls``.
        **kwargs
            This is passed through unchanged to both ``Mixing`` and
            ``Classifier``.
        """

        self.matchs = matchs
        self.add_bias = add_bias
        self.phi = phi
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits this model to the provided data.

        :param X: input matrix (N × D_X)
        :param y: output matrix (N × D_y)
        """
        # TODO Use NumPy style of param dimension descriptions

        check_consistent_length(X, y)

        if self.add_bias:
            X = add_bias(X)

        random_state = check_random_state(self.random_state)

        if self.phi is None:
            Phi = np.ones((len(X), 1))
        else:
            Phi = self.phi(X)

        self.p_M_D_, self.params_ = model_probability(self.matchs,
                                                      X=X,
                                                      Y=y,
                                                      Phi=Phi)

        self.D_X_ = X.shape[1]
        self.D_y_ = y.shape[1]
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
        self.L_q_ = self.params_["L_q"]
        self.ln_p_M_ = self.params_["ln_p_M"]
        self.L_k_q_ = self.params_["L_k_q"]
        self.L_M_q_ = self.params_["L_M_q"]
        self.K_ = len(self.W_)

        return self

    def predict(self, X):

        return self.predict_mean_var(X)[0]

    def predict_mean_var(self, X):
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

        X = add_bias(X)

        N = len(X)

        y = np.zeros((N, self.D_y_))
        y_var = np.zeros((N, self.D_y_))
        for i in range(N):
            y[i], y_var[i] = self.predict1_mean_var(X[i])

        return y, y_var

    def predict1_mean_var(self, x):
        X = x.reshape((1, -1))
        M = matching_matrix(self.matchs, X)
        N, K = M.shape
        # TODO use self.D_y_ instead of D_Y
        # TODO Rename properly (dY?)

        # TODO Reduce duplication
        if self.phi is None:
            Phi = np.ones((len(X), 1))
        else:
            Phi = self.phi(X)

        G = mixing(M, Phi, self.V_)  # (N=1) × K
        g = G[0]
        D_Y, D_X = self.W_[0].shape

        W = np.array(self.W_)

        # (\sum_k g_k(x) W_k) x, (7.108)
        # TODO This can probably be vectorized
        gW = 0
        for k in range(len(W)):
            gW += g[k] * W[k]
        # TODO It's probably more efficient to have each classifier predict x
        # and then mix the predictions afterwards (as it is done in
        # LCSBookCode).
        y = gW @ x

        var = np.zeros(D_Y)
        # TODO Can this be vectorized?
        for j in range(D_Y):
            for k in range(K):
                var[j] += g[k] * (2 * self.b_tau_[k] / (self.a_tau_[k] - 1) *
                                  (1 + x @ self.Lambda_1_[k] @ x) +
                                  (self.W_[k][j] @ x)**2)
            var[j] -= y[j]**2
        return y, var
