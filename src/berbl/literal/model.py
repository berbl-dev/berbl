from typing import *

import numpy as np  # type: ignore

from ..utils import EXP_MIN, LN_MAX, add_bias, check_phi, matching_matrix
from . import mixing, model_probability


def fix_y_shape(y):
    # Some people (looking at you, sklearn.pipeline) deform y to shape
    # (-1,).
    if len(y.shape) == 1:
        y = y[:,np.newaxis]
    return y


class Model:
    def __init__(self,
                 matchs: List,
                 random_state,
                 add_bias=True,
                 phi=None):
        """
        A model based on mixing localized linear submodels using the given model
        structure.

        Parameters
        ----------
        matchs
            A list of matching functions (i.e. objects implementing a `match`
            attribute) defining the structure of this mixture.
        random_state : RandomState object
        add_bias : bool
            Whether to add an all-ones bias column to the input data.
        phi
            mixing feature extractor (N × Dx → N × DV); if `None` uses the
            default LCS mixing feature matrix based on `phi(x) = 1`.
        """
        self.matchs = matchs
        self.add_bias = add_bias
        self.phi = phi
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):

        y = fix_y_shape(y)

        if self.add_bias:
            X = add_bias(X)

        self.K_ = len(self.matchs)
        _, self.Dx_ = X.shape
        _, self.Dy_ = y.shape

        Phi = check_phi(self.phi, X)

        self.metrics_, self.params_ = model_probability(
            matchs=self.matchs,
            X=X,
            Y=y,
            Phi=Phi,
            random_state=self.random_state)

        self.L_q_ = self.metrics_["L_q"]
        self.ln_p_M_ = self.metrics_["ln_p_M"]
        self.L_C_q_ = self.metrics_["L_C_q"]
        self.L_M_q_ = self.metrics_["L_M_q"]
        self.p_M_D_ = self.metrics_["p_M_D"]

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
        Calculates prediction means and variances of the model for the provided
        inputs.
        """
        y = np.zeros((len(X), self.Dy_))
        y_var = np.zeros((len(X), self.Dy_))
        for i in range(len(X)):
            y[i], y_var[i] = self.predict_mean_var1(X[i])

        return y, y_var

    def predict_mean_var1(self, x: np.ndarray):
        """
        Calculates prediction mean and variance of the model for the provided
        input.

        Literal (and inefficient) version given in [Drugowitsch's book](/).
        """
        Dy, Dx = self.W_[0].shape

        if self.add_bias:
            x = np.append(1, x)

        X = np.array([x])

        Phi = check_phi(self.phi, X)
        M = matching_matrix(self.matchs, X)
        G = mixing(M, Phi, self.V_, exp_min=EXP_MIN, ln_max=LN_MAX)  # shape ((N=1), K)
        # assert G.shape == (1, self.K_)
        g = G[0]

        # Mean (7.108).
        gW = 0
        for k in range(len(self.W_)):
            gW += g[k] * self.W_[k]
        y = gW @ x

        # Variance (7.109).
        var = np.zeros(Dy)
        for j in range(Dy):
            for k in range(self.K_):
                var[j] += g[k] * (2 * self.b_tau_[k] / (self.a_tau_[k] - 1) *
                                  (1 + x @ self.Lambda_1_[k] @ x) +
                                  (self.W_[k][j] @ x)**2)
            var[j] -= y[j]**2

        return y, var

    def predict_mean_var_(self, X: np.ndarray):
        """
        [PDF p. 224]

        The mean and variance of the predictive density described by this model
        for each of the provided data points.

        “As the mixture of Student’s t distributions might be multimodal, there
        exists no clear definition for the 95% confidence intervals, but a
        mixture density-related study that deals with this problem can be found
        in [118].  Here, we take the variance as a sufficient indicator of the
        prediction’s confidence.” [^1]
        
        [^1]: Jan Drugowitsch. 2008. Design and Analysis of Learning Classifier
        Systems - A Probabilistic Approach. [PDF p. 224]
        
        Parameters
        ----------
        X : array of shape (N, Dx)
            Input vector.

        Returns
        -------
        y : array of shape (N, Dy)
            Mean output vector.
        y_var : array of shape (N, Dy)
            Variance of output.
        """
        N, _ = X.shape
        Dy, Dx = self.W_[0].shape

        if self.add_bias:
            X = add_bias(X)

        # Collect the independent predictions and variances of each submodel. We
        # use the definitions of those that do neither perform input checking
        # nor bias adding to save some time.
        ys = self._predicts(X)
        y_vars = self._predict_vars(X)

        # Next, mix the predictions.
        Phi = check_phi(self.phi, X)
        M = matching_matrix(self.matchs, X)
        # shape: ((N=1), K)
        G_ = mixing(M, Phi, self.V_, exp_min=EXP_MIN, ln_max=LN_MAX)

        # For each rule's prediction, we weigh every dimension of the output
        # vector by the same amount, thus we simply repeat the G values over Dy.
        G = G_.reshape(ys.shape).repeat(Dy, axis=2)  # K × N × Dy

        y = np.sum(G * ys, axis=0)
        y_var = np.sum(G * (y_vars + ys**2), axis=0) - y**2

        return y, y_var

    def predicts(self, X):
        """
        Returns this model's submodels' predictions, one by one, without mixing
        them.

        Returns
        -------
        array of shape (K, N, Dy)
            Mean output vectors of each submodel.
        """
        if self.add_bias:
            X = add_bias(X)

        return self._predicts(X)

    def _predicts(self, X):
        """
        No bias is added.
        """
        N = len(X)

        y = np.zeros((self.K_, N, self.Dy_))
        # TODO Maybe more efficient: np.sum(W[k] * X, axis=1)
        for k in range(self.K_):
            # A submodel's prediction.
            y[k] = X @ self.W_[k].T
        return y

    def predict_vars(self, X):
        """
        Returns this model's submodels' prediction variances, one by one, without
        mixing them.

        Returns
        -------
        array of shape (K, N)
            Prediction variances of each submodel.
        """
        if self.add_bias:
            X = add_bias(X)

        return self._predict_vars(X)

    def _predict_vars(self, X):
        """
        No bias is added.
        """
        N = len(X)

        y_var = np.zeros((self.K_, N, self.Dy_))
        for k in range(self.K_):
            # A submodel's prediction variance.
            var = 2 * self.b_tau_[k] / (self.a_tau_[k] - 1) * (
                1 + np.sum(X * X @ self.Lambda_1_[k], axis=1))
            y_var[k] = var.reshape((len(X), self.Dy_)).repeat(self.Dy_, axis=1)

        return y_var

    def predict_distribution(self, x):
        raise NotImplementedError("Not yet implemented, has to be ported to literal")
