from typing import *

import numpy as np  # type: ignore
from sklearn.utils import check_consistent_length  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from .utils import add_bias, check_phi
from .rule import Rule
from .mixing import Mixing
from .mixing_laplace import MixingLaplace


class Mixture():
    def __init__(self,
                 matchs: List,
                 add_bias=True,
                 phi=None,
                 fit_mixing="bouchard",
                 random_state=None,
                 **kwargs):
        """
        A model based on mixing linear regression rules using the given model
        structure.

        Parameters
        ----------
        matchs
            A list of matching functions (i.e. objects implementing a ``match``
            attribute) defining the structure of this mixture.
        add_bias : bool
            Whether to add an all-ones bias column to the input data.
        phi
            Mixing feature extractor (N × D_X → N × D_V); if ``None`` uses the
            default LCS mixing feature matrix based on ``phi(x) = 1``.
        fit_mixing
            Either of "bouchard" or "laplace"
        random_state
            See ``n_cls``.
        **kwargs
            This is passed through unchanged to both ``Mixing`` and ``Rule``.
        """

        self.matchs = matchs
        self.add_bias = add_bias
        self.phi = phi
        self.fit_mixing = fit_mixing
        self.random_state = random_state
        # TODO Should probably validate these kwargs because otherwise we don't
        # notice when something is used the wrong way.
        self.__kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits this model to the provided data.

        Parameters
        ----------
        X : array of shape (N, D_X)
            Input matrix.
        y : array of shape (N, D_y)
            Output matrix.
        """
        # TODO Use NumPy style of param dimension descriptions

        check_consistent_length(X, y)

        if self.add_bias:
            X = add_bias(X)

        random_state = check_random_state(self.random_state)

        self.K_ = len(self.matchs)
        _, self.D_X_ = X.shape
        y = y.reshape((len(X), -1))
        _, self.Dy_ = y.shape

        # Train submodels.
        #
        # “When fit is called, any previous call to fit should be ignored.”
        self.rules_ = list(
            map(lambda m: Rule(m, **self.__kwargs), self.matchs))
        for k in range(self.K_):
            self.rules_[k].fit(X, y)

        # Train mixing model.
        #
        # “When fit is called, any previous call to fit should be ignored.”
        if self.fit_mixing == "bouchard":
            self.mixing_ = Mixing(rules=self.rules_,
                                  phi=self.phi,
                                  random_state=random_state,
                                  **self.__kwargs)
        elif self.fit_mixing == "laplace":
            self.mixing_ = MixingLaplace(rules=self.rules_,
                                         phi=self.phi,
                                         random_state=random_state,
                                         **self.__kwargs)
        else:
            raise NotImplementedError(
                "Only 'bouchard' and 'laplace' supported for fit_mixing")
        self.mixing_.fit(X, y)

        # We need to recalculate the rules' variational bounds here because we
        # now have access to the final value of R (which we substituted by M
        # during submodel training for ensuring indenpendence).
        self.L_C_q_ = np.repeat(-np.inf, len(self.rules_))
        for k in range(self.K_):
            self.L_C_q_[k] = self.rules_[k].var_bound(
                X, y, r=self.mixing_.R_[:, [k]])
        self.L_M_q_ = self.mixing_.L_M_q_
        self.L_q_ = np.sum(self.L_C_q_) + self.L_M_q_
        # TODO Replace this with volume-dependent formula (e.g. the one from the
        # book s.t. p(K) = \exp(-V) V^K/K!).
        self.ln_p_M_ = -np.log(float(np.math.factorial(
            self.K_)))  # (7.3), i.e. p_M \propto 1/K
        self.p_M_D_ = self.L_q_ + self.ln_p_M_

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

        :returns: mean output vector (N × Dy), variance of output (N × Dy)
        """
        check_is_fitted(self)

        N, _ = X.shape
        Dy, D_X = self.rules_[0].W_.shape

        Phi = check_phi(self.phi, X)

        # After having called ``predicts``, add the bias term (``predicts`` also
        # adds the bias term internally).
        if self.add_bias:
            X = add_bias(X)

        # Collect the independent predictions and variances of each submodel. We
        # use the implementations of those that do neither perform input
        # checking nor bias adding to save some time.
        ys = self._predicts(X)
        y_vars = self._predict_vars(X)

        G_ = self.mixing_.mixing(X).T  # K × N

        # For each submodel's prediction, we weigh every dimension of the output
        # vector by the same amount, thus we simply repeat the G values over Dy.
        G = G_.reshape(ys.shape).repeat(Dy, axis=2)  # K × N × Dy

        y = np.sum(G * ys, axis=0)

        y_var = np.sum(G * (y_vars + ys**2), axis=0) - y**2

        # TODO Re-check this for correctness (should(?) probably be the same as
        # the following loop but is not?)
        # var = np.zeros((N, Dy))
        # for n in range(N):
        #     x_ = X[n]
        #     g = G_.T[n]
        #     for j in range(Dy):
        #         for k in range(self.K_):
        #             cl = self.rules_[k]
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
        Returns this model's submodel's predictions, one by one, without mixing
        them.

        Returns
        -------
        array of shape (K, N, Dy)
            Mean output vectors of each rule.
        """
        check_is_fitted(self)

        if self.add_bias:
            X = add_bias(X)

        return self._predicts(X)

    def _predicts(self, X):
        """
        No bias is added and no fitted check is performed.

        Parameters
        ----------
        X : array of shape (N, D_X)
            Input matrix.
        """
        N = len(X)

        y = np.zeros((self.K_, N, self.Dy_))
        for k in range(self.K_):
            y[k] = self.rules_[k].predict(X)
        return y

    def predict_vars(self, X):
        """
        Returns this model's submodel's prediction variances, one by one, without
        mixing them.

        Parameters
        ----------
        X : array of shape (N, D_X)
            Input matrix.

        Returns
        -------
        array of shape (K, N)
            Prediction variances of each submodel.
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

        y_vars = np.zeros((self.K_, N, self.Dy_))
        for k in range(self.K_):
            y_vars[k] = self.rules_[k].predict_var(X)

        return y_vars
