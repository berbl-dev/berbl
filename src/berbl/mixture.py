from typing import *

import numpy as np  # type: ignore
from tqdm import trange  # type: ignore

from .utils import add_bias, check_phi, t
from .rule import Rule
from .mixing import Mixing
from .mixing_laplace import MixingLaplace


class Mixture:

    def __init__(self,
                 matchs: List,
                 random_state,
                 add_bias=True,
                 phi=None,
                 fit_mixing="laplace",
                 **kwargs):
        """
        A model based on mixing linear regression rules using the given model
        structure.

        Parameters
        ----------
        matchs
            A list of matching functions (i.e. objects implementing a `match`
            attribute) defining the structure of this mixture.
        random_state : RandomState object
        add_bias : bool
            Whether to add an all-ones bias column to the input data.
        phi : callable
            Mixing feature extractor (N × DX → N × DV); if `None` uses the
            default LCS mixing feature matrix based on `phi(x) = 1`.
        fit_mixing : str
            Either of "bouchard" or "laplace".
        **kwargs
            This is passed through unchanged to both
            [`Mixing`][berbl.mixing.Mixing] and [`Rule`][berbl.rule.Rule].
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
        X : array of shape (N, DX)
            Input matrix.
        y : array of shape (N, Dy)
            Output matrix.
        """

        if self.add_bias:
            X = add_bias(X)

        self.K_ = len(self.matchs)
        _, self.DX_ = X.shape
        y = y.reshape((len(X), -1))
        _, self.Dy_ = y.shape

        # Train submodels.
        #
        # “When fit is called, any previous call to fit should be ignored.”
        self.rules_ = list(map(lambda m: Rule(m, **self.__kwargs),
                               self.matchs))
        # TODO Cache trained rules at the GA level.
        for k in trange(self.K_, desc="Fit rules", leave=False):
            self.rules_[k].fit(X, y)

        # Train mixing model.
        #
        # “When fit is called, any previous call to fit should be ignored.”
        if self.fit_mixing == "bouchard":
            raise NotImplementedError(
                "Mixing based on Bouchard bound not yet implemented")
            self.mixing_ = Mixing(rules=self.rules_,
                                  phi=self.phi,
                                  random_state=self.random_state,
                                  **self.__kwargs)
        elif self.fit_mixing == "laplace":
            self.mixing_ = MixingLaplace(rules=self.rules_,
                                         phi=self.phi,
                                         random_state=self.random_state,
                                         **self.__kwargs)
        else:
            raise NotImplementedError(
                "Only 'bouchard' and 'laplace' supported for fit_mixing")
        self.mixing_.fit(X, y)

        # We need to recalculate the rules' variational bounds here because we
        # now have access to the final value of R (which we substituted by M
        # during submodel training for ensuring independence).
        self.L_C_q_ = np.repeat(-np.inf, len(self.rules_))
        for k in range(self.K_):
            self.L_C_q_[k] = self.rules_[k].var_bound(X,
                                                      y,
                                                      r=self.mixing_.R_[:,
                                                                        [k]])
        self.L_M_q_ = self.mixing_.L_M_q_
        self.L_q_ = np.sum(self.L_C_q_) + self.L_M_q_
        # TODO Replace this with volume-dependent formula (e.g. the one from the
        # book s.t. p(K) = \exp(-V) V^K/K!).
        self.ln_p_M_ = -np.log(float(np.math.factorial(
            self.K_)))  # (7.3), i.e. p_M \propto 1/K
        self.ln_p_M_D_ = self.L_q_ + self.ln_p_M_

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
        prediction’s confidence.” [^1]

        [^1]: Jan Drugowitsch. 2008. Design and Analysis of Learning Classifier
        Systems - A Probabilistic Approach. [PDF p. 224]

        Parameters
        ----------
        X : array of shape (N, DX)
            Input matrix.

        Returns
        -------
        y : array of shape (N, Dy)
        y_var : array of shape (N, Dy)
        """
        N, _ = X.shape
        Dy, DX = self.rules_[0].W_.shape

        Phi = check_phi(self.phi, X)

        # After having called ``predicts``, add the bias term (``predicts`` also
        # adds the bias term internally).
        if self.add_bias:
            X = add_bias(X)

        # Collect the independent predictions and variances of each submodel. We
        # use the implementations of those that do neither perform input
        # checking nor bias adding to save some time.
        ys = self._predicts(X)  # shape (K, N, Dy)
        y_vars = self._predict_vars(X)

        G_ = self.mixing_.mixing(X).T  # K × N

        # For each submodel's prediction, we weigh every dimension of the output
        # vector by the same amount, thus we simply repeat the G values over Dy.
        G = G_[:, :, np.newaxis].repeat(Dy, axis=2)  # K × N × Dy

        y = np.sum(G * ys, axis=0)

        y_var = np.sum(G * (y_vars + ys**2), axis=0) - y**2

        return y, y_var

    def predicts(self, X):
        """
        Returns this model's submodel's predictions, one by one, without mixing
        them.

        Parameters
        ----------
        X : array of shape (N, DX)
            Input matrix.

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
        for k in range(self.K_):
            y[k] = self.rules_[k].predict(X)
        return y

    def predict_vars(self, X):
        """
        Returns this model's submodel's prediction variances, one by one,
        without mixing them.

        Parameters
        ----------
        X : array of shape (N, DX)
            Input matrix.

        Returns
        -------
        array of shape (K, N, Dy)
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

        y_vars = np.zeros((self.K_, N, self.Dy_))
        for k in range(self.K_):
            y_vars[k] = self.rules_[k].predict_var(X)

        return y_vars

    def predict_distribution(self, X):
        """
        Returns
        -------
        callable
            A function expecting a `y` and returning the values of the
            predictive distributions at positions `X`.
        """

        if self.add_bias:
            X = add_bias(X)

        G = self.mixing_.mixing(X).T  # (K, N)
        W = self._predicts(X)  # (K, N, Dy)
        var = self._predict_vars(X)  # (K, N, Dy)

        def pdf(y):
            # TODO Vectorize if possible
            res = 0
            for k in range(self.K_):
                prd = 1
                for j in range(self.Dy_):
                    prd *= t(mu=W[k][:, j],
                             prec=2 / var[k][:, j],
                             df=2 * self.rules_[k].a_tau_)(y)
                res += G[k] * prd
            return res

        return pdf
