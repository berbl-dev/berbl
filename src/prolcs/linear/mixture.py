from typing import *

import numpy as np  # type: ignore
from sklearn.utils import check_consistent_length  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from ..radialmatch1d import RadialMatch1D
from .classifier import Classifier
from .mixing import Mixing
from .mixing_laplace import MixingLaplace


class Mixture():
    def __init__(self,
                 n_cls=10,
                 cl_class=RadialMatch1D,
                 ranges=None,
                 matchs: List = None,
                 phi=None,
                 fit_mixing="bouchard",
                 random_state=None,
                 **kwargs):
        """
        A model based on mixing linear classifiers using the given model
        structure.

        :param n_cls: Generate ``n_cls`` many random classifiers (using
            ``random_state``) with class ``cl_class``. If ``n_cls`` is given,
            ``matchs`` must be ``None``.
        :param cl_class: See ``n_cls``.
        :param random_state: See ``n_cls``.
        :param matchs: A list of matching functions (i.e. objects implementing a
            ``match`` attribute) defining the structure of this mixture. If
            given, ``n_cls`` and ``cl_class`` are not used to generate
            classifiers randomly.
        :param phi: mixing feature extractor (N × D_X → N × D_V); if ``None``
            uses the default LCS mixing feature matrix based on ``phi(x) = 1``
        :param fit_mixing: either of "bouchard" or "laplace"
        :param **kwargs: This is passed through unchanged to both ``Mixing`` and
            ``Classifier``.
        """

        self.n_cls = n_cls
        self.cl_class = cl_class
        self.ranges = ranges
        self.matchs = matchs
        self.phi = phi
        self.fit_mixing = fit_mixing
        self.random_state = random_state
        self.__kwargs = kwargs

    def _random_matchs(self, n_cls, cl_class, ranges, random_state):
        return [
            cl_class.random(ranges, random_state=random_state)
            for i in range(n_cls)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits this model to the provided data.

        Note that unless ``X`` contains a bias column (e.g. a column of ones),
        the individual classifiers do not fit the intercept.

        :param X: input matrix (N × D_X)
        :param y: output matrix (N × D_y)
        """
        # TODO Use NumPy style of param dimension descriptions

        check_consistent_length(X, y)

        random_state = check_random_state(self.random_state)

        if (self.matchs is None and self.n_cls is not None
                and self.cl_class is not None):
            self.matchs_ = self._random_matchs(self.n_cls, self.cl_class,
                                               self.ranges, random_state)
        elif self.matchs is not None:
            self.matchs_ = self.matchs
        else:
            raise ValueError(
                f"If matchs isn't given, must provide at least n_cls, cl_class "
                f"and ranges and these are {self.n_cls}, {self.cl_class} and"
                f"{self.ranges}")

        self.K_ = len(self.matchs_)
        _, self.D_X_ = X.shape
        _, self.D_y_ = y.shape

        # Train classifiers.
        #
        # “When fit is called, any previous call to fit should be ignored.”
        self.classifiers_ = list(
            map(lambda m: Classifier(m, **self.__kwargs), self.matchs_))
        for k in range(self.K_):
            self.classifiers_[k].fit(X, y)

        # Train mixing model.
        #
        # “When fit is called, any previous call to fit should be ignored.”
        if self.fit_mixing == "bouchard":
            self.mixing_ = Mixing(classifiers=self.classifiers_,
                                  phi=self.phi,
                                  random_state=random_state,
                                  **self.__kwargs)
        elif self.fit_mixing == "laplace":
            self.mixing_ = MixingLaplace(classifiers=self.classifiers_,
                                         phi=self.phi,
                                         random_state=random_state,
                                         **self.__kwargs)
        else:
            raise NotImplementedError(
                "Only 'bouchard' and 'laplace' supported for fit_mixing")
        self.mixing_.fit(X, y)

        # We need to recalculate the classifiers' here because we now have
        # access to the final value of R (which we substituted by M during
        # classifier training for ensuring indenpendence).
        self.L_C_q_ = np.repeat(-np.inf, len(self.classifiers_))
        for k in range(self.K_):
            self.L_C_q_[k] = self.classifiers_[k].var_bound(
                X, y, r=self.mixing_.R_[:, [k]])
        self.L_M_q_ = self.mixing_.L_M_q_
        self.L_q_ = np.sum(self.L_C_q_) + self.L_M_q_
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

        :returns: mean output vector (N × D_y), variance of output (N × D_y)
        """
        check_is_fitted(self)

        N, _ = X.shape
        D_y, D_X = self.classifiers_[0].W_.shape

        if self.phi is None:
            Phi = np.ones((len(X), 1))
        else:
            raise NotImplementedError("phi is not None in Mixing")

        y = self.predicts(X)

        y_var = np.zeros((self.K_, N, D_y))
        for k in range(self.K_):
            y_var[k] = self.classifiers_[k].predict_var(X)

        G_ = self.mixing_.mixing(X).T  # K × N

        # For each classifier's prediction, we weigh every dimension of the
        # output vector by the same amount, thus we simply repeat the G values
        # over D_y.
        G = G_.reshape(y.shape).repeat(D_y, axis=2)  # K × N × D_y

        y = np.sum(G * y, axis=0)

        var = np.zeros((N, D_y))
        # TODO Re-check this for correctness (should(?) probably be the same as
        # the following loop but is not?)
        y_var = np.sum(G * (y_var + y**2), axis=0) - y**2
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

        :returns: mean output vectors of each classifier (K × N × D_y)
        """
        check_is_fitted(self)

        N = len(X)

        y = np.zeros((self.K_, N, self.D_y_))
        for k in range(self.K_):
            y[k] = self.classifiers_[k].predict(X)
        return y
