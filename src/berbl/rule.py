import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore


class Rule():
    """
    A local linear regression model (in LCS speak, a “linear regression
    classifier”) based on the provided match function.
    """
    def __init__(self,
                 match,
                 A_ALPHA=10**-2,
                 B_ALPHA=10**-4,
                 A_TAU=10**-2,
                 B_TAU=10**-4,
                 DELTA_S_L_K_Q=10**-4,
                 MAX_ITER=20,
                 **kwargs):
        """
        Parameters
        ----------
        match : object
            ``match.match`` is this rule's match function. According to
            Drugowitsch's framework (or mixture of experts), each rule should
            get assigned a responsibility for each data point. However, in order
            to be able to train the submodels independently, that responsibility
            (which depends on the matching function but also on the other rules'
            responsibilities) is replaced with the matching function.
        A_ALPHA : float
            Scale parameter of weight vector variance prior.
        B_ALPHA : float
            Shape parameter of weight vector variance prior.
        A_TAU : float
            Scale parameter of noise variance prior.
        B_TAU : float
            Shape parameter of noise variance prior.
        DELTA_S_L_K_Q : float
            Stopping criterion for variational update loop.
        MAX_ITER : int
            Only perform up to this many iterations of variational
            updates (abort then, even if stopping criterion is not yet met).
        **kwargs : kwargs
            This is here so that we don't need to repeat all the hyperparameters
            in ``Mixture`` etc. ``Mixture`` simply passes through all
            ``**kwargs`` to both ``Mixing`` and ``Rule``. This means that during
            implementation, we need to be aware that if there are parameters in
            those two classes with the same name, they always receive the same
            value.
        """
        self.match = match
        self.A_ALPHA = A_ALPHA
        self.B_ALPHA = B_ALPHA
        self.A_TAU = A_TAU
        self.B_TAU = B_TAU
        self.DELTA_S_L_K_Q = DELTA_S_L_K_Q
        self.MAX_ITER = MAX_ITER

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits this rule's (sub)model to the part of the provided data that the
        rule matches.
        """

        self.m_ = self.match.match(X)

        N, self.DX_ = X.shape
        N, self.Dy_ = y.shape
        X_ = X * np.sqrt(self.m_)
        y_ = y * np.sqrt(self.m_)

        self.a_alpha_, self.b_alpha_ = self.A_ALPHA, self.B_ALPHA
        self.a_tau_, self.b_tau_ = self.A_TAU, self.B_TAU
        self.L_q_ = -np.inf
        delta_L_q = self.DELTA_S_L_K_Q + 1

        # Since this is constant, there's no need to put it into the loop.
        self.a_alpha_ = self.A_ALPHA + self.DX_ * self.Dy_ / 2
        self.a_tau_ = self.A_TAU + 0.5 * np.sum(self.m_)

        iter = 0
        while delta_L_q > self.DELTA_S_L_K_Q and iter < self.MAX_ITER:
            iter += 1
            E_alpha_alpha = self.a_alpha_ / self.b_alpha_
            self.Lambda_ = np.diag([E_alpha_alpha] * self.DX_) + X_.T @ X_
            # While, in theory, Lambda is always invertible here and we thus
            # should be able to use inv (as it is described in the algorithm we
            # implement), we (seldomly) get a singular matrix, probably due to
            # numerical issues. Thus we simply use pinv which yields the same
            # result as inv anyways if the matrix is in fact non-singular. Also,
            # in his own code, Drugowitsch always uses pseudo inverse here.
            self.Lambda_1_ = np.linalg.pinv(self.Lambda_)
            self.W_ = y_.T @ X_ @ self.Lambda_1_
            self.b_tau_ = self.B_TAU + 1 / (2 * self.Dy_) * (
                np.sum(y_ * y_) - np.sum(self.W_ * (self.W_ @ self.Lambda_)))
            E_tau_tau = self.a_tau_ / self.b_tau_
            # Dy factor in front of trace due to sum over Dy elements (7.100).
            self.b_alpha_ = self.B_ALPHA + 0.5 * (E_tau_tau * np.sum(
                self.W_ * self.W_) + self.Dy_ * np.trace(self.Lambda_1_))
            L_q_prev = self.L_q_
            self.L_q_ = self.var_bound(
                X=X,
                y=y,
                # Substitute r by m in order to train submodels independently
                # (see [PDF p. 219]). Note, however, that after having trained
                # the mixing model we finally evaluate the submodels using
                # ``r=R[:,[k]]`` though.
                r=self.m_)
            delta_L_q = self.L_q_ - L_q_prev

        return self

    def predict(self, X):
        """
        This model's mean at the given positions; may serve as a prediction.

        Parameters
        ----------
        X : array of shape (N, DX)

        Returns
        -------
        mean : array of shape (N, Dy)
        """
        return X @ self.W_.T

    def predict_var(self, X):
        """
        This model's variance at the given positions; may serve as some kind of
        confidence estimate for the prediction.

        The model currently assumes the same variance in all dimensions; thus
        the same value is repeated for each dimension.

        Parameters
        ----------
        X : array of shape (N, DX)

        Returns
        -------
        variance : array of shape (N, Dy)
        """
        # The sum corresponds to x @ self.Lambda_1 @ x for each x in X (i.e.
        # np.diag(X @ self.Lambda_1_ @ X.T)).
        var = 2 * self.b_tau_ / (self.a_tau_ - 1) * (
            1 + np.sum((X @ self.Lambda_1_) * X, axis=1))
        # The same value is repeated for each dimension since the model
        # currently assumes the same variance in all dimensions.
        return var[:,np.newaxis].repeat(self.Dy_, axis=1)

    def var_bound(self, X: np.ndarray, y: np.ndarray, r: np.ndarray):
        """
        The components of the variational bound specific to this rule.

        See VarClBound [PDF p. 247].

        Parameters
        ----------
        X : array of shape (N, DX)
            Input matrix.
        y : array of shape (N, Dy)
            Output matrix.
        r : array of shape (N, 1)
            Responsibilities (during training replaced with matching array of
            this rule in order to enable independent submodel training).
        """
        E_tau_tau = self.a_tau_ / self.b_tau_
        L_1_q = self.Dy_ / 2 * (ss.digamma(self.a_tau_) - np.log(self.b_tau_)
                                 - np.log(2 * np.pi)) * np.sum(r)
        # We reshape r to a NumPy row vector since NumPy seems to understand
        # what we want to do when we multiply two row vectors (i.e. a^T a).
        L_2_q = (-0.5 * r).reshape(
            (-1)) @ (E_tau_tau * np.sum((y - X @ self.W_.T)**2, 1)
                     + self.Dy_ * np.sum(X * (X @ self.Lambda_1_), 1))
        L_3_q = -ss.gammaln(self.A_ALPHA) + self.A_ALPHA * np.log(
            self.B_ALPHA) + ss.gammaln(self.a_alpha_) - self.a_alpha_ * np.log(
                self.b_alpha_
            ) + self.DX_ * self.Dy_ / 2 + self.Dy_ / 2 * np.log(
                np.linalg.det(self.Lambda_1_))
        L_4_q = self.Dy_ * (
            -ss.gammaln(self.A_TAU) + self.A_TAU * np.log(self.B_TAU) +
            (self.A_TAU - self.a_tau_) * ss.digamma(self.a_tau_)
            - self.A_TAU * np.log(self.b_tau_) - self.B_TAU * E_tau_tau
            + ss.gammaln(self.a_tau_) + self.a_tau_)
        return L_1_q + L_2_q + L_3_q + L_4_q
