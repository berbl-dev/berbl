import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore

from .literal import hessian, _kl
from .mixing import Mixing
from .utils import check_phi, known_issue


# It's not *that* nice to inherit from Mixing because they should be siblings
# and not parent and child.
class MixingLaplace(Mixing):
    """
    Model for the mixing weights of a set of linear regression rules. Fitted
    using Drugowitsch's Laplace approximation.

    Structurally, the main difference to [`Mixing`][berbl.mixing.Mixing] is 
    that `Lambda_V_1` is a matrix of shape `(K * DV, K * DV)` (i.e. the 
    mixing problem is solved for all submodels at once), whereas 
    [`Mixing`][berbl.mixing.Mixing] has `K` mixing matrices (one for each 
    submodel).
    """
    def __init__(self, DELTA_S_KLRG=10**-8, **kwargs):
        """
        Parameters
        ----------
        DELTA_S_KLRG : float
            Stopping criterion for the iterative Laplace approximation in the
            mixing weight update.
        **kwargs : kwargs
            This is here for two reasons: To be able to provide the parent with
            all the parameters it uses (we only add `DELTA_S_KLRG`) and so
            that we don't need to repeat all the hyperparameters in 
            [`Mixture`][berbl.mixture.Mixture] etc. 
            [`Mixture`][berbl.mixture.Mixture] simply passes through all 
            `**kwargs` to both [`Mixing`][berbl.mixing.Mixing] and 
            [`Rule`][berbl.rule.Rule]. This means that during implementation,
            we need to be aware that if there are parameters in those two 
            classes with the same name, they always receive the same value.
        """
        self.DELTA_S_KLRG = DELTA_S_KLRG
        super().__init__(**kwargs)

    def fit(self, X, y):
        Phi = check_phi(self.phi, X)

        M = np.hstack([rule.m_ for rule in self.rules])

        _, self.DX_ = X.shape
        _, self.Dy_ = y.shape
        N, self.DV_ = Phi.shape

        # NOTE The scale of this normal is wrong in TrainMixing in Drugowitsch's
        # book (but correct in the text accompanying that algorithm).
        self.V_ = self.random_state.normal(loc=0,
                                           scale=self.B_BETA / self.A_BETA,
                                           size=(self.DV_, self.K))
        # self.a_beta_ is constant (but for the first run of the loop).
        self.a_beta_ = np.repeat(self.A_BETA + self.DV_ / 2, self.K)
        self.b_beta_ = np.repeat(self.B_BETA, self.K)

        # NOTE We extracted this from _train_mix_weights for efficiency's sake.
        self.G_ = self._mixing(M, Phi, self.V_)
        self.R_ = self._responsibilities(X=X, y=y, G=self.G_)

        # Required for being able to calculate an initial variational bound.
        # self.Lambda_V_1_ = np.diag(np.repeat(self.B_BETA / self.A_BETA, self.K))

        # TODO Perform one computation of the loop beforehand with initial
        # a_alpha, a_tau etc. and after that use the values that are constant in
        # the loop

        # self.L_M_q_ = self._var_bound(G=self.G_,
        #                               R=self.R_,
        #                               V=self.V_,
        #                               Lambda_V_1=self.Lambda_V_1_,
        #                               a_beta=np.repeat(self.A_BETA, self.K),
        #                               b_beta=self.b_beta_)
        self.L_M_q_ = -np.inf
        delta_L_M_q = self.DELTA_S_L_M_Q + 1
        i = 0
        while delta_L_M_q > self.DELTA_S_L_M_Q and i < self.MAX_ITER_MIXING:
            i += 1

            # NOTE We inline TrainMixWeights here for better control of
            # performance optimizations (e.g. precomputing stuff efc.).

            # TODO Improve this a_beta business
            a_beta = (self.a_beta_ if i > 1 else np.repeat(
                self.A_BETA, self.K))
            E_beta_beta = a_beta / self.b_beta_

            KLRG = _kl(self.R_, self.G_)
            delta_KLRG = self.DELTA_S_KLRG + 1
            j = 0
            while (delta_KLRG > self.DELTA_S_KLRG and j < self.MAX_ITER_MIXING
                   and not np.isclose(KLRG, 0)):
                j += 1
                # Actually, this should probably be named nabla_E.
                E = Phi.T @ (self.G_ - self.R_) + self.V_ * E_beta_beta
                e = E.T.ravel()
                H = hessian(Phi=Phi,
                            G=self.G_,
                            a_beta=a_beta,
                            b_beta=self.b_beta_)
                # Preference of `-` and `@` is OK here, we checked.
                #
                # While, in theory, H is always invertible here and we thus
                # should be able to use inv (as it is described in the algorithm
                # we implement), we (seldomly) get a singular H, probably due to
                # numerical issues. Thus we simply use pinv which yields the
                # same result as inv anyways if H is non-singular. Also, in his
                # own code, Drugowitsch always uses pseudo inverse here.
                delta_v = -np.linalg.pinv(H) @ e
                # “DV × K matrix with jk'th element given by ((k - 1) K + j)'th
                # element of v.” (Probably means “delta_v”.)
                self.V_ += delta_v.reshape((self.K, self.DV_)).T

                self.G_ = self._mixing(M=M, Phi=Phi, V=self.V_)
                self.R_ = self._responsibilities(X=X, y=y, G=self.G_)

                KLRG_prev = KLRG
                KLRG = _kl(self.R_, self.G_)
                delta_KLRG = np.abs(KLRG_prev - KLRG)

            H = hessian(Phi=Phi, G=self.G_, a_beta=a_beta, b_beta=self.b_beta_)
            # While, in theory, H is always invertible here and we thus should
            # be able to use inv (as it is described in the algorithm we
            # implement), we (seldomly) get a singular H, probably due to
            # numerical issues. Thus we simply use pinv which yields the same
            # result as inv anyways if H is non-singular. Also, in his own code,
            # Drugowitsch always uses pseudo inverse here.
            # NOTE that instead of storing Lambda_V_1, Drugowitsch's LCSBookCode
            # computes and stores np.slogdet(Lambda_V_1) and cov_Tr (the latter
            # of which is used in his update_gating).
            self.Lambda_V_1_ = np.linalg.pinv(H)

            # Interestingly, LCSBookCode performs this *before* TrainMixWeights.
            # That doesn't change a thing, though, and also makes slightly
            # awkward initialization of self.Lambda_V_1 necessary.
            self.b_beta_ = self._train_b_beta(V=self.V_,
                                              Lambda_V_1=self.Lambda_V_1_)

            L_M_q_prev = self.L_M_q_
            self.L_M_q_ = self._var_bound(G=self.G_,
                                          R=self.R_,
                                          V=self.V_,
                                          Lambda_V_1=self.Lambda_V_1_,
                                          a_beta=self.a_beta_,
                                          b_beta=self.b_beta_)
            # LCSBookCode states: “as we are using an approximation, the variational
            # bound might decrease, so we're not checking and need to take the
            # abs()”. I guess with approximation he means the use of the Laplace
            # approximation (which may violate the lower bound nature of L_M_q).
            delta_L_M_q = np.abs(self.L_M_q_ - L_M_q_prev)
            # if self.L_M_q < L_M_q_prev:
            #     print(f"self.L_M_q < L_M_q_prev: {self.L_M_q} < {L_M_q_prev}")

        return self

    def _train_mix_weights(self, M, X, y, Phi, G, R, V, a_beta, b_beta):
        """
        Training routine for mixing weights based on a Laplace approximation
        (see [Drugowitsch's book](/)).

        Parameters
        ----------
        M : array of shape (N, K)
            Matching matrix.
        X : array of shape (N, DX)
            Input matrix.
        y : array of shape (N, Dy)
            Output matrix.
        Phi : array of shape (N, DV)
            Mixing feature matrix.
        G : array of shape (N, K)
            Mixing (“gating”) matrix.
        R : array of shape (N, K)
            Responsibility matrix.
        V : array of shape (DV, K)
            Mixing weight matrix.
        a_beta : array of shape (K,)
            Mixing weight prior parameter (row vector).
        b_beta : array of shape (K,)
            Mixing weight prior parameter (row vector).

        Returns
        -------
        V : array of shape (DV, K) 
            Updated mixing weight matrix.
        Lambda_V_1 : array of shape (K DV, K DV)
            Updated mixing weight covariance matrix.
        """
        # NOTE We don't use the version from literal here because we
        # cache/precompute several values that are computed each time
        # literal.train_mix_weights is called.

    def _train_b_beta(self, V: np.ndarray, Lambda_V_1: np.ndarray):
        """
        [PDF p. 244]

        TrainMixPriors but only the part concerned with `b_beta` since
        `a_beta` is constant.

        Note that we override this because `Lambda_V_1` has a different form
        here than in [`Mixing`][berbl.mixing.Mixing] (where it is a list of 
        `K` matrices with shapes `(DV, DV)`).

        Parameters
        ----------
        V : array of shape (DV, K)
            Mixing weight matrix.
        Lambda_V_1 : array of shape (K DV, K DV)
            Mixing weight covariance matrix.

        Returns
        -------
        b_beta : array of shape (K,)
            Mixing weight vector prior parameter.
        """
        DV, K = V.shape
        b_beta = np.repeat(self.B_BETA, (self.K, ))
        Lambda_V_1_diag = np.diag(Lambda_V_1)
        # TODO Performance: LCSBookCode vectorized this:
        # b[:,1] = b_b + 0.5 * (sum(V * V, 0) + self.cov_Tr)
        for k in range(self.K):
            v_k = V[:, [k]]
            l = k * DV
            u = (k + 1) * DV
            # print(f"sum {k} Lambda_V_1_diag", np.sum(Lambda_V_1_diag[l:u:1]))
            # Not that efficient, I think (but very close to [PDF p. 244]).
            # Lambda_V_1_kk = Lambda_V_1[l:u:1, l:u:1]
            # b_beta[k] = B_BETA + 0.5 * (np.trace(Lambda_V_1_kk) + v_k.T @ v_k)
            # More efficient.
            try:
                b_beta[k] += 0.5 * (np.sum(Lambda_V_1_diag[l:u:1])
                                    + v_k.T @ v_k)
            except FloatingPointError as e:
                known_issue("FloatingPointError in train_mix_priors",
                            (f"v_k = {v_k}, "
                             f"K = {self.K}, "
                             f"V = {V}, "
                             f"Lambda_V_1 = {Lambda_V_1}"),
                            report=True)
                mlflow.set_tag("FloatingPointError_train_mix_priors",
                               "occurred")
                raise e

        return b_beta

    def _var_bound(self, G: np.ndarray, R: np.ndarray, V: np.ndarray,
                   Lambda_V_1: np.ndarray, a_beta: np.ndarray,
                   b_beta: np.ndarray):
        """
        [PDF p. 245]

        Parameters
        ----------
        G : array of shape (N, K)
            Mixing (“gating”) matrix.
        R : array of shape (N, K)
            Responsibility matrix.
        V : array of shape (DV, K)
            Mixing weight matrix.
        Lambda_V_1 : array of shape (K DV, K DV)
            Mixing weight covariance matrix.
        a_beta : array of shape (K,)
            Mixing weight prior parameter (row vector).
        b_beta : array of shape (K,)

        Returns
        -------
        L_M_q : float
            Mixing component L_M(q) of variational bound.
        """
        # NOTE We don't use the version from literal here because we
        # can cache/precompute several values that are computed each time
        # literal.train_mix_weights is called.
        DV, K = V.shape

        L_M1q = K * (-ss.gammaln(self.A_BETA)
                     + self.A_BETA * np.log(self.B_BETA))
        # TODO Performance: LCSBookCode vectorized this
        # TODO Performance: ss.gammaln(a_beta[k]) is constant throughout the
        # loop in the calling function
        for k in range(self.K):
            L_M1q += ss.gammaln(a_beta[k]) - a_beta[k] * np.log(b_beta[k])

        # L_M2q is the negative Kullback-Leibler divergence [PDF p. 246].
        L_M2q = _kl(R, G)

        # TODO Performance: slogdet can be cached, is computed more than once

        # L_M3q may be -inf after the following line but that is probably OK since
        # the ``train_mixing`` loop then aborts (also see comment in
        # ``train_mixing``).
        L_M3q = 0.5 * np.linalg.slogdet(Lambda_V_1)[1] + K * DV / 2
        if np.any(~np.isfinite([L_M1q, L_M2q, L_M3q])):
            known_issue("Infinite var_mix_bound",
                        (f"Lambda_V_1 = {Lambda_V_1}, "
                         f"L_M1q = {L_M1q}, "
                         f"L_M2q = {L_M2q}, "
                         f"L_M3q = {L_M3q}"))
        return L_M1q + L_M2q + L_M3q
