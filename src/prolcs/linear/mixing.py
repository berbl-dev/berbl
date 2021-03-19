from typing import List

import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore

from ..common import matching_matrix


class Mixing():
    def __init__(
            self,
            classifiers,
            phi,
            A_BETA=10**-2,
            B_BETA=10**-4,
            DELTA_S_L_M_Q=10**-2,
            MAX_ITER=40,
            EXP_MIN=np.log(np.finfo(None).tiny),
            LN_MAX=np.log(np.finfo(None).max),
    ):
        """
        :param classifiers: List of classifier models (which are held fixed
            during mixing training).
        :param phi: Mixing feature function (X → V), if `None` use the LCS
            default of `phi(x) = 1`.
        :param A_BETA: Scale parameter of mixing weight vector variance prior.
        :param B_BETA: Shape parameter of mixing weight vector variance prior.
        :param DELTA_S_L_M_Q: Stopping criterion for variational update loop.
        :param MAX_ITER: Only perform up to this many iterations of variational
            updates (abort then, even if stopping criterion is not yet met).
        :param exp_min: Lowest real number ``x`` on system such that ``exp(x) >
            0``. The default is the logarithm of the smallest positive number of
            the default dtype (as of 2020-10-06, this dtype is float64).
        :param ln_max: ``ln(x)``, where ``x`` is the highest real number on the
            system. The default is the logarithm of the highest number of the
            default dtype.
        """

        # The set of classifiers is constant here.
        self.CLS = classifiers
        # … as is phi.
        self.PHI = phi
        self.A_BETA = A_BETA
        self.B_BETA = B_BETA
        self.DELTA_S_L_M_Q = DELTA_S_L_M_Q
        self.MAX_ITER = MAX_ITER
        self.EXP_MIN = EXP_MIN
        self.LN_MAX = LN_MAX

    def fit(self, X, y, random_state):
        # NOTE We don't check_random_state here to be more performant (although
        # I'm not sure how much this gains us).
        # TODO Check gain of not using check_random_state here

        if self.PHI is None:
            Phi = np.ones((len(X), 1))
        else:
            raise NotImplementedError("phi is not None in Mixing")

        M = matching_matrix([cl.match for cl in self.CLS], X)

        _, self.K = M.shape
        _, self.D_X = X.shape
        _, self.D_y = y.shape
        N, self.D_V = Phi.shape

        self.V = random_state.normal(loc=0,
                                     scale=self.A_BETA / self.B_BETA,
                                     size=(self.D_V, self.K))
        # a_beta is actually constant so we can set it here and be done with it.
        self.a_beta = np.repeat(self.A_BETA + self.D_V / 2, self.K)
        self.b_beta = np.repeat(self.B_BETA, self.K)

        # Initialize parameters for the Bouchard approximation.
        self.alpha = random_state.normal(loc=0,
                                         scale=self.A_BETA / self.B_BETA,
                                         size=(N, 1))
        # lxi stands for λ(ξ) which is used in Bouchard's approximation. Its
        # supremum value is one eighth.
        self.lxi = random_state.random(size=(N, self.K)) * 0.125
        self.alpha, self.lxi = self._opt_bouchard(M=M,
                                                  Phi=Phi,
                                                  V=self.V,
                                                  alpha=self.alpha,
                                                  lxi=self.lxi)

        self.G = self._mixing(M, Phi, self.V)
        self.R = self._responsibilities(X=X, y=y, G=self.G)

        self.L_M_q = -np.inf
        delta_L_M_q = self.DELTA_S_L_M_Q + 1
        i = 0
        while delta_L_M_q > self.DELTA_S_L_M_Q and i < self.MAX_ITER:
            i += 1

            self.V, self.Lambda_V_1 = self._train_mix_weights(
                M=M,
                X=X,
                y=y,
                Phi=Phi,
                R=self.R,
                V=self.V,
                a_beta=self.a_beta,
                b_beta=self.b_beta,
                lxi=self.lxi,
                alpha=self.alpha)

            # TODO How much faster would in-place ugliness be?
            self.alpha, self.lxi = self._opt_bouchard(M=M,
                                                      Phi=Phi,
                                                      V=self.V,
                                                      alpha=self.alpha,
                                                      lxi=self.lxi)

            self.b_beta = self._train_b_beta(V=self.V,
                                             Lambda_V_1=self.Lambda_V_1)

            self.G = self._mixing(M, Phi, self.V)
            self.R = self._responsibilities(X=X, y=y, G=self.G)

            L_M_q_prev = self.L_M_q
            self.L_M_q = self._var_bound(G=self.G,
                                         R=self.R,
                                         V=self.V,
                                         Lambda_V_1=self.Lambda_V_1,
                                         a_beta=self.a_beta,
                                         b_beta=self.b_beta)
            # LCSBookCode states: “as we are using an approximation, the variational
            # bound might decrease, so we're not checking and need to take the
            # abs()”. I guess with approximation he means the use of the Laplace
            # approximation (which may violate the lower bound nature of L_M_q).
            delta_L_M_q = np.abs(self.L_M_q - L_M_q_prev)
            # TODO Check whether the abs is necessary for Bouchard.
            # if self.L_M_q < L_M_q_prev:
            #     print(f"self.L_M_q < L_M_q_prev: {self.L_M_q} < {L_M_q_prev}")

    def _train_mix_weights(self, M, X, y, Phi, R, V, a_beta, b_beta, lxi,
                           alpha):
        """
        Training routine for mixing weights based on Bouchard's upper bound.

        :param M: matching matrix (N × K)
        :param X: input matrix (N × D_X)
        :param y: output matrix (N × D_y)
        :param Phi: mixing feature matrix (N × D_V)
        :param R: responsibility matrix (N × K)
        :param V: mixing weight matrix (D_V × K)
        :param a_beta: mixing weight prior parameter (row vector of length K)
        :param b_beta: mixing weight prior parameter (row vector of length K)
        :param lxi, alpha: Parameters of Bouchard's bound

        :returns: mixing weight matrix (D_V × K), mixing weight covariance
            matrix (K D_V × K D_V)
        """
        N, _ = X.shape

        E_beta_beta = a_beta / b_beta

        Lambda_V_1 = [np.zeros((self.D_V, self.D_V))] * self.K
        Lambda_V_1_ = [np.zeros((self.D_V, self.D_V))] * self.K

        for k in range(self.K):
            # TODO Performance: This can be vectorized, e.g.
            # https://gregorygundersen.com/blog/2020/07/17/matmul/
            # Lambda_V_1[k] = (R[:, [k]] * Phi) @ Phi.T
            for n in range(N):
                Lambda_V_1[k] += R[n][k] * lxi[n][k] * np.outer(Phi[n], Phi[n])

            # TODO Performance: Put this and the above assignment together.
            Lambda_V_1[k] *= 2
            Lambda_V_1[k] = Lambda_V_1[k] + E_beta_beta[k] * np.identity(
                Lambda_V_1[k].shape[0])

            t = R[:, [k]] * (1 / 2 - 2 * np.log(M[:, [k]]) * lxi[:, [k]]
                             + alpha * lxi[:, [k]])
            V[:, [k]] = np.linalg.pinv(Lambda_V_1[k]) @ Phi.T @ t

        return V, Lambda_V_1

    def _opt_bouchard(self, M: np.ndarray, Phi: np.ndarray, V, alpha, lxi):
        """
        Updates the parameters of Bouchard's lower bound to their optimal
        values.

        :returns: updates variational parameters λ(ξ), α in-place
        """
        N, _ = Phi.shape

        h = np.log(M) + Phi @ V

        alpha = (1 / 2 *
                 (self.K / 2 - 1) + np.sum(h * lxi, axis=1)) / np.sum(lxi,
                                                                      axis=1)
        alpha = alpha.reshape((N, 1))

        xi = np.abs(alpha - h)

        lxi = 1 / (2 * xi) * (1 / (1 + np.exp(-xi)) - 1 / 2)

        return self.alpha, self.lxi

    def _mixing(self, M: np.ndarray, Phi: np.ndarray, V: np.ndarray):
        """
        [PDF p. 239]

        Is zero wherever a classifier does not match.

        :param M: matching matrix (N × K)
        :param Phi: mixing feature matrix (N × D_V)
        :param V: mixing weight matrix (D_V × K)

        :returns: mixing matrix (N × K)
        """
        # If Phi is phi_standard, this simply broadcasts V to a matrix [V, V, V,
        # …]  of shape (N, D_V).
        G = Phi @ V

        # This quasi never happens (at least for the run I checked it did not).
        # That run also oscillated so this is probably not the source.
        G = np.clip(G, self.EXP_MIN, self.LN_MAX - np.log(self.K))

        G = np.exp(G) * M

        # The sum can be 0 meaning we do 0/0 (== NaN) but we ignore it because
        # it is fixed one line later (this is how Drugowitsch does it).
        # Drugowitsch does, however, also say that: “Usually, this should never
        # happen as only model structures are accepted where [(np.sum(G, 1) >
        # 0).all()]. Nonetheless, this check was added to ensure that even these
        # cases are handled gracefully.”
        with np.errstate(invalid="ignore"):
            G = G / np.sum(G, axis=1)[:, np.newaxis]
        G = np.nan_to_num(G, nan=1 / self.K)
        return G

    def _responsibilities(self, X: np.ndarray, y: np.ndarray, G: np.ndarray):
        """
        [PDF p. 240]

        :param X: input matrix (N × D_X)
        :param y: output matrix (N × D_y)
        :param G: mixing (“gating”) matrix (N × K)
        :param W: classifier weight matrices (list of D_y × D_X)
        :param Lambda_1: classifier covariance matrices (list of D_X × D_X)
        :param a_tau: classifier noise precision parameters
        :param b_tau: classifier noise precision parameters

        :returns: responsibility matrix (N × K)
        """
        N, D_y = y.shape

        # We first create the transpose of R because indexing is easier. We then
        # transpose before multiplying elementwise with G.
        R_T = np.zeros((self.K, N))
        for k in range(self.K):
            cl = self.CLS[k]
            R_T[k] = np.exp(D_y / 2 * (ss.digamma(cl.a_tau) - np.log(cl.b_tau))
                            - 0.5 * (cl.a_tau / cl.b_tau * np.sum(
                                (y - X @ cl.W.T)**2, 1)
                                     + D_y * np.sum(X * (X @ cl.Lambda_1), 1)))
        R = R_T.T * G
        # The sum can be 0 meaning we do 0/0 (== NaN in Python) but we ignore it
        # because it is fixed one line later (this is how Drugowitsch does it).
        with np.errstate(invalid="ignore"):
            R = R / np.sum(R, 1)[:, np.newaxis]
        R = np.nan_to_num(R, nan=0)
        return R

    def _train_b_beta(self, V: np.ndarray, Lambda_V_1: np.ndarray):
        """
        [PDF p. 244]

        :param V: mixing weight matrix (D_V × K)
        :param Lambda_V_1: list of K mixing covariance matrices (D_V × D_V)

        :returns: mixing weight vector prior parameter b_beta
        """
        b_beta = np.zeros(self.K)
        Lambda_V_1_diag = np.array(list(map(np.diag, Lambda_V_1)))
        # TODO Performance: LCSBookCode vectorized this:
        # b[:,1] = b_b + 0.5 * (sum(V * V, 0) + self.cov_Tr)
        for k in range(self.K):
            v_k = V[:, [k]]
            l = k * self.D_V
            u = (k + 1) * self.D_V
            # Not that efficient, I think (but very close to [PDF p. 244]).
            # Lambda_V_1_kk = Lambda_V_1[l:u:1, l:u:1]
            # b_beta[k] = B_BETA + 0.5 * (np.trace(Lambda_V_1_kk) + v_k.T @ v_k)
            # More efficient.
            b_beta[k] = self.B_BETA + 0.5 * (np.sum(Lambda_V_1_diag[l:u:1])
                                             + v_k.T @ v_k)

        return b_beta

    def _var_bound(self, G: np.ndarray, R: np.ndarray, V: np.ndarray,
                   Lambda_V_1: np.ndarray, a_beta: np.ndarray,
                   b_beta: np.ndarray):
        """
        [PDF p. 245]

        :param G: mixing matrix (N × K)
        :param R: responsibilities matrix (N × K)
        :param V: mixing weight matrix (D_V × K)
        :param Lambda_V_1: list of K mixing covariance matrices (D_V × D_V)
        :param a_beta: mixing weight prior parameter (row vector of length K)
        :param b_beta: mixing weight prior parameter (row vector of length K)

        :returns: mixing component L_M(q) of variational bound
        """
        L_M1q = self.K * (-ss.gammaln(self.A_BETA)
                          + self.A_BETA * np.log(self.B_BETA))
        # TODO Performance: LCSBookCode vectorized this
        # TODO Performance: ss.gammaln(a_beta[k]) is constant throughout the
        # loop in the calling function
        L_M3q = self.K * self.D_V
        for k in range(self.K):
            # NOTE this is just the negated form of the update two lines prior?
            L_M1q = L_M1q + ss.gammaln(
                a_beta[k]) - a_beta[k] * np.log(b_beta[k])
            # TODO Vectorize or at least get rid of for loop
            # TODO Maybe cache determinant
            if Lambda_V_1[k].shape == (1, ):
                L_M3q += Lambda_V_1[k]
            else:
                L_M3q += np.linalg.slogdet(Lambda_V_1[k])[1]

        L_M3q /= 2
        # L_M2q is the negative Kullback-Leibler divergence [PDF p. 246].
        #
        # ``responsibilities`` performs a ``nan_to_num(…, nan=0, …)``, so we
        # might divide by 0 here. The intended behaviour is to silently get a
        # NaN that can then be replaced by 0 again (this is how Drugowitsch does
        # it [PDF p.  213]). Drugowitsch expects dividing ``x`` by 0 to result
        # in NaN, however, in Python this is only true for ``x == 0``; for any
        # other ``x`` this instead results in ``inf`` (with sign depending on
        # the sign of x). The two cases also throw different errors (‘invalid
        # value encountered’ for ``x == 0`` and ‘divide by zero’ otherwise).
        #
        # NOTE I don't think the neginf is strictly required but let's be safe.
        with np.errstate(divide="ignore", invalid="ignore"):
            L_M2q = np.sum(
                R * np.nan_to_num(np.log(G / R), nan=0, posinf=0, neginf=0))
        # This fixes(?) some numerical problems.
        if L_M2q > 0 and np.isclose(L_M2q, 0):
            L_M2q = 0
        assert L_M2q <= 0, f"Kullback-Leibler divergence less than zero: {L_M2q}"
        return L_M1q + L_M2q + L_M3q
