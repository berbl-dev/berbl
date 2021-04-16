import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from ..common import matching_matrix
from .mixing import Mixing


# NOTE It's not *that* nice to inherit from Mixing because they should be
# siblings and not parent and child.
class MixingLaplace(Mixing):
    """
    Mixing model that is fitted using Drugowitsch's Laplace approximation.

    Structurally, the main difference `Mixing` is that `Lambda_V_1` is a `K *
    D_V` × `K * D_V` matrix (i.e. the mixing problem is solved for all
    classifiers at once), whereas the parent has `K` mixing matrices (one for
    each classifier).
    """
    def __init__(self, DELTA_S_KLRG=10**-8, **kwargs):
        """
        :param DELTA_S_KLRG: Stopping criterion for the iterative Laplace
            approximation in the mixing weight update.
        """
        self.DELTA_S_KLRG = DELTA_S_KLRG
        super().__init__(**kwargs)

    def fit(self, X, y):

        random_state = check_random_state(self.random_state)

        if self.PHI is None:
            Phi = np.ones((len(X), 1))
        else:
            raise NotImplementedError("phi is not None in Mixing")

        M = matching_matrix([cl.match for cl in self.CLS], X)

        _, self.D_X_ = X.shape
        _, self.D_y_ = y.shape
        N, self.D_V_ = Phi.shape

        self.V_ = random_state.normal(loc=0,
                                     scale=self.A_BETA / self.B_BETA,
                                     size=(self.D_V_, self.K))
        # a_beta is actually constant so we can set it here and be done with it.
        self.a_beta_ = np.repeat(self.A_BETA + self.D_V_ / 2, self.K)
        self.b_beta_ = np.repeat(self.B_BETA, self.K)

        self.G_ = self._mixing(M, Phi, self.V_)
        self.R_ = self._responsibilities(X=X, y=y, G=self.G_)

        self.L_M_q_ = -np.inf
        delta_L_M_q = self.DELTA_S_L_M_Q + 1
        i = 0
        while delta_L_M_q > self.DELTA_S_L_M_Q and i < self.MAX_ITER:
            i += 1

            self.V_, self.Lambda_V_1_ = self._train_mix_weights(
                M=M,
                X=X,
                y=y,
                Phi=Phi,
                G=self.G_,
                R=self.R_,
                V=self.V_,
                a_beta=self.a_beta_,
                b_beta=self.b_beta_)

            self.b_beta_ = self._train_b_beta(V=self.V_,
                                              Lambda_V_1=self.Lambda_V_1_)

            self.G_ = self._mixing(M, Phi, self.V_)
            self.R_ = self._responsibilities(X=X, y=y, G=self.G_)

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
        (see Drugowitsch's book).

        :param M: matching matrix (N × K)
        :param X: input matrix (N × D_X)
        :param y: output matrix (N × D_y)
        :param Phi: mixing feature matrix (N × D_V)
        :param R: responsibility matrix (N × K)
        :param V: mixing weight matrix (D_V × K)
        :param a_beta: mixing weight prior parameter (row vector of length K)
        :param b_beta: mixing weight prior parameter (row vector of length K)

        :returns: mixing weight matrix (D_V × K), mixing weight covariance
            matrix (K D_V × K D_V)
        """
        N, _ = X.shape
        D_V, _ = V.shape

        E_beta_beta = a_beta / b_beta

        KLRG = np.inf
        delta_KLRG = self.DELTA_S_KLRG + 1
        i = 0
        KLRGs = np.repeat(-np.inf, 10)
        j = 0
        while delta_KLRG > self.DELTA_S_KLRG and i < self.MAX_ITER:
            i += 1
            # Actually, this should probably be named nabla_E.
            E = Phi.T @ (G - R) + V * E_beta_beta
            e = E.T.ravel()
            H = self.hessian(Phi=Phi, G=G, a_beta=a_beta, b_beta=b_beta)
            # Preference of `-` and `@` is OK here, we checked.
            #
            # While, in theory, H is always invertible here and we thus should
            # be able to use inv (as it is described in the algorithm we
            # implement), we (seldomly) get a singular H, probably due to
            # numerical issues. Thus we simply use pinv which yields the same
            # result as inv anyways if H is non-singular. Also, in his own code,
            # Drugowitsch always uses pseudo inverse here.
            delta_v = -np.linalg.pinv(H) @ e
            # “D_V × K matrix with jk'th element given by ((k - 1) K + j)'th
            # element of v.” (Probably means “delta_v”.)
            delta_V = delta_v.reshape((self.K, D_V)).T
            V = V + delta_V

            G = self._mixing(M, Phi, V)
            R = self._responsibilities(X=X, y=y, G=G)

            KLRG_prev = KLRG
            # ``responsibilities`` performs a ``nan_to_num(…, nan=0, …)``, so we
            # might divide by 0 here. The intended behaviour is to silently get
            # a NaN that can then be replaced by 0 again (this is how
            # Drugowitsch does it [PDF p.  213]). Drugowitsch expects dividing
            # ``x`` by 0 to result in NaN, however, in Python this is only true
            # for ``x == 0``; for any other ``x`` this instead results in
            # ``inf`` (with sign depending on the sign of x). The two cases also
            # throw different errors (‘invalid value encountered’ for ``x == 0``
            # and ‘divide by zero’ otherwise).
            #
            # NOTE I don't think the neginf is strictly required but let's be
            # safe.
            with np.errstate(divide="ignore", invalid="ignore"):
                # Note that KLRG is actually the negative Kullback-Leibler
                # divergence (other than is stated in the book).
                KLRG = np.sum(
                    R
                    * np.nan_to_num(np.log(G / R), nan=0, posinf=0, neginf=0))
            # This fixes(?) some numerical problems.
            if KLRG > 0 and np.isclose(KLRG, 0):
                KLRG = 0
            assert KLRG <= 0, (f"Kullback-Leibler divergence less than zero:"
            f" {KLRG}\n{G}\n{R}")

            if KLRG in KLRGs and KLRG != KLRG_prev:
                # We only log and break when we're at the best KLRG value of the
                # current oscillation.
                if (KLRG <= KLRGs).all():
                    State().oscillation_count += 1
                    mlflow.log_metric("algorithm.oscillations.occurred", 1,
                                      State().step)
                    break

            KLRGs[j] = KLRG
            j = (j + 1) % 10

            delta_KLRG = np.abs(KLRG_prev - KLRG)

        H = self.hessian(Phi=Phi, G=G, a_beta=a_beta, b_beta=b_beta)
        # While, in theory, H is always invertible here and we thus should be
        # able to use inv (as it is described in the algorithm we implement), we
        # (seldomly) get a singular H, probably due to numerical issues. Thus we
        # simply use pinv which yields the same result as inv anyways if H is
        # non-singular. Also, in his own code, Drugowitsch always uses pseudo
        # inverse here.
        Lambda_V_1 = np.linalg.pinv(H)
        # NOTE that instead of returning/storing Lambda_V_1, Drugowitsch's
        # LCSBookCode computes and stores np.slogdet(Lambda_V_1) and cov_Tr (the
        # latter of which is used in his update_gating).
        # NOTE Doing this in-place instead of returning values doesn't seem to
        # result in a significant speedup.
        return V, Lambda_V_1

    def hessian(self, Phi: np.ndarray, G: np.ndarray, a_beta: np.ndarray,
                b_beta: np.ndarray):
        """
        [PDF p. 243]

        :param Phi: mixing feature matrix (N × D_V)
        :param G: mixing matrix (N × K)
        :param a_beta: mixing weight prior parameter (row vector of length K)
        :param b_beta: mixing weight prior parameter (row vector of length K)

        :returns: Hessian matrix (K D_V × K D_V)
        """
        N, D_V = Phi.shape
        K, = a_beta.shape
        assert G.shape == (N, K)
        assert a_beta.shape == b_beta.shape

        H = np.zeros((K * D_V, K * D_V))
        for k in range(K):
            for j in range(k):
                lk = k * D_V
                uk = (k + 1) * D_V
                lj = j * D_V
                uj = (j + 1) * D_V
                H_kj = -Phi.T @ (Phi * (G[:, [k]] * G[:, [j]]))
                H[lk:uk:1, lj:uj:1] = H_kj
                H[lj:uj:1, lk:uk:1] = H_kj
            l = k * D_V
            u = (k + 1) * D_V
            H[l:u:1, l:u:1] = Phi.T @ (
                Phi *
                (G[:, [k]] *
                 (1 - G[:, [k]]))) + a_beta[k] / b_beta[k] * np.identity(D_V)
        return H

    def _var_bound(self, G: np.ndarray, R: np.ndarray, V: np.ndarray,
                   Lambda_V_1: np.ndarray, a_beta: np.ndarray,
                   b_beta: np.ndarray):
        """
        [PDF p. 245]

        :param G: mixing matrix (N × K)
        :param R: responsibilities matrix (N × K)
        :param V: mixing weight matrix (D_V × K)
        :param Lambda_V_1: mixing covariance matrix (K D_V × K D_V)
        :param a_beta: mixing weight prior parameter (row vector of length K)
        :param b_beta: mixing weight prior parameter (row vector of length K)

        :returns: mixing component L_M(q) of variational bound
        """
        D_V, K = V.shape

        L_M1q = K * (-ss.gammaln(self.A_BETA)
                     + self.A_BETA * np.log(self.B_BETA))
        # TODO Performance: LCSBookCode vectorized this
        # TODO Performance: ss.gammaln(a_beta[k]) is constant throughout the
        # loop in the calling function
        for k in range(self.K):
            L_M1q += ss.gammaln(a_beta[k]) - a_beta[k] * np.log(b_beta[k])

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
        # TODO Performance: slogdet can be cached, is computed more than once
        L_M3q = 0.5 * np.linalg.slogdet(Lambda_V_1)[1] + K * D_V / 2
        return L_M1q + L_M2q + L_M3q
