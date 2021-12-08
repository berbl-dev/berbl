import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore

from .literal import responsibilities
from .utils import check_phi, matching_matrix


class Mixing:
    """
    Model for the mixing weights of a set of linear regression rules.
    """
    def __init__(self,
                 rules,
                 phi,
                 random_state,
                 A_BETA=10**-2,
                 B_BETA=10**-4,
                 DELTA_S_L_M_Q=10**-2,
                 MAX_ITER_MIXING=40,
                 EXP_MIN=np.log(np.finfo(None).tiny),
                 LN_MAX=np.log(np.finfo(None).max),
                 **kwargs):
        """
        Parameters
        ----------
        rules : list of rule object
            List of rules (which are held fixed during mixing training).
        phi : callable
            Mixing feature function taking input matrices of shape (N, DX) and
            returning mixing feature matrices of shape (n, V). If ``None`` use
            the LCS default of ``phi(x) = 1``.
        random_state : RandomState object
        A_BETA : float
            Scale parameter of mixing weight vector variance prior.
        B_BETA : float
            Shape parameter of mixing weight vector variance prior.
        DELTA_S_L_M_Q : float
            Stopping criterion for variational update loop.
        MAX_ITER_MIXING : int
            Only perform up to this many iterations of variational updates
            (abort then, even if stopping criterion is not yet met).
        EXP_MIN : float
            Lowest real number ``x`` on system such that ``exp(x) > 0``. The
            default is the logarithm of the smallest positive number of the
            default dtype (as of 2020-10-06, this dtype is float64).
        LN_MAX : float
            ``ln(x)``, where ``x`` is the highest real number on the system. The
            default is the logarithm of the highest number of the default dtype
            (as of 2020-10-06, this dtype is float64).
        **kwargs : kwargs
            This is here so that we don't need to repeat all the hyperparameters
            in ``Mixture`` etc. ``Mixture`` simply passes through all
            ``**kwargs`` to both ``Mixing`` and ``Rule``. This means that during
            implementation, we need to be aware that if there are parameters in
            those two classes with the same name, they always receive the same
            value.
        """
        self.rules = rules
        self.phi = phi
        self.A_BETA = A_BETA
        self.B_BETA = B_BETA
        self.DELTA_S_L_M_Q = DELTA_S_L_M_Q
        self.MAX_ITER_MIXING = MAX_ITER_MIXING
        self.EXP_MIN = EXP_MIN
        self.LN_MAX = LN_MAX
        self.random_state = random_state
        self.K = len(self.rules)

    def fit(self, X, y):
        """
        Fits mixing weights for this mixing weight model's set of rules to the
        provided data.
        """
        Phi = check_phi(self.phi, X)

        M = np.hstack([cl.m_ for cl in self.rules])

        _, self.DX_ = X.shape
        _, self.Dy_ = y.shape
        N, self.DV_ = Phi.shape

        self.V_ = self.random_state.normal(loc=0,
                                           scale=self.A_BETA / self.B_BETA,
                                           size=(self.DV_, self.K))
        # a_beta is actually constant so we can set it here and be done with it.
        self.a_beta_ = np.repeat(self.A_BETA + self.DV_ / 2, self.K)
        self.b_beta_ = np.repeat(self.B_BETA, self.K)

        # Initialize parameters for the Bouchard approximation.
        self.alpha_ = self.random_state.normal(loc=0,
                                               scale=self.A_BETA / self.B_BETA,
                                               size=(N, 1))
        # lxi stands for λ(ξ) which is used in Bouchard's approximation. Its
        # supremum value is one over eight.
        self.lxi_ = self.random_state.random(size=(N, self.K)) * 0.125
        self.alpha_, self.lxi_ = self._opt_bouchard(M=M,
                                                    Phi=Phi,
                                                    V=self.V_,
                                                    alpha=self.alpha_,
                                                    lxi=self.lxi_)

        self.G_ = self._mixing(M, Phi, self.V_)
        self.R_ = self._responsibilities(X=X, y=y, G=self.G_)

        self.L_M_q_ = -np.inf
        delta_L_M_q = self.DELTA_S_L_M_Q + 1
        i = 0
        while delta_L_M_q > self.DELTA_S_L_M_Q and i < self.MAX_ITER_MIXING:
            i += 1

            self.V_, self.Lambda_V_1_ = self._train_mix_weights(
                M=M,
                X=X,
                y=y,
                Phi=Phi,
                R=self.R_,
                V=self.V_,
                a_beta=self.a_beta_,
                b_beta=self.b_beta_,
                lxi=self.lxi_,
                alpha=self.alpha_)

            self.alpha_, self.lxi_ = self._opt_bouchard(M=M,
                                                        Phi=Phi,
                                                        V=self.V_,
                                                        alpha=self.alpha_,
                                                        lxi=self.lxi_)

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
            # LCSBookCode states: “as we are using an approximation, the
            # variational bound might decrease, so we're not checking and need
            # to take the abs()”. I guess with approximation he means the use of
            # the Laplace approximation (which may violate the lower bound
            # nature of L_M_q).
            delta_L_M_q = np.abs(self.L_M_q_ - L_M_q_prev)
            # TODO Check whether the abs is necessary for the Bouchard bound.
            # if self.L_M_q < L_M_q_prev:
            #     print(f"self.L_M_q < L_M_q_prev: {self.L_M_q} < {L_M_q_prev}")
            assert np.all(~np.isnan(self.lxi_))

        return self

    def mixing(self, X):
        """
        Calculates the mixing weights for each of the given inputs.

        Parameters
        ----------
        X : array of shape (N, DX)
            Input matrix.

        Returns
        -------
        array of shape (N, K)
            Mixing matrix containing the rules' mixing weights for each input.
        """
        Phi = check_phi(self.phi, X)

        # TODO When predicting (which uses this mixing method), I currently
        # calculate M twice, once when matching for each rule and once in
        # mixing (see same comment in Mixture). Add as an optional parameter to
        # Mixing.predict/fit etc.
        M = matching_matrix([cl.match for cl in self.rules], X)

        return self._mixing(M, Phi, self.V_)

    def _train_mix_weights(self, M, X, y, Phi, R, V, a_beta, b_beta, lxi,
                           alpha):
        """
        Training routine for mixing weights based on Bouchard's upper bound.

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
        R : array of shape (N, K)
            Responsibility matrix.
        V : array of shape (DV, K)
            Mixing weight matrix.
        a_beta : array of shape (K,)
            Mixing weight prior parameter (row vector).
        b_beta : array of shape (K,)
            Mixing weight prior parameter (row vector).
        lxi : array of shape (N, K)
            Parameter of Bouchard's bound.
        alpha : array of shape (N, 1)
            Parameter of Bouchard's bound.

        Returns
        -------
        V, Lambda_V_1 : tuple of array of shapes (DV, K) and list (length K) of arrays of shape (DV, DV)
            Updated mixing weight matrix and mixing weight covariance matrices.
        """
        N, _ = X.shape
        DV, _ = V.shape

        E_beta_beta = a_beta / b_beta

        Lambda_V_1 = [np.zeros((DV, DV))] * self.K

        Rlxi = R * lxi
        for k in range(self.K):
            Lambda_V_1[k] = 2 * (Rlxi[:, [k]].T
                                 * Phi.T) @ Phi + E_beta_beta[k] * np.identity(
                                     Lambda_V_1[k].shape[0])

            t = R[:, [k]] * (1 / 2 - 2 * np.log(M[:, [k]]) * lxi[:, [k]]
                             + alpha * lxi[:, [k]])
            V[:, [k]] = np.linalg.pinv(Lambda_V_1[k]) @ Phi.T @ t

        # NOTE Doing this in-place instead of returning values doesn't seem to
        # result in a significant speedup. We thus opted for the more
        # descriptive alternative.
        return V, Lambda_V_1

    def _opt_bouchard(self, M: np.ndarray, Phi: np.ndarray, V, alpha, lxi):
        """
        Update for the parameters of Bouchard's lower bound.

        Parameters
        ----------
        M : array of shape (N, K)
            Matching matrix.
        Phi : array of shape (N, DV)
            Mixing feature matrix.
        V : array of shape (DV, K)
            Mixing weight matrix.
        alpha : array of shape (N, 1)
            Current value of ``alpha`` variational parameters of Bouchard's
            bound.
        lxi : array of shape (N, K)
            Current value of ``lxi`` variational parameters of Bouchard's bound.

        Returns
        -------
        lxi, alpha : tuple of arrays of shapes (N, 1) and (N, K)
            New values for the variational parameters ``alpha`` and ``lxi``.
        """
        N, _ = Phi.shape

        h = np.log(M) + Phi @ V

        alpha = (1 / 2 *
                 (self.K / 2 - 1) + np.sum(h * lxi, axis=1)) / np.sum(lxi,
                                                                      axis=1)
        alpha = alpha.reshape((N, 1))

        xi = np.abs(alpha - h)

        # If ``alpha == h``, then the following contains a division by zero and
        # then a multiplication by NaN which results in a divide and an invalid
        # value warning to be thrown.
        with np.errstate(divide="ignore", invalid="ignore"):
            lxi = 1 / (2 * xi) * (1 / (1 + np.exp(-xi)) - 1 / 2)
        # Where `alpha == h` we get NaN's in the previous formula due to xi = 0
        # there. We simply solve that by setting the corresponding entries to
        # the limit for `x -> 0` of `lambda(x)` which is `0.125`.
        lxi[np.where(np.logical_and(xi == 0, np.isnan(lxi)))] = 0.125

        # NOTE Doing this in-place instead of returning values doesn't seem to
        # result in a significant speedup. We thus opted for the more
        # descriptive alternative.
        return alpha, lxi

    def _mixing(self, M: np.ndarray, Phi: np.ndarray, V: np.ndarray):
        """
        [PDF p. 239]

        Is zero wherever a rule does not match.

        Parameters
        ----------
        M : array of shape (N, K)
            Matching matrix.
        Phi : array of shape (N, DV)
            Mixing feature matrix.
        V : array of shape (DV, K)
            Mixing weight matrix.

        Returns
        -------
        G : array of shape (N, K)
            Mixing (“gating”) matrix.
        """
        # If Phi is standard, this simply broadcasts V to a matrix [V, V, V, …]
        # of shape (N, DV).
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

        Parameters
        ----------
        X : array of shape (N, DX)
            Input matrix.
        y : array of shape (N, Dy)
            Output matrix.
        G : array of shape (N, K)
            Mixing (“gating”) matrix.

        Returns
        -------
        R : array of shape (N, K)
            Responsibility matrix.
        """
        # NOTE: The code duplication solution used is faster for larger
        # len(self.rules) than the technically cleaner
        # W, Lambda_1, a_tau, b_tau = zip(
        #     *[(cl.W_, cl.Lambda_1_, cl.a_tau_, cl.b_tau_) for cl in self.rules]
        # )
        W = [cl.W_ for cl in self.rules]
        Lambda_1 = [cl.Lambda_1_ for cl in self.rules]
        a_tau = [cl.a_tau_ for cl in self.rules]
        b_tau = [cl.b_tau_ for cl in self.rules]
        return responsibilities(X=X,
                                Y=y,
                                G=G,
                                W=W,
                                Lambda_1=Lambda_1,
                                a_tau=a_tau,
                                b_tau=b_tau)

    def _train_b_beta(self, V: np.ndarray, Lambda_V_1: np.ndarray):
        """
        [PDF p. 244]

        TrainMixPriors but only the part concerned with ``b_beta`` since
        ``a_beta`` is constant.

        Parameters
        ----------
        V : array of shape (DV, K)
            Mixing weight matrix.
        Lambda_V_1 : list (length K) of arrays of shape (DV, DV)
            List of mixing weight covariance matrices.

        Returns
        -------
        b_beta : array of shape (K,)
            mixing weight vector prior parameter
        """
        DV, _ = V.shape
        b_beta = np.repeat(self.B_BETA, (self.K, ))
        Lambda_V_1_diag = np.array(list(map(np.diag, Lambda_V_1)))
        # TODO Performance: LCSBookCode vectorized this:
        # b[:,1] = b_b + 0.5 * (sum(V * V, 0) + self.cov_Tr)
        for k in range(self.K):
            v_k = V[:, [k]]
            l = k * DV
            u = (k + 1) * DV
            # Not that efficient, I think (but very close to [PDF p. 244]).
            # Lambda_V_1_kk = Lambda_V_1[l:u:1, l:u:1]
            # b_beta[k] = B_BETA + 0.5 * (np.trace(Lambda_V_1_kk) + v_k.T @ v_k)
            # More efficient.
            try:
                b_beta[k] += 0.5 * (np.sum(Lambda_V_1_diag[l:u:1])
                                    + v_k.T @ v_k)
            except FloatingPointError as e:
                print(
                    f"FloatingPointError in _train_b_beta "
                    f"(solved by keeping prior b_beta[k] = {b_beta[k]}): "
                    f"v_k = {v_k}, K = {self.K}, V = {V}, Lambda_V_1 = {Lambda_V_1}"
                )

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
        Lambda_V_1 : list (length K) of arrays of shape (DV, DV)
            List of mixing weight covariance matrices.
        a_beta : array of shape (K,)
            Mixing weight prior parameter (row vector).
        b_beta : array of shape (K,)
            Mixing weight prior parameter (row vector).

        Returns
        -------
        L_M_q : float
            Mixing component L_M(q) of variational bound.
        """
        DV, _ = V.shape
        L_M1q = self.K * (-ss.gammaln(self.A_BETA)
                          + self.A_BETA * np.log(self.B_BETA))
        # TODO Performance: LCSBookCode vectorized this
        # TODO Performance: ss.gammaln(a_beta[k]) is constant throughout the
        # loop in the calling function
        L_M3q = self.K * DV
        for k in range(self.K):
            L_M1q += ss.gammaln(a_beta[k]) - a_beta[k] * np.log(b_beta[k])
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
