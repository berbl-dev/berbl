import sys
from typing import *

import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore
import scipy.stats as sstats  # type: ignore
from .model import Model

from ..utils import logstartstop
from .hyperparams import HParams

# Underflows may occur in many places, e.g. if X contains values very close to
# 0.
# TODO Are underflows really OK?
# np.seterr(all="raise", under="ignore")


def model_probability(model: Model, X: np.ndarray, Y: np.ndarray,
                      Phi: np.ndarray, exp_min: float, ln_max: float):
    """
    [PDF p. 235]

    Note that this deviates from [PDF p. 235] in that we return ``L(q) - ln K!``
    instead of ``L(q) + ln K!`` because the latter is not consistent with (7.3).

    We also first augment X by a bias term within this function.

    We also compute the matching matrix within this function instead of
    providing it to it (this way we can return a complete trained model
    including its matching functions instead of just having access to the
    matching information for the given input).

    We also return the model itself and not merely its approximate probability.
    The model contains a field ``p_M_D`` which holds that value.

    :param M: matching matrix (N × K)
    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param Phi: mixing feature matrix (N × D_V)

    :returns: model
    """
    N, _ = X.shape
    K = model.size()

    # Get the matching matrix before we augment X.
    M = model.matching_matrix(X)

    # Augment X by a bias term. [PDF p. 113] assumes that input is always
    # augmented with a single constant element. We simply enforce that here.
    X = np.hstack([np.ones((N, 1)), X])

    W = [None] * K
    Lambda_1 = [None] * K
    a_tau = [None] * K
    b_tau = [None] * K
    a_alpha = [None] * K
    b_alpha = [None] * K
    for k in range(K):
        W[k], Lambda_1[k], a_tau[k], b_tau[k], a_alpha[k], b_alpha[
            k] = train_classifier(M[:, [k]], X, Y)

    V, Lambda_V_1, a_beta, b_beta = train_mixing(M=M,
                                                 X=X,
                                                 Y=Y,
                                                 Phi=Phi,
                                                 W=W,
                                                 Lambda_1=Lambda_1,
                                                 a_tau=a_tau,
                                                 b_tau=b_tau,
                                                 exp_min=exp_min,
                                                 ln_max=ln_max)
    L_q = var_bound(M=M,
                    X=X,
                    Y=Y,
                    Phi=Phi,
                    W=W,
                    Lambda_1=Lambda_1,
                    a_tau=a_tau,
                    b_tau=b_tau,
                    a_alpha=a_alpha,
                    b_alpha=b_alpha,
                    V=V,
                    Lambda_V_1=Lambda_V_1,
                    a_beta=a_beta,
                    b_beta=b_beta)

    ln_p_M = -np.log(float(
        np.math.factorial(K)))  # (7.3), i.e. p_M \propto 1/K
    p_M_D = L_q + ln_p_M

    return model.fitted(p_M_D=p_M_D,
                        W=W,
                        Lambda_1=Lambda_1,
                        a_tau=a_tau,
                        b_tau=b_tau,
                        a_alpha=a_alpha,
                        b_alpha=b_alpha,
                        V=V,
                        Lambda_V_1=Lambda_V_1,
                        a_beta=a_beta,
                        b_beta=b_beta)


def train_classifier(m_k, X, Y):
    """
    :param m_k: matching vector (N)
    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param Phi: mixing feature matrix (N × D_V)

    :returns: approximate model probability L(q) + ln p(M)
    """
    N, D_X = X.shape
    N, D_Y = Y.shape
    X_k = X * np.sqrt(m_k)
    Y_k = Y * np.sqrt(m_k)
    a_alpha_k, b_alpha_k = HParams().A_ALPHA, HParams().B_ALPHA
    a_tau_k, b_tau_k = HParams().A_TAU, HParams().B_TAU
    L_k_q = -np.inf
    delta_L_k_q = HParams().DELTA_S_L_K_Q + 1
    # This is constant; Drugowitsch nevertheless puts it into the while loop
    # (probably for readability).
    a_alpha_k = HParams().A_ALPHA + D_X * D_Y / 2
    # Drugowitsch reaches convergence usually after 3-4 iterations [PDF p. 237].
    i = 0
    while delta_L_k_q > HParams().DELTA_S_L_K_Q and i < HParams().MAX_ITER_CLS:
        i += 1
        # print(f"train_classifier: {delta_L_k_q} > {DELTA_S_L_K_Q}")
        E_alpha_alpha_k = a_alpha_k / b_alpha_k
        Lambda_k = np.diag([E_alpha_alpha_k] * D_X) + X_k.T @ X_k
        Lambda_k_1 = np.linalg.inv(Lambda_k)
        W_k = Y_k.T @ X_k @ Lambda_k_1
        a_tau_k = HParams().A_TAU + 0.5 * np.sum(m_k)
        b_tau_k = HParams().B_TAU + 1 / (2 * D_Y) * (
            np.sum(Y_k * Y_k) - np.sum(W_k * (W_k @ Lambda_k)))
        E_tau_tau_k = a_tau_k / b_tau_k
        # D_Y factor in front of trace due to sum over D_Y elements (7.100).
        b_alpha_k = HParams().B_ALPHA + 0.5 * (E_tau_tau_k * np.sum(W_k * W_k)
                                               + D_Y * np.trace(Lambda_k_1))
        L_k_q_prev = L_k_q
        L_k_q = var_cl_bound(
            X=X,
            Y=Y,
            W_k=W_k,
            Lambda_k_1=Lambda_k_1,
            a_tau_k=a_tau_k,
            b_tau_k=b_tau_k,
            a_alpha_k=a_alpha_k,
            b_alpha_k=b_alpha_k,
            # Substitute r_k by m_k in order to train classifiers independently
            # (see [PDF p. 219]).
            r_k=m_k)
        delta_L_k_q = L_k_q - L_k_q_prev
        # “Each parameter update either increases L_k_q or leaves it unchanged
        # (…). If this is not the case, then the implementation is faulty and/or
        # suffers from numerical instabilities.” [PDF p. 237]
        assert delta_L_k_q >= 0
    return W_k, Lambda_k_1, a_tau_k, b_tau_k, a_alpha_k, b_alpha_k


# TODO Drugowitsch also gives this a_alpha and b_alpha although they are not
# used. Maybe investigate more in-depth whether that is a mistake?
def train_mixing(M: np.ndarray, X: np.ndarray, Y: np.ndarray, Phi: np.ndarray,
                 W: List[np.ndarray], Lambda_1: List[np.ndarray],
                 a_tau: np.ndarray, b_tau: np.ndarray, exp_min: float,
                 ln_max: float):
    """
    [PDF p. 238]

    :param M: matching matrix (N × K)
    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param Phi: mixing feature matrix (N × D_V)
    :param W: classifier weight matrices (list of D_Y × D_X)
    :param Lambda_1: classifier covariance matrices (list of D_X × D_X)
    :param a_tau: classifier noise precision parameters
    :param b_tau: classifier noise precision parameters

    :returns: mixing weight matrix (D_V × K), mixing weight covariance matrix (K
        D_V × K D_V), mixing weight vector prior parameters a_beta/b_beta
    """
    N, K = M.shape
    N, D_X = X.shape
    N, D_Y = Y.shape
    N, D_V = Phi.shape

    V = HParams().random_state.normal(loc=0,
                                      scale=HParams().A_BETA
                                      / HParams().B_BETA,
                                      size=(D_V, K))
    a_beta = np.repeat(HParams().A_BETA, K)
    b_beta = np.repeat(HParams().B_BETA, K)
    L_M_q = -np.inf
    delta_L_M_q = HParams().DELTA_S_L_M_Q + 1
    i = 0
    while delta_L_M_q > HParams().DELTA_S_L_M_Q and i < HParams().MAX_ITER_MIXING:
        i += 1
        # This is not monotonous due to the Laplace approximation used [PDF p.
        # 202, 160]. Also: “This desirable monotonicity property is unlikely to
        # arise with other types of approximation methods, such as the Laplace
        # approximation.” (Bayesian parameter estimation via variational methods
        # (Jaakkola, Jordan), p. 10)
        V, Lambda_V_1 = train_mix_weights(M=M,
                                          X=X,
                                          Y=Y,
                                          Phi=Phi,
                                          W=W,
                                          Lambda_1=Lambda_1,
                                          a_tau=a_tau,
                                          b_tau=b_tau,
                                          V=V,
                                          a_beta=a_beta,
                                          b_beta=b_beta)
        # TODO LCSBookCode only updates b_beta here as a_beta is constant.
        a_beta, b_beta = train_mix_priors(V, Lambda_V_1)
        G = mixing(M, Phi, V)
        R = responsibilities(X=X,
                             Y=Y,
                             G=G,
                             W=W,
                             Lambda_1=Lambda_1,
                             a_tau=a_tau,
                             b_tau=b_tau)
        L_M_q_prev = L_M_q
        L_M_q = var_mix_bound(G=G,
                              R=R,
                              V=V,
                              Lambda_V_1=Lambda_V_1,
                              a_beta=a_beta,
                              b_beta=b_beta)
        # LCSBookCode states: “as we are using an approximation, the variational
        # bound might decrease, so we're not checking and need to take the
        # abs()”. I guess with approximation he means the use of the Laplace
        # approximation (which may violate the lower bound nature of L_M_q).
        delta_L_M_q = np.abs(L_M_q - L_M_q_prev)
    return V, Lambda_V_1, a_beta, b_beta


def mixing(M: np.ndarray, Phi: np.ndarray, V: np.ndarray):
    """
    [PDF p. 239]

    :param M: matching matrix (N × K)
    :param Phi: mixing feature matrix (N × D_V)
    :param V: mixing weight matrix (D_V × K)

    :returns: mixing matrix (N × K)
    """
    D_V, K = V.shape
    G = Phi @ V

    G = np.clip(G, HParams().EXP_MIN, HParams().LN_MAX - np.log(K))

    G = np.exp(G) * M

    # The sum can be 0 meaning we do 0/0 (== NaN) but we ignore it because it is
    # fixed one line later (this is how Drugowitsch does it). Drugowitsch does,
    # however, also say that: “Usually, this should never happen as only model
    # structures are accepted where [(np.sum(G, 1) > 0).all()]. Nonetheless,
    # this check was added to ensure that even these cases are handled
    # gracefully.”
    with np.errstate(invalid="ignore"):
        G = G / np.sum(G, 1)[:, np.newaxis]
    G = np.nan_to_num(G, nan=1 / K)
    return G


def responsibilities(X: np.ndarray, Y: np.ndarray, G: np.ndarray,
                     W: List[np.ndarray], Lambda_1: List[np.ndarray],
                     a_tau: np.ndarray, b_tau: np.ndarray):
    """
    [PDF p. 240]

    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param G: mixing (“gating”) matrix (N × K)
    :param W: classifier weight matrices (list of D_Y × D_X)
    :param Lambda_1: classifier covariance matrices (list of D_X × D_X)
    :param a_tau: classifier noise precision parameters
    :param b_tau: classifier noise precision parameters

    :returns: responsibility matrix (N × K)
    """
    N, K = G.shape
    N, D_Y = Y.shape

    # We first create the transpose of R because indexing is easier. We then
    # transpose before multiplying elementwise with G.
    R_T = np.zeros((K, N))
    for k in range(K):
        R_T[k] = np.exp(D_Y / 2 * (ss.digamma(a_tau[k]) - np.log(b_tau[k]))
                        - 0.5
                        * (a_tau[k] / b_tau[k] * np.sum((Y - X @ W[k].T)**2, 1)
                           + D_Y * np.sum(X * (X @ Lambda_1[k]), 1)))
    R = R_T.T * G
    # The sum can be 0 meaning we do 0/0 (== NaN in Python) but we ignore it
    # because it is fixed one line later (this is how Drugowitsch does it).
    with np.errstate(invalid="ignore"):
        R = R / np.sum(R, 1)[:, np.newaxis]
    R = np.nan_to_num(R, nan=0)
    return R


def train_mix_weights(M: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      Phi: np.ndarray, W: List[np.ndarray],
                      Lambda_1: List[np.ndarray], a_tau: np.ndarray,
                      b_tau: np.ndarray, V: np.ndarray, a_beta: np.ndarray,
                      b_beta: np.ndarray):
    """
    [PDF p. 241]

    :param M: matching matrix (N × K)
    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param Phi: mixing feature matrix (N × D_V)
    :param W: classifier weight matrices (list of D_Y × D_X)
    :param Lambda_1: classifier covariance matrices (list of D_X × D_X)
    :param a_tau: classifier noise precision parameters
    :param b_tau: classifier noise precision parameters
    :param V: mixing weight matrix (D_V × K)
    :param a_beta: mixing weight prior parameter (row vector of length K)
    :param b_beta: mixing weight prior parameter (row vector of length K)

    :returns: mixing weight matrix (D_V × K), mixing weight covariance matrix (K
        D_V × K D_V)
    """
    D_V, K = V.shape

    E_beta_beta = a_beta / b_beta
    # TODO Performance: This can probably be cached (is calculated above)
    G = mixing(M, Phi, V)
    R = responsibilities(X=X,
                         Y=Y,
                         G=G,
                         W=W,
                         Lambda_1=Lambda_1,
                         a_tau=a_tau,
                         b_tau=b_tau)
    # TODO Performance: Why always make TWO steps instead of checking the true
    # KLRG here? The delta has to be close to 0 and that is only the case for
    # two steps where KLRG was not np.inf.
    KLRG = np.inf
    delta_KLRG = HParams().DELTA_S_KLRG + 1
    i = 0
    while delta_KLRG > HParams().DELTA_S_KLRG and i < HParams().MAX_ITER_MIXING:
        i += 1
        # Actually, this should probably be named nabla_E.
        E = Phi.T @ (G - R) + V * E_beta_beta
        e = E.T.reshape((-1))
        H = hessian(Phi=Phi, G=G, a_beta=a_beta, b_beta=b_beta)
        assert np.all(np.linalg.eigvals(H) > 0
                      ), "H is not positive definite although it should be"
        # Preference of `-` and `@` is OK here, we checked.
        delta_v = -np.linalg.inv(H) @ e
        # “D_V × K matrix with jk'th element given by ((k - 1) K + j)'th element
        # of v.” (Probably means “delta_v”.)
        delta_V = delta_v.reshape((K, D_V)).T
        V = V + delta_V
        G = mixing(M, Phi, V)
        R = responsibilities(X=X,
                             Y=Y,
                             G=G,
                             W=W,
                             Lambda_1=Lambda_1,
                             a_tau=a_tau,
                             b_tau=b_tau)
        KLRG_prev = KLRG
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
            # Drugowitsch doesn't add the `-` although it should be there,
            # strictly speaking.
            KLRG = -np.sum(
                R * np.nan_to_num(np.log(G / R), nan=0, posinf=0, neginf=0))
        # This fixes(?) some numerical problems.
        if KLRG < 0 and np.isclose(KLRG, 0):
            print(
                "Warning: Kullback-Leibler divergence ever so slightly less than zero, fixing"
            )
            KLRG = 0
        assert KLRG >= 0, f"Kullback-Leibler divergence less than zero: {KLRG}\n{G}\n{R}"
        delta_KLRG = np.abs(KLRG_prev - KLRG)
    H = hessian(Phi=Phi, G=G, a_beta=a_beta, b_beta=b_beta)
    Lambda_V_1 = np.linalg.inv(H)
    return V, Lambda_V_1


def hessian(Phi: np.ndarray, G: np.ndarray, a_beta: np.ndarray,
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
            Phi * (G[:, [k]] *
                   (1 - G[:, [k]]))) + a_beta[k] / b_beta[k] * np.identity(D_V)
    return H


def train_mix_priors(V: np.ndarray, Lambda_V_1: np.ndarray):
    """
    [PDF p. 244]

    :param V: mixing weight matrix (D_V × K)
    :param Lambda_V_1: mixing covariance matrix (K D_V × K D_V)

    :returns: mixing weight vector prior parameters a_beta, b_beta
    """
    D_V, K = V.shape
    assert Lambda_V_1.shape == (K * D_V, K * D_V)

    a_beta = np.zeros(K)
    b_beta = np.zeros(K)
    Lambda_V_1_diag = np.diag(Lambda_V_1)
    # TODO Performance: LCSBookCode vectorized this:
    # b[:,1] = b_b + 0.5 * (sum(V * V, 0) + self.cov_Tr)
    for k in range(K):
        v_k = V[:, [k]]
        l = k * D_V
        u = (k + 1) * D_V
        # Not that efficient, I think (but very close to [PDF p. 244]).
        # Lambda_V_1_kk = Lambda_V_1[l:u:1, l:u:1]
        # a_beta[k] = A_BETA + D_V / 2
        # b_beta[k] = B_BETA + 0.5 * (np.trace(Lambda_V_1_kk) + v_k.T @ v_k)
        # More efficient.
        # TODO Performance: a_beta is constant, extract from loop (and probably
        # from loop in using function as well)
        a_beta[k] = HParams().A_BETA + D_V / 2
        b_beta[k] = HParams().B_BETA + 0.5 * (np.sum(Lambda_V_1_diag[l:u:1])
                                              + v_k.T @ v_k)

    return a_beta, b_beta


def var_bound(M: np.ndarray, X: np.ndarray, Y: np.ndarray, Phi: np.ndarray,
              W: List[np.ndarray], Lambda_1: List[np.ndarray],
              a_tau: np.ndarray, b_tau: np.ndarray, a_alpha: np.ndarray,
              b_alpha: np.ndarray, V: np.ndarray, Lambda_V_1: np.ndarray,
              a_beta, b_beta):
    """
    [PDF p. 244]

    :param M: matching matrix (N × K)
    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param Phi: mixing feature matrix (N × D_V)
    :param W: classifier weight matrices (list of D_Y × D_X)
    :param Lambda_1: classifier covariance matrices (list of D_X × D_X)
    :param a_tau: classifier noise precision parameters
    :param b_tau: classifier noise precision parameters
    :param a_alpha: weight vector prior parameters
    :param b_alpha: weight vector prior parameters
    :param V: mixing weight matrix (D_V × K)
    :param Lambda_V_1: mixing covariance matrix (K D_V × K D_V)
    :param a_beta: mixing weight prior parameter (row vector of length K)
    :param b_beta: mixing weight prior parameter (row vector of length K)

    :returns: variational bound L(q)
    """
    D_V, K = V.shape
    assert Lambda_V_1.shape == (K * D_V, K * D_V)
    assert a_beta.shape == b_beta.shape
    assert a_beta.shape == (K, )

    G = mixing(M, Phi, V)
    R = responsibilities(X=X,
                         Y=Y,
                         G=G,
                         W=W,
                         Lambda_1=Lambda_1,
                         a_tau=a_tau,
                         b_tau=b_tau)
    L_K_q = 0
    for k in range(K):
        L_K_q = L_K_q + var_cl_bound(X=X,
                                     Y=Y,
                                     W_k=W[k],
                                     Lambda_k_1=Lambda_1[k],
                                     a_tau_k=a_tau[k],
                                     b_tau_k=b_tau[k],
                                     a_alpha_k=a_alpha[k],
                                     b_alpha_k=b_alpha[k],
                                     r_k=R[:, [k]])
    L_M_q = var_mix_bound(G, R, V, Lambda_V_1, a_beta, b_beta)
    return L_K_q + L_M_q


def var_cl_bound(X: np.ndarray, Y: np.ndarray, W_k: np.ndarray,
                 Lambda_k_1: np.ndarray, a_tau_k: float, b_tau_k: float,
                 a_alpha_k: float, b_alpha_k: float, r_k: np.ndarray):
    """
    [PDF p. 245]

    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param W_k: classifier weight matrix (D_Y × D_X)
    :param Lambda_k_1: classifier covariance matrix (D_X × D_X)
    :param a_tau_k: classifier noise precision parameter
    :param b_tau_k: classifier noise precision parameter
    :param a_alpha_k: weight vector prior parameter
    :param b_alpha_k: weight vector prior parameter
    :param r_k: responsibility vector (NumPy row or column vector, we reshape to
        (-1) anyways)

    :returns: classifier component L_k(q) of variational bound
    """
    D_Y, D_X = W_k.shape
    E_tau_tau_k = a_tau_k / b_tau_k
    L_k_1_q = D_Y / 2 * (ss.digamma(a_tau_k) - np.log(b_tau_k)
                         - np.log(2 * np.pi)) * np.sum(r_k)
    # We reshape r_k to a NumPy row vector since NumPy seems to understand what
    # we want to do when we multiply two row vectors (i.e. a^T a).
    L_k_2_q = (-0.5 * r_k).reshape((-1)) @ (E_tau_tau_k * np.sum(
        (Y - X @ W_k.T)**2, 1) + D_Y * np.sum(X * (X @ Lambda_k_1), 1))
    L_k_3_q = -ss.gammaln(HParams().A_ALPHA) + HParams().A_ALPHA * np.log(
        HParams().B_ALPHA) + ss.gammaln(a_alpha_k) - a_alpha_k * np.log(
            b_alpha_k) + D_X * D_Y / 2 + D_Y / 2 * np.log(
                np.linalg.det(Lambda_k_1))
    L_k_4_q = D_Y * (-ss.gammaln(HParams().A_TAU)
                     + HParams().A_TAU * np.log(HParams().B_TAU) +
                     (HParams().A_TAU - a_tau_k) * ss.digamma(a_tau_k)
                     - HParams().A_TAU * np.log(b_tau_k) - HParams().B_TAU
                     * E_tau_tau_k + ss.gammaln(a_tau_k) + a_tau_k)
    return L_k_1_q + L_k_2_q + L_k_3_q + L_k_4_q


def var_mix_bound(G: np.ndarray, R: np.ndarray, V: np.ndarray,
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

    assert G.shape == R.shape
    assert G.shape[1] == K
    assert Lambda_V_1.shape == (K * D_V, K * D_V)
    assert a_beta.shape == (K, )
    assert b_beta.shape == (K, )

    # TODO Semantics: Check var_bound of LCSBookCode (they only multiply K with
    # the first summand)
    L_M1q = K * (-ss.gammaln(HyperParams().A_BETA)
                 + HyperParams().A_BETA * np.log(HyperParams().B_BETA))
    # TODO Performance: LCSBookCode vectorized this
    # TODO Performance: ss.gammaln(a_beta[k]) is constant throughout the loop in the calling
    # function
    for k in range(K):
        # NOTE this is just the negated form of the update two lines prior?
        L_M1q = L_M1q + ss.gammaln(a_beta[k]) - a_beta[k] * np.log(b_beta[k])

    # L_M2q is the Kullback-Leibler divergence [PDF p. 246].
    #
    # ``responsibilities`` performs a ``nan_to_num(…, nan=0, …)``, so we might
    # divide by 0 here. The intended behaviour is to silently get a NaN that can
    # then be replaced by 0 again (this is how Drugowitsch does it [PDF p.
    # 213]). Drugowitsch expects dividing ``x`` by 0 to result in NaN, however,
    # in Python this is only true for ``x == 0``; for any other ``x`` this
    # instead results in ``inf`` (with sign depending on the sign of x). The two
    # cases also throw different errors (‘invalid value encountered’ for ``x ==
    # 0`` and ‘divide by zero’ otherwise).
    #
    # NOTE I don't think the neginf is strictly required but let's be safe.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Drugowitsch doesn't add the `-` although it should be there, strictly
        # speaking.
        L_M2q = -np.sum(
            R * np.nan_to_num(np.log(G / R), nan=0, posinf=0, neginf=0))
    # This fixes(?) some numerical problems.
    if L_M2q < 0 and np.isclose(L_M2q, 0):
        L_M2q = 0
    assert L_M2q >= 0, f"Kullback-Leibler divergence less than zero: {L_M2q}"
    # TODO Performance: slogdet can be cached, is computed more than once
    L_M3q = 0.5 * np.linalg.slogdet(Lambda_V_1)[1] + K * D_V / 2
    return L_M1q + L_M2q + L_M3q
