# TODO Performance: Do many things in place instead of allocating new memory
"""
Module implementing the algorithm presented in ‘Design and Analysis of Learning
Classifier Systems – A Probabilistic Approach’ by Jan Drugowitsch.

This implementation intentionially does break with several Python conventions
(e.g. PEP8 regarding variable naming) in order to stay as close as possible to
the formulation of the algorithm in aforementioned work.

The only deviations from the book are:
* ``model_probability`` returns L(q) - ln K! instead of L(q) + ln K! as the
  latter is presumably a typographical error in the book (the corresponding
  formula in Section 7 uses ``-`` as well, which seems to be correct).
* We always use Moore-Penrose pseudo-inverses instead of actual inverses due to
  (very seldomly) matrices being invertible—probably due to numerical
  inaccuracies. This is also done in the code that Jan Drugowitsch published to
  accompany his book: `1
  <https://github.com/jdrugo/LCSBookCode/blob/master/cl.py#L120>`_, `2
  <https://github.com/jdrugo/LCSBookCode/blob/master/cl.py#L385>`_, `3
  <https://github.com/jdrugo/LCSBookCode/blob/master/cl.py#L409>`_.
* Since the IRLS training of the mixing weights sometimes starts to oscillate in
  an infinite loop between several weight values, we add a maximum number of
  iterations to the three main training loops:

  * classifier training (``train_classifier``)
  * mixing model training (``train_mixing``)
  * mixing weight training (``train_mix_weights``)

  This seems reasonable, especially since Jan Drugowitsch's code does the same
  (a behaviour that is *not documented in the book*).

Within the code, comments referring to “LCSBookCode” refer to `Jan Drugowitsch's
code <https://github.com/jdrugo/LCSBookCode>`_.
"""
import os
import sys
from typing import *

import mlflow
import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore
import scipy.stats as sstats  # type: ignore

from ..utils import add_bias
from .hyperparams import HParams
from .model import Model
from .state import State

# Underflows may occur in many places, e.g. if X contains values very close to
# 0. However, they mostly occur in the very first training iterations so they
# should be OK to ignore for now.
np.seterr(all="raise", under="warn")


def model_probability(model: Model, X: np.ndarray, Y: np.ndarray,
                      Phi: np.ndarray):
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
    X = add_bias(X)

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
                                                 b_tau=b_tau)
    L_q, L_k_q, L_M_q = var_bound(M=M,
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
                        b_beta=b_beta,
                        L_q=L_q,
                        ln_p_M=ln_p_M,
                        L_k_q=L_k_q,
                        L_M_q=L_M_q)


def train_classifier(m_k, X, Y):
    """
    :param m_k: matching vector (N)
    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param Phi: mixing feature matrix (N × D_V)

    :returns: weight matrix (D_Y × D_X), covariance matrix (D_X × D_X), two
        noise precision parameters, two weight vector parameters
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
        # While, in theory, Lambda_k is always invertible here and we thus
        # should be able to use inv (as it is described in the algorithm we
        # implement), we (seldomly) get a singular matrix, probably due to
        # numerical issues. Thus we simply use pinv which yields the same result
        # as inv anyways if H is non-singular. Also, in his own code,
        # Drugowitsch always uses pseudo inverse here.
        Lambda_k_1 = np.linalg.pinv(Lambda_k)
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


def train_mixing(M: np.ndarray, X: np.ndarray, Y: np.ndarray,
                          Phi: np.ndarray, W: List[np.ndarray],
                          Lambda_1: List[np.ndarray], a_tau: np.ndarray,
                          b_tau: np.ndarray):
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

    V = State().random_state.normal(loc=0,
                                    scale=HParams().A_BETA / HParams().B_BETA,
                                    size=(D_V, K))

    alpha = State().random_state.normal(loc=0,
                                        scale=HParams().A_BETA
                                        / HParams().B_BETA,
                                        size=(N, 1))
    xi = State().random_state.random(size=(N, K))
    lambda_xi = 1 / (2 * xi) * (1 / (1 + np.exp(xi) - 1 / 2))
    lambda_xi, alpha = opt_bouchard(M, Phi, V, lambda_xi)

    a_beta = np.repeat(HParams().A_BETA, K)
    b_beta = np.repeat(HParams().B_BETA, K)
    L_M_q = -np.inf
    delta_L_M_q = HParams().DELTA_S_L_M_Q + 1
    i = 0
    while delta_L_M_q > HParams().DELTA_S_L_M_Q and i < HParams(
    ).MAX_ITER_MIXING:
        i += 1
        V, Lambda_V_1 = train_mix_weights_bouchard(M=M,
                                                   X=X,
                                                   Y=Y,
                                                   Phi=Phi,
                                                   W=W,
                                                   Lambda_1=Lambda_1,
                                                   a_tau=a_tau,
                                                   b_tau=b_tau,
                                                   V=V,
                                                   a_beta=a_beta,
                                                   b_beta=b_beta,
                                                   lambda_xi=lambda_xi,
                                                   alpha=alpha)
        lambda_xi, alpha = opt_bouchard(M, Phi, V, lambda_xi)
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
        # TODO Check whether the abs is necessary for Bouchard.
        delta_L_M_q = np.abs(L_M_q - L_M_q_prev)
    return V, Lambda_V_1, a_beta, b_beta


def mixing(M: np.ndarray, Phi: np.ndarray, V: np.ndarray):
    """
    [PDF p. 239]

    Is zero wherever a classifier does not match.

    :param M: matching matrix (N × K)
    :param Phi: mixing feature matrix (N × D_V)
    :param V: mixing weight matrix (D_V × K)

    :returns: mixing matrix (N × K)
    """
    D_V, K = V.shape
    # If Phi is standard, this simply broadcasts V to a matrix [V, V, V, …] of
    # shape (N, D_V).
    G = Phi @ V

    # This quasi never happens (at least for the run I checked it did not). That
    # run also oscillated so this is probably not the source.
    G = np.clip(G, HParams().EXP_MIN, HParams().LN_MAX - np.log(K))

    G = np.exp(G) * M

    # The sum can be 0 meaning we do 0/0 (== NaN) but we ignore it because it is
    # fixed one line later (this is how Drugowitsch does it). Drugowitsch does,
    # however, also say that: “Usually, this should never happen as only model
    # structures are accepted where [(np.sum(G, 1) > 0).all()]. Nonetheless,
    # this check was added to ensure that even these cases are handled
    # gracefully.”
    with np.errstate(invalid="ignore"):
        G = G / np.sum(G, axis=1)[:, np.newaxis]
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


def train_mix_weights_bouchard(M: np.ndarray, X: np.ndarray, Y: np.ndarray,
                               Phi: np.ndarray, W: List[np.ndarray],
                               Lambda_1: List[np.ndarray], a_tau: np.ndarray,
                               b_tau: np.ndarray, V: np.ndarray,
                               a_beta: np.ndarray, b_beta: np.ndarray,
                               lambda_xi: np.ndarray, alpha: np.ndarray):
    """
    Potentially more efficient version of ``train_mix_weights``.

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
    N, _ = X.shape
    D_V, K = V.shape

    # TODO Should probably cache G and R
    G = mixing(M, Phi, V)
    R = responsibilities(X=X,
                         Y=Y,
                         G=G,
                         W=W,
                         Lambda_1=Lambda_1,
                         a_tau=a_tau,
                         b_tau=b_tau)

    E_beta_beta = a_beta / b_beta

    Lambda_V_1 = [np.zeros((D_V, D_V))] * K

    for k in range(K):
        # TODO Performance: Vectorize this
        for n in range(N):
            Lambda_V_1[k] += R[n][k] * lambda_xi[n][k] * np.outer(
                Phi[n], Phi[n])

        Lambda_V_1[k] *= 2
        Lambda_V_1[k] = Lambda_V_1[k] + E_beta_beta[k] * np.identity(
            Lambda_V_1[k].shape[0])

        t = R[:, [k]] * (1 / 2 - 2 * np.log(M[:, [k]]) * lambda_xi[:, [k]]
                         + alpha * lambda_xi[:, [k]])
        V[:, [k]] = np.linalg.pinv(Lambda_V_1[k]) @ Phi.T @ t

    return V, Lambda_V_1


def opt_bouchard(M: np.ndarray, Phi: np.ndarray, V: np.ndarray,
                 lambda_xi: np.ndarray):
    """
    Updates the parameters of Bouchard's lower bound to their optimal values.

    :returns: variational parameters λ(ξ), α
    """
    _, K = V.shape
    N, D_V = Phi.shape

    h = np.log(M) + Phi @ V

    # TODO We maybe can get rid of one of the sums?
    alpha = (1 / 2 *
             (K / 2 - 1) + np.sum(h * lambda_xi, axis=1)) / np.sum(lambda_xi,
                                                                   axis=1)
    alpha = alpha.reshape((N, 1))

    xi = np.abs(alpha - h)

    lambda_xi = 1 / (2 * xi) * (1 / (1 + np.exp(-xi)) - 1 / 2)

    return lambda_xi, alpha


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
    mlflow.log_metric("algorithm.oscillations.occurred", 0, State().step)
    KLRGs = np.repeat(-np.inf, 10)
    j = 0
    while delta_KLRG > HParams().DELTA_S_KLRG and i < HParams(
    ).MAX_ITER_MIXING:
        i += 1
        # Actually, this should probably be named nabla_E.
        E = Phi.T @ (G - R) + V * E_beta_beta
        e = E.T.ravel()
        H = hessian(Phi=Phi, G=G, a_beta=a_beta, b_beta=b_beta)
        # Preference of `-` and `@` is OK here, we checked.
        # While, in theory, H is always invertible here and we thus should be able
        # to use inv (as it is described in the algorithm we implement), we
        # (seldomly) get a singular H, probably due to numerical issues. Thus we
        # simply use pinv which yields the same result as inv anyways if H is
        # non-singular. Also, in his own code, Drugowitsch always uses pseudo
        # inverse here.
        delta_v = -np.linalg.pinv(H) @ e
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
            # Note that KLRG is actually the negative Kullback-Leibler
            # divergence (other than is stated in the book).
            KLRG = np.sum(
                R * np.nan_to_num(np.log(G / R), nan=0, posinf=0, neginf=0))
        # This fixes(?) some numerical problems.
        if KLRG > 0 and np.isclose(KLRG, 0):
            print(
                "Warning: Kullback-Leibler divergence ever so slightly less than zero, fixing"
            )
            KLRG = 0
        assert KLRG <= 0, f"Kullback-Leibler divergence less than zero: {KLRG}\n{G}\n{R}"

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

    H = hessian(Phi=Phi, G=G, a_beta=a_beta, b_beta=b_beta)
    # While, in theory, H is always invertible here and we thus should be able
    # to use inv (as it is described in the algorithm we implement), we
    # (seldomly) get a singular H, probably due to numerical issues. Thus we
    # simply use pinv which yields the same result as inv anyways if H is
    # non-singular. Also, in his own code, Drugowitsch always uses pseudo
    # inverse here.
    Lambda_V_1 = np.linalg.pinv(H)
    # Note that instead of returning/storing Lambda_V_1, Drugowitsch's
    # LCSBookCode computes and stores np.slogdet(Lambda_V_1) and cov_Tr (the
    # latter of which is used in his update_gating).
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
    :param Lambda_V_1: list of K mixing covariance matrices (D_V × D_V)

    :returns: mixing weight vector prior parameters a_beta, b_beta
    """
    D_V, K = V.shape

    a_beta = np.zeros(K)
    b_beta = np.zeros(K)
    Lambda_V_1_diag = np.array(list(map(np.diag, Lambda_V_1)))
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
    return L_K_q + L_M_q, L_K_q, L_M_q


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
    :param Lambda_V_1: list of K mixing covariance matrices (D_V × D_V)
    :param a_beta: mixing weight prior parameter (row vector of length K)
    :param b_beta: mixing weight prior parameter (row vector of length K)

    :returns: mixing component L_M(q) of variational bound
    """
    D_V, K = V.shape

    assert G.shape == R.shape
    assert G.shape[1] == K
    assert a_beta.shape == (K, )
    assert b_beta.shape == (K, )

    L_M1q = K * (-ss.gammaln(HParams().A_BETA)
                 + HParams().A_BETA * np.log(HParams().B_BETA))
    # TODO Performance: LCSBookCode vectorized this
    # TODO Performance: ss.gammaln(a_beta[k]) is constant throughout the loop in the calling
    # function
    L_M3q = K * D_V
    for k in range(K):
        # NOTE this is just the negated form of the update two lines prior?
        L_M1q = L_M1q + ss.gammaln(a_beta[k]) - a_beta[k] * np.log(b_beta[k])
        # TODO Vectorize or at least get rid of for loop
        # TODO Maybe cache determinant
        if Lambda_V_1[k].shape == (1, ):
            L_M3q += Lambda_V_1[k]
        else:
            L_M3q += np.linalg.slogdet(Lambda_V_1[k])[1]

    L_M3q /= 2
    # L_M2q is the negative Kullback-Leibler divergence [PDF p. 246].
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
        L_M2q = np.sum(
            R * np.nan_to_num(np.log(G / R), nan=0, posinf=0, neginf=0))
    # This fixes(?) some numerical problems.
    if L_M2q > 0 and np.isclose(L_M2q, 0):
        L_M2q = 0
    assert L_M2q <= 0, f"Kullback-Leibler divergence less than zero: {L_M2q}"
    return L_M1q + L_M2q + L_M3q
