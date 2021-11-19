"""
Module implementing the algorithm presented in ‘Design and Analysis of Learning
Classifier Systems – A Probabilistic Approach’ by Jan Drugowitsch.

This implementation intentionally breaks with several Python conventions (e.g.
PEP8 regarding variable naming) in order to stay as close as possible to the
formulation of the algorithm in aforementioned work.

Also, other than in the remainder of this implementation, we do not as strictly
avoid the overloaded term “classifier” within this module since it is sometimes
used by the original algorithms.

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

  * submodel training (``train_classifier``)
  * mixing model training (``train_mixing``)
  * mixing weight training (``train_mix_weights``)

  This seems reasonable, especially since Jan Drugowitsch's code does the same
  (a behaviour that is *not documented in the book*).
* Since the oscillations in ``train_mix_weights`` are (at least sometimes)
  caused by the Kullback-Leibler divergence between ``G`` and ``R`` being
  optimal followed by another unnecessary execution of the loop thereafter we
  also abort if that is the case (i.e. if the Kullback-Leibler divergence is
  zero).
* We have deal with numerical issues in a few places (e.g. in
  ``train_mix_priors``, ``responsibilities``).

Within the code, comments referring to “LCSBookCode” refer to `Jan Drugowitsch's
code <https://github.com/jdrugo/LCSBookCode>`_.
"""
from typing import *

import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore
import scipy.stats as sstats  # type: ignore
import mlflow  # type: ignore

from ..utils import known_issue, matching_matrix
from .hyperparams import HParams


def model_probability(matchs: List,
                      X: np.ndarray,
                      Y: np.ndarray,
                      Phi: np.ndarray,
                      random_state: np.random.RandomState,
                      exp_min: float = np.log(np.finfo(None).tiny),
                      ln_max: float = np.log(np.finfo(None).max)):
    """
    [PDF p. 235]

    Note that this deviates from [PDF p. 235] in that we return ``p(M | D) =
    L(q) - ln K!`` instead of ``L(q) + ln K!`` because the latter is not
    consistent with (7.3).

    We also compute the matching matrix *within* this function instead of
    providing it to it.

    Parameters
    ----------
    M : array of shape (N, K)
        Matching matrix.
    X : array of shape (N, DX)
        Input matrix.
    Y : array of shape (N, DY)
        Output matrix.
    Phi : array of shape (N, DV)
        Mixing feature matrix.

    Returns
    -------
    metrics, params : pair of dict
        Model metrics and model parameters.
    """

    # Underflows may occur in many places, e.g. if X contains values very close to
    # 0. However, they mostly occur in the very first training iterations so they
    # should be OK to ignore for now. We want to stop (for now) if any other
    # floating point error occurs, though.
    with np.errstate(all="raise", under="ignore"):

        N, _ = X.shape
        K = len(matchs)

        M = matching_matrix(matchs, X)

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
                                                     ln_max=ln_max,
                                                     random_state=random_state)
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

        ln_p_M = -np.log(float(np.math.factorial(K)))
        p_M_D = L_q + ln_p_M

        return {
            "p_M_D": p_M_D,
            "L_q": L_q,
            "ln_p_M": ln_p_M,
            "L_k_q": L_k_q,
            "L_M_q": L_M_q
        }, {
            "matchs": matchs,
            "W": W,
            "Lambda_1": Lambda_1,
            "a_tau": a_tau,
            "b_tau": b_tau,
            "a_alpha": a_alpha,
            "b_alpha": b_alpha,
            "V": V,
            "Lambda_V_1": Lambda_V_1,
            "a_beta": a_beta,
            "b_beta": b_beta,
        }


def train_classifier(m_k, X, Y):
    """
    [PDF p. 238]

    Parameters
    ----------
    m_k : array of shape (N,)
        Matching vector of rule k.
    X : array of shape (N, DX)
        Input matrix.
    Y : array of shape (N, DY)
        Output matrix.

    Returns
    -------
    W_k, Lambda_k_1, a_tau_k, b_tau_k, a_alpha_k, b_alpha_k : arrays of shapes (DY, DX) and (DX, DX) and float
        Weight matrix (DY × DX), covariance matrix (DX × DX), noise precision
        parameters, weight vector parameters.
    """
    N, DX = X.shape
    N, DY = Y.shape
    X_k = X * np.sqrt(m_k)
    Y_k = Y * np.sqrt(m_k)
    a_alpha_k, b_alpha_k = HParams().A_ALPHA, HParams().B_ALPHA
    a_tau_k, b_tau_k = HParams().A_TAU, HParams().B_TAU
    L_k_q = -np.inf
    delta_L_k_q = HParams().DELTA_S_L_K_Q + 1
    # Drugowitsch reaches convergence usually after 3-4 iterations [PDF p. 237].
    # NOTE Deviation from the original text (but not from LCSBookCode) since we
    # add a maximum number of iterations (see module doc string).
    i = 0
    while delta_L_k_q > HParams().DELTA_S_L_K_Q and i < HParams().MAX_ITER_CLS:
        i += 1
        E_alpha_alpha_k = a_alpha_k / b_alpha_k
        Lambda_k = np.diag([E_alpha_alpha_k] * DX) + X_k.T @ X_k
        # While, in theory, Lambda_k is always invertible here and we thus
        # should be able to use inv (as it is described in the algorithm we
        # implement), we (seldomly) get a singular matrix, probably due to
        # numerical issues. Thus we simply use pinv which yields the same result
        # as inv anyways if H is non-singular. Also, in his own code,
        # Drugowitsch always uses pseudo inverse here.
        Lambda_k_1 = np.linalg.pinv(Lambda_k)
        W_k = Y_k.T @ X_k @ Lambda_k_1
        a_tau_k = HParams().A_TAU + 0.5 * np.sum(m_k)
        b_tau_k = HParams().B_TAU + 1 / (2 * DY) * (np.sum(Y_k * Y_k)
                                                    - np.sum(W_k *
                                                             (W_k @ Lambda_k)))
        E_tau_tau_k = a_tau_k / b_tau_k
        a_alpha_k = HParams().A_ALPHA + DX * DY / 2
        # DY factor in front of trace due to sum over DY elements (7.100).
        b_alpha_k = HParams().B_ALPHA + 0.5 * (E_tau_tau_k * np.sum(W_k * W_k)
                                               + DY * np.trace(Lambda_k_1))
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
            # Substitute r_k by m_k in order to train submodels independently
            # (see [PDF p. 219]).
            r_k=m_k)
        delta_L_k_q = L_k_q - L_k_q_prev
        # “Each parameter update either increases L_k_q or leaves it unchanged
        # (…). If this is not the case, then the implementation is faulty and/or
        # suffers from numerical instabilities.” [PDF p. 237]
        # TODO Consider extracting this to a test
        assert delta_L_k_q >= 0 or np.isclose(delta_L_k_q, 0), (
            "Error: L_k(q) not increasing monotonically! "
            "Often caused by data not being standardized "
            "(both X and y should be standardized). "
            f"Iteration: {i}; Δ L_k(q) = {delta_L_k_q}; L_k(q) = {L_k_q}.")
    return W_k, Lambda_k_1, a_tau_k, b_tau_k, a_alpha_k, b_alpha_k


# NOTE Drugowitsch also gives a_alpha and b_alpha as parameters here although
# they are not used.
def train_mixing(M: np.ndarray, X: np.ndarray, Y: np.ndarray, Phi: np.ndarray,
                 W: List[np.ndarray], Lambda_1: List[np.ndarray],
                 a_tau: np.ndarray, b_tau: np.ndarray, exp_min: float,
                 ln_max: float, random_state: np.random.RandomState):
    """
    [PDF p. 238]

    Parameters
    ----------
    M : array of shape (N, K)
        Matching matrix.
    X : array of shape (N, DX)
        Input matrix.
    Y : array of shape (N, DY)
        Output matrix.
    Phi : array of shape (N, DV)
        Mixing feature matrix.
    W : list (length K) of arrays of shape (DY, DX)
        Submodel weight matrices.
    Lambda_1 : list (length K) of arrays of shape (DX, DX)
        Submodel covariance matrices.
    a_tau : array of shape (K,)
        Submodel noise precision parameter.
    b_tau : array of shape (K,)
        Submodel noise precision parameter.

    Returns
    -------
    V, Lambda_V_1, a_beta, b_beta : tuple of arrays of shapes (DV, K), (K * DV, K * DV), (K,) and (K,)
        Mixing weight matrix, mixing weight covariance matrix, mixing weight
        prior parameter vectors.
    """
    N, K = M.shape
    N, DX = X.shape
    N, DY = Y.shape
    N, DV = Phi.shape

    # NOTE LCSBookCode initializes this with np.ones(…).
    V = random_state.normal(loc=0,
                            scale=HParams().A_BETA / HParams().B_BETA,
                            size=(DV, K))
    a_beta = np.repeat(HParams().A_BETA, K)
    b_beta = np.repeat(HParams().B_BETA, K)
    L_M_q = -np.inf
    delta_L_M_q = HParams().DELTA_S_L_M_Q + 1
    # NOTE Deviation from the original text (but not from LCSBookCode) since we
    # add a maximum number of iterations (see module doc string).
    i = 0
    while delta_L_M_q > HParams().DELTA_S_L_M_Q and i < HParams(
    ).MAX_ITER_MIXING:
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
        # NOTE LCSBookCode only updates b_beta here as a_beta is constant.
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
        try:
            # LCSBookCode states: “as we are using a [Laplace] approximation,
            # the variational bound might decrease, so we're not checking and
            # need to take the abs()”.
            delta_L_M_q = np.abs(L_M_q - L_M_q_prev)
        except FloatingPointError as e:
            # ``L_M_q`` and ``L_M_q_prev`` are sometimes ``-inf`` which results
            # in a FloatingPointError (as a nan is generated from ``-inf -
            # (-inf)``).
            #
            # However, ``delta_L_M_q`` being ``nan`` makes the loop abort
            # anyway, so we should be fine. We'll log to stdout and mlflow that
            # this happened, anyway.
            known_issue("FloatingPointError in train_mixing",
                        f"L_M_q: {L_M_q}, L_M_q_prev: {L_M_q_prev}")
            mlflow.set_tag("FloatingPointError delta_L_M_q", "occurred")

    return V, Lambda_V_1, a_beta, b_beta


def mixing(M: np.ndarray, Phi: np.ndarray, V: np.ndarray):
    """
    [PDF p. 239]

    Is zero wherever a rule does not match.

    :param M: matching matrix (N × K)
    :param Phi: mixing feature matrix (N × DV)
    :param V: mixing weight matrix (DV × K)

    :returns: mixing matrix (N × K)
    """
    DV, K = V.shape
    # If Phi is standard, this simply broadcasts V to a matrix [V, V, V, …] of
    # shape (N, DV).
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
        G = G / np.sum(G, axis=1)[:, np.newaxis]
    G = np.nan_to_num(G, nan=1 / K)
    return G


def responsibilities(X: np.ndarray, Y: np.ndarray, G: np.ndarray,
                     W: List[np.ndarray], Lambda_1: List[np.ndarray],
                     a_tau: np.ndarray, b_tau: np.ndarray):
    """
    [PDF p. 240]

    :param X: input matrix (N × DX)
    :param Y: output matrix (N × DY)
    :param G: mixing (“gating”) matrix (N × K)
    :param W: submodel weight matrices (list of DY × DX)
    :param Lambda_1: submodel covariance matrices (list of DX × DX)
    :param a_tau: submodel noise precision parameters
    :param b_tau: submodel noise precision parameters

    :returns: responsibility matrix (N × K)
    """
    N, K = G.shape
    N, DY = Y.shape

    # This fixes instabilities that occur if there is only a single rule.
    if K == 1:
        return np.ones((N, K))

    # We first create the transpose of R because indexing is easier (assigning
    # to rows instead of columns is rather straightforward). We then transpose
    # before multiplying elementwise with G.
    R_T = np.zeros((K, N))
    for k in range(K):
        R_T[k] = np.exp(DY / 2 * (ss.digamma(a_tau[k]) - np.log(b_tau[k]))
                        - 0.5
                        * (a_tau[k] / b_tau[k] * np.sum((Y - X @ W[k].T)**2, 1)
                           + DY * np.sum(X * (X @ Lambda_1[k]), 1)))
    R = R_T.T * G
    # Make a copy of the reference for checking for nans a few lines later.
    R_ = R
    # The sum can be 0 meaning we do 0/0 (== NaN in Python) but we ignore it
    # because it is fixed one line later (this is how Drugowitsch does it).
    with np.errstate(invalid="ignore"):
        R = R / np.sum(R, 1)[:, np.newaxis]
    # This is safer than Drugowitsch's plain `R = np.nan_to_num(R, nan=0)`
    # (i.e. we check whether the nan really came from the cause described above
    # at the cost of an additional run over R to check for zeroes).
    R[np.where(np.logical_and(R_ == 0, np.isnan(R)))] = 0
    return R


def train_mix_weights(M: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      Phi: np.ndarray, W: List[np.ndarray],
                      Lambda_1: List[np.ndarray], a_tau: np.ndarray,
                      b_tau: np.ndarray, V: np.ndarray, a_beta: np.ndarray,
                      b_beta: np.ndarray):
    """
    Training routine for mixing weights based on a Laplace approximation
    (see Drugowitsch's book [PDF p. 241]).

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
    W : list (length K) of arrays of shape (DY, DX)
        Submodel weight matrices.
    Lambda_1 : list (length K) of arrays of shape (DX, DX)
        Submodel covariance matrices.
    a_tau : array of shape (K,)
        Submodel noise precision parameter.
    b_tau : array of shape (K,)
        Submodel noise precision parameter.
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
    V, Lambda_V_1 : tuple of arrays of shapes (DV, K) and (K * DV, K * DV)
        Updated mixing weight matrix and mixing weight covariance matrix.
    """
    DV, K = V.shape

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
    # NOTE Deviation from the original text since we do not always perform at
    # least TWO runs of the loop (Drugowitsch sets KLRG = np.inf). We instead
    # use its real value which solves a certain numerical issue that occurrs
    # very seldomly: if the actual KLRG is almost 0 and the loop is forced to
    # execute (which is the case for KLRG = np.inf) then V becomes unstable and
    # very large.
    KLRG = _kl(R, G)
    delta_KLRG = HParams().DELTA_S_KLRG + 1
    # NOTE Deviation from the original text since we add a maximum number of
    # iterations (see module doc string). In addition, we abort as soon as KLRG
    # is 0 (and do not wait until it was 0 twice, as would be the case if we
    # used the delta-based test only) since numerical issues arise in the loop
    # which then leads to endless oscillations between bad values, one of which
    # is then used in the remainder of the algorithm causing further bad things
    # to happen.
    i = 0
    # TODO Check whether we can get rid of MAX_ITER_MIXING if we check for KLRG
    # being close to zero.
    while delta_KLRG > HParams().DELTA_S_KLRG and i < HParams(
    ).MAX_ITER_MIXING and not np.isclose(KLRG, 0):
        i += 1
        # Actually, this should probably be named nabla_E.
        E = Phi.T @ (G - R) + V * E_beta_beta
        e = E.T.ravel()
        H = hessian(Phi=Phi, G=G, a_beta=a_beta, b_beta=b_beta)
        # While, in theory, H is always invertible here and we thus should be able
        # to use inv (as it is described in the algorithm we implement), we
        # (seldomly) get a singular H, probably due to numerical issues. Thus we
        # simply use pinv which yields the same result as inv anyways if H is
        # non-singular. Also, in his own code, Drugowitsch always uses pseudo
        # inverse here.
        delta_v = -np.linalg.pinv(H) @ e
        # “DV × K matrix with jk'th element given by ((k - 1) K + j)'th element
        # of v.” (Probably means “delta_v”.)
        delta_V = delta_v.reshape((K, DV)).T
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
        KLRG = _kl(R, G)

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


def _kl(R, G):
    """
    Computes the negative Kullback-Leibler divergence between the given arrays.

    Drugowitsch does not introduce this subroutine. We do so to reduce code
    duplication in ``train_mix_weights`` (where we deviated from the original
    text by one additional calculation of ``_kl(R, G)``).
    """
    # ``responsibilities`` performs a ``nan_to_num(…, nan=0, …)``, so we
    # might divide by 0 here due to 0's in ``R``. The intended behaviour is to
    # silently get a NaN that can then be replaced by 0 again (this is how
    # Drugowitsch does it [PDF p.  213]). Drugowitsch expects dividing ``x`` by
    # 0 to result in NaN, however, in Python this is only true for ``x == 0``;
    # for any other ``x`` this instead results in ``inf`` (with sign depending
    # on the sign of x). The two cases also throw different errors (‘invalid
    # value encountered’ for ``x == 0`` and ‘divide by zero’ otherwise).
    #
    # NOTE I don't think the neginf is strictly required but let's be safe.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Note that KLRG is actually the negative Kullback-Leibler
        # divergence (other than is stated in the book).
        KLRG = np.sum(
            R * np.nan_to_num(np.log(G / R), nan=0, posinf=0, neginf=0))
        # TODO Is it really correct to map inf to 0 here?
    # This fixes(?) some numerical problems. Note that KLRG is actually the
    # negative Kullback-Leibler divergence (hence the > 0 comparison instead
    # of < 0).
    if KLRG > 0 and np.isclose(KLRG, 0):
        KLRG = 0
    assert KLRG <= 0, (f"Kullback-Leibler divergence less than zero: "
                        f"KLRG = {-KLRG}")
    return KLRG


def hessian(Phi: np.ndarray, G: np.ndarray, a_beta: np.ndarray,
            b_beta: np.ndarray):
    """
    [PDF p. 243]

    Parameters
    ----------
    Phi : array of shape (N, DV)
        Mixing feature matrix.
    G : array of shape (N, K)
        Mixing (“gating”) matrix.
    a_beta : array of shape (K,)
        Mixing weight prior parameter (row vector).
    b_beta : array of shape (K,)
        Mixing weight prior parameter (row vector).

    Returns
    -------
    array of shape (K * DV, K * DV)
        Hessian matrix.
    """
    N, DV = Phi.shape
    K, = a_beta.shape
    assert G.shape == (N, K)
    assert a_beta.shape == b_beta.shape

    H = np.zeros((K * DV, K * DV))
    for k in range(K):
        for j in range(k):
            lk = k * DV
            uk = (k + 1) * DV
            lj = j * DV
            uj = (j + 1) * DV
            H_kj = -Phi.T @ (Phi * (G[:, [k]] * G[:, [j]]))
            H[lk:uk:1, lj:uj:1] = H_kj
            H[lj:uj:1, lk:uk:1] = H_kj
        l = k * DV
        u = (k + 1) * DV
        H[l:u:1, l:u:1] = Phi.T @ (
            Phi * (G[:, [k]] *
                   (1 - G[:, [k]]))) + a_beta[k] / b_beta[k] * np.identity(DV)
    return H


def train_mix_priors(V: np.ndarray, Lambda_V_1: np.ndarray):
    """
    [PDF p. 244]

    :param V: mixing weight matrix (DV × K)
    :param Lambda_V_1: mixing covariance matrix (K DV × K DV)

    :returns: mixing weight vector prior parameters a_beta, b_beta
    """
    DV, K = V.shape
    assert Lambda_V_1.shape == (K * DV, K * DV)

    a_beta = np.repeat(HParams().A_BETA, (K, ))
    b_beta = np.repeat(HParams().B_BETA, (K, ))
    Lambda_V_1_diag = np.diag(Lambda_V_1)
    for k in range(K):
        v_k = V[:, [k]]
        l = k * DV
        u = (k + 1) * DV
        a_beta[k] += DV / 2
        try:
            b_beta[k] += 0.5 * (np.sum(Lambda_V_1_diag[l:u:1]) + v_k.T @ v_k)
        except FloatingPointError as e:
            known_issue(
                "FloatingPointError in train_mix_priors",
                f"v_k = {v_k}, K = {K}, V = {V}, Lambda_V_1 = {Lambda_V_1}",
                report=True)
            mlflow.set_tag("FloatingPointError_train_mix_priors", "occurred")
            raise e

    return a_beta, b_beta


def var_bound(M: np.ndarray, X: np.ndarray, Y: np.ndarray, Phi: np.ndarray,
              W: List[np.ndarray], Lambda_1: List[np.ndarray],
              a_tau: np.ndarray, b_tau: np.ndarray, a_alpha: np.ndarray,
              b_alpha: np.ndarray, V: np.ndarray, Lambda_V_1: np.ndarray,
              a_beta, b_beta):
    """
    [PDF p. 244]

    :param M: matching matrix (N × K)
    :param X: input matrix (N × DX)
    :param Y: output matrix (N × DY)
    :param Phi: mixing feature matrix (N × DV)
    :param W: submodel weight matrices (list of DY × DX)
    :param Lambda_1: submodel covariance matrices (list of DX × DX)
    :param a_tau: submodel noise precision parameters
    :param b_tau: submodel noise precision parameters
    :param a_alpha: weight vector prior parameters
    :param b_alpha: weight vector prior parameters
    :param V: mixing weight matrix (DV × K)
    :param Lambda_V_1: mixing covariance matrix (K DV × K DV)
    :param a_beta: mixing weight prior parameter (row vector of length K)
    :param b_beta: mixing weight prior parameter (row vector of length K)

    :returns: variational bound L(q)
    """
    DV, K = V.shape
    assert Lambda_V_1.shape == (K * DV, K * DV)
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

    :param X: input matrix (N × DX)
    :param Y: output matrix (N × DY)
    :param W_k: submodel weight matrix (DY × DX)
    :param Lambda_k_1: submodel covariance matrix (DX × DX)
    :param a_tau_k: submodel noise precision parameter
    :param b_tau_k: submodel noise precision parameter
    :param a_alpha_k: weight vector prior parameter
    :param b_alpha_k: weight vector prior parameter
    :param r_k: responsibility vector (NumPy row or column vector, we reshape to
        (-1) anyways)

    :returns: rule component L_k(q) of variational bound
    """
    DY, DX = W_k.shape
    E_tau_tau_k = a_tau_k / b_tau_k
    L_k_1_q = DY / 2 * (ss.digamma(a_tau_k) - np.log(b_tau_k)
                        - np.log(2 * np.pi)) * np.sum(r_k)
    # We reshape r_k to a NumPy row vector since NumPy seems to understand what
    # we want to do when we multiply two row vectors (i.e. a^T a).
    L_k_2_q = (-0.5 * r_k).reshape((-1)) @ (E_tau_tau_k * np.sum(
        (Y - X @ W_k.T)**2, 1) + DY * np.sum(X * (X @ Lambda_k_1), 1))
    L_k_3_q = -ss.gammaln(HParams().A_ALPHA) + HParams().A_ALPHA * np.log(
        HParams().B_ALPHA) + ss.gammaln(a_alpha_k) - a_alpha_k * np.log(
            b_alpha_k) + DX * DY / 2 + DY / 2 * np.log(
                np.linalg.det(Lambda_k_1))
    L_k_4_q = DY * (-ss.gammaln(HParams().A_TAU)
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

    Parameters
    ----------
    G : array of shape (N, K)
        Mixing (“gating”) matrix.
    R : array of shape (N, K)
        Responsibility matrix.
    V : array of shape (DV, K)
        Mixing weight matrix.
    Lambda_V_1 : array of shape (K * DV, K * DV)
        Mixing weight covariance matrix.
    a_beta : array of shape (K,)
        Mixing weight prior parameter (row vector).
    b_beta : array of shape (K,)

    Returns
    -------
    L_M_q : float
        Mixing component L_M(q) of variational bound.
    """
    DV, K = V.shape

    assert G.shape == R.shape
    assert G.shape[1] == K
    assert Lambda_V_1.shape == (K * DV, K * DV)
    assert a_beta.shape == (K, )
    assert b_beta.shape == (K, )

    L_M1q = K * (-ss.gammaln(HParams().A_BETA)
                 + HParams().A_BETA * np.log(HParams().B_BETA))
    # TODO Performance: LCSBookCode vectorized this
    # TODO Performance: ss.gammaln(a_beta[k]) is constant throughout the loop in
    # the calling function
    for k in range(K):
        # NOTE this is just the negated form of the update two lines prior?
        try:
            L_M1q = L_M1q + ss.gammaln(
                a_beta[k]) - a_beta[k] * np.log(b_beta[k])
        except FloatingPointError as e:
            known_issue("FloatingPointError in var_mix_bound",
                        (f"Lambda_V_1: {Lambda_V_1}"
                         f"V: {V}"
                         f"K: {K}"
                         f"k: {k}"
                         f"b_beta[k]: {b_beta[k]}"),
                        report=True)
            raise e

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
    # This fixes(?) some numerical problems. Note that L_M2q is the *negative*
    # Kullback-Leibler divergence (hence the > 0 comparison instead of < 0).
    if L_M2q > 0 and np.isclose(L_M2q, 0):
        L_M2q = 0
    assert L_M2q <= 0, f"Kullback-Leibler divergence less than zero: {-L_M2q}"
    # TODO Performance: slogdet can be cached, is computed more than once
    # L_M3q may be -inf after the following line but that is probably OK since
    # the ``train_mixing`` loop then aborts (also see comment in
    # ``train_mixing``).
    L_M3q = 0.5 * np.linalg.slogdet(Lambda_V_1)[1] + K * DV / 2
    if np.any(~np.isfinite([L_M1q, L_M2q, L_M3q])):
        known_issue("Infinite var_mix_bound", (f"Lambda_V_1 = {Lambda_V_1}, "
                                               f"L_M1q = {L_M1q}, "
                                               f"L_M2q = {L_M2q}, "
                                               f"L_M3q = {L_M3q}"))
    return L_M1q + L_M2q + L_M3q
