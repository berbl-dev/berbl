from typing import *
import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore
import scipy.stats as sstats  # type: ignore
from .radialmatch import RadialMatch
from copy import deepcopy

# Underflows may occur at many places, e.g. if X contains values very close to
# 0.
# TODO Are underflows problematic?
np.seterr(all="raise", under="ignore")

# hyper parameters, Table 8.1 (PDF p. 233)
A_ALPHA = 10**-2
B_ALPHA = 10**-4
A_BETA = 10**-2
B_BETA = 10**-4
A_TAU = 10**-2
B_TAU = 10**-4
DELTA_S_L_K_Q = 10**-4
DELTA_S_L_M_Q = 10**-2
DELTA_S_KLRG = 10**-8
# We use the default dtype everywhere (as of 2020-10-06, it's float64).
EXP_MIN = np.log(np.finfo(None).tiny)
LN_MAX = np.log(np.finfo(None).max)
"""
How NumPy interprets some things.

``
row_vector = [1, 2, 3]
col_vector = [[1], [2], [3]]
``

[PDF p. 234]

* special product and division: simply ``*`` and ``/``
* Sum(A): ``np.sum(a)``
* RowSum(A): ``np.sum(a, 1)``, with a reshape(-1, 1) afterwards (unless we use @
  right away, as then, NumPy treats it correctly as a column vector anyways)
* FixNaN(A, b): ``np.nan_to_num(a, nan=b)``
"""

rng = np.random.default_rng(0)




def phi_standard(X: np.ndarray):
    """
    The mixing feature extractor usually employed by LCSs, i.e. ``phi(x) = 1``
    for each sample ``x``.
    """
    N, D_X = X.shape
    return np.ones((N, 1))


def predict(X, mstruct, params, phi):
    """
    [PDF p. 224]

    The mean and variance of the predictive density described by the parameters.

    “As the mixture of Student’s t distributions might be multimodal, there
    exists no clear definition for the 95% confidence intervals, but a mixture
    density-related study that deals with this problem can be found in [118].
    Here, we take the variance as a sufficient indicator of the prediction’s
    confidence.” [PDF p. 224]

    :param X: input matrix (N × D_X)
    :param params: model parameters as returned by `ga`
    :param phi: mixing feature extractor, see `ga`

    :returns: mean output matrix (N × D_Y), variance of output matrix
    """
    # TODO


def predict1(x, mstruct, W, Lambda_1, a_tau, b_tau, V, phi):
    """
    [PDF p. 224]

    The mean and variance of the predictive density described by the parameters.

    “As the mixture of Student’s t distributions might be multimodal, there
    exists no clear definition for the 95% confidence intervals, but a mixture
    density-related study that deals with this problem can be found in [118].
    Here, we take the variance as a sufficient indicator of the prediction’s
    confidence.” [PDF p. 224]

    :param X: input vector (D_X)
    :param params: model parameters as returned by `ga`
    :param phi: mixing feature extractor, see `ga`

    :returns: mean output matrix (D_Y), variance of output (D_Y)
    """
    X = x.reshape((1, -1))
    M = matching_matrix(mstruct, X)
    N, K = M.shape
    Phi = phi(X)
    G = mixing(M, Phi, V)  # (N=1) × K
    g = G[0]
    D_Y, D_X = W[0].shape

    x_ = np.append(1, x)

    W = np.array(W)

    # (\sum_k g_k(x) W_k) x, (7.108)
    # TODO This can probably be vectorized
    gW = 0
    for k in range(len(W)):
        gW += g[k] * W[k]
    y = gW @ x_

    var = np.zeros(D_Y)
    # TODO Can this be vectorized?
    for j in range(D_Y):
        for k in range(K):
            var[j] += g[k] * (2 * b_tau[k] / (a_tau[k] - 1) *
                              (1 + x_ @ Lambda_1[k] @ x_) +
                              (W[k][j] @ x_)**2) - y[j]**2
    return y, var


def predictive_density(M, Phi, W, Lambda_1, a_tau, b_tau, V):
    """
    [PDF p. 223], i.e. (7.106).

    :param M: matching matrix for the sample considered ((N=1) × K )
    :param Phi: mixing feature matrix for the sample considered ((N=1) × D_V)
    :param W: classifier weight matrices (list of D_Y × D_X)
    :param Lambda_1: classifier covariance matrices (list of D_X × D_X)
    :param a_tau: classifier noise precision parameters
    :param b_tau: classifier noise precision parameters
    :param V: mixing weight matrix (D_V × K)

    :returns: predictive density function ``p`` such that ``p(x)(y) == p(y |
        x)``
    """
    N, K = M.shape
    assert N == 1, str(N)

    def p(x: np.ndarray):
        def p_(y: np.ndarray):
            D_Y, = y.shape
            # mixing matrix containing the values for g_k(x_n) for each
            # classifier/input combination
            G = mixing(M, Phi, V)  # (N=1) × K
            assert G.shape == (1, K)

            # TODO Vectorize this loop if possible
            pxy = 0
            for k in range(K):
                prod = 1
                for j in range(D_Y):
                    # Drugowitsch's w_kj is a row vector of W[k] [PDF p. 195]
                    # with W[k] being (D_Y × D_X)
                    mu = W[k][j] @ x
                    lambd = (1 + x @ (Lambda_1[k] @ x))**(
                        -1) * a_tau[k] / b_tau[k]
                    df = 2 * a_tau[k]
                    prod *= sstats.t(df=df, loc=mu, scale=lambd).pdf(y[j])
                pxy += G[0][k] * prod
            return pxy

        return p_

    return p


def ga(X: np.ndarray,
       Y: np.ndarray,
       phi: Callable[[np.ndarray], np.ndarray] = phi_standard,
       iter: int = 250,
       pop_size: int = 20,
       tnmt_size: int = 5,
       cross_prob: float = 0.4,
       muta_prob: float = 0.4):
    """
    [PDF p. 248 ff.]

    “Allele of an individual's genome is given by the representation of a single
    classifier's matching function, which makes the genome's length determined
    by the number of classifiers of the associated model structure. As this
    number is not fixed, the individuals in the population can be of variable
    length.” [PDF p. 249]

    Fitness is model probability (which manages bloat by having lower values for
    overly complex model structures).

    Default values are the ones from [PDF p. 260].

    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param phi: mixing feature extractor (N × D_X → N × D_V), in LCS usually
        ``phi(X) = np.ones(…)`` [PDF p. 234]. For performance reasons, we
        transform the whole input matrix to the feature matrix at once (other
        than Drugowitsch, who specifies a function operating on a single
        sample).
    :param iter: iterations to run the GA
    :param pop_size: population size
    :param tnmt_size: tournament size
    :param cross_prob: crossover probability
    :param muta_prob: mutation probability

    :returns: model structures (list of N × K) with their probabilities
    """
    N, D_X = X.shape

    Phi = phi(X)

    # [PDF p. 221, 3rd paragraph]
    # TODO Extract initialization
    # TODO Hardcoded 100 here …
    # Ks = rng.integers(low=1, high=10, size=pop_size)
    # TODO Drugowitsch samples Ks from problem-dependent Binomial distribution
    # Ks = np.clip(rng.binomial(8, 0.5, size=pop_size - 1), 1, 100)
    Ks = np.clip(rng.binomial(8, 0.5, size=pop_size), 1, 100)
    ranges = np.vstack([np.min(X, axis=0), np.max(X, axis=0)]).T
    P = [individual(ranges, k) for k in Ks]
    # TODO Parametrize number of elitists
    elitist_index = None
    elitist = None
    p_M_D_elitist = -np.inf
    params_elitist = None
    for i in range(iter):
        sys.stdout.write(f"\rStarting iteration {i+1}/{iter}, "
                         f"best solution of size {len(elitist) if elitist is not None else '?'} "
                         f"at p_M(D) = {p_M_D_elitist}\t")
        Ms = map(lambda ind: matching_matrix(ind, X), P)
        # Compute fitness for each individual (i.e. model probabilities). (Also: Get params.)
        p_M_D_and_params = list(
            map(lambda M: model_probability(M, X, Y, Phi), Ms))
        p_M_D, params = tuple(zip(*p_M_D_and_params))
        p_M_D, params = list(p_M_D), list(params)
        elitist_index = np.argmax(p_M_D)
        if p_M_D[elitist_index] > p_M_D_elitist:
            elitist = deepcopy(P[elitist_index])
            p_M_D_elitist = p_M_D[elitist_index]
            params_elitist = deepcopy(params[elitist_index])
        P_: List[np.ndarray] = []
        while len(P_) < pop_size:
            i1, i2 = deterministic_tournament(
                p_M_D,
                size=tnmt_size), deterministic_tournament(p_M_D,
                                                          size=tnmt_size)
            c1, c2 = P[i1], P[i2]
            if rng.random() < cross_prob:
                c1, c2 = crossover(c1, c2)
            if rng.random() < muta_prob:
                c1, c2 = mutate(c1), mutate(c2)
            P_.append(c1)
            P_.append(c2)

        P = P_
    return phi, elitist, p_M_D_elitist, params_elitist


    return P, p_M_D, params, phi, elitist_index


def matching_matrix(ind: List, X: np.ndarray):
    """
    :param ind: an individual for which the matching matrix is returned
    :param X: input matrix (N × D_X)

    :returns: matching matrix (N × K)
    """
    # TODO Can we maybe vectorize this?
    return np.hstack(list(map(lambda m: m.match(X), ind)))


def individual(ranges: np.ndarray, k: int):
    """
    Individuals are simply lists of matching functions (the length of the list
    is the number of classifiers, the matching functions specify their
    localization).
    """
    return [RadialMatch.random(ranges) for i in range(k)]


def deterministic_tournament(fits: List[float], size: int):
    """
    I can only guess, what [PDF p. 249] means by “deterministic tournament
    selection”.

    :param fits: List of fitness values

    :returns: Index of fitness value of selected individual
    """
    tournament = rng.choice(range(len(fits)), size=size)
    return max(tournament, key=lambda i: fits[i])


def mutate(MM: List):
    return list(map(lambda i: i.mutate(), MM))


def crossover(M_a: List, M_b: List):
    """
    [PDF p. 250]

    :param M_a: a model structure (a simple Python list; originally the number
        of classifiers and their localization but the length of a Python list is
        easily obtainable)
    :param M_b: another model structure

    :returns: two model structures resulting from crossover of inputs
    """
    K_a = len(M_a)
    K_b = len(M_b)
    # We deepcopy because we have to get proper copies of the matching
    # functions.
    M_a_ = deepcopy(M_a) + deepcopy(M_b)
    rng.shuffle(M_a_)
    # TODO Is this correct: This is how Drugowitsch does it but this way we get
    # many small individuals (higher probability of creating small individuals
    # due to selecting 9 ∈ [0, 10] being equal to selecting 1 ∈ [0, 10]).
    K_b_ = rng.integers(low=1, high=K_a + K_b)
    # This way we might be able to maintain current individual sizes on average.
    # K_b_ = int(np.clip(np.random.normal(loc=K_a), a_min=1, a_max=K_a + K_b))
    M_b_ = []
    for k in range(K_b_):
        i = rng.integers(low=0, high=len(M_a_))
        M_b_.append(M_a_[i])
        del M_a_[i]
    assert K_a + K_b - K_b_ == len(M_a_)
    assert K_b_ == len(M_b_)
    assert K_a + K_b == len(M_b_) + len(M_a_)
    return M_a_, M_b_


def model_probability(M: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      Phi: np.ndarray):
    """
    [PDF p. 235]

    Note that this deviates from [PDF p. 235] in that we return `L(q) - ln K!`
    instead of `L(q) + ln K!` because the latter is not consistent with (7.3).

    :param M: matching matrix (N × K)
    :param X: input matrix (N × D_X)
    :param Y: output matrix (N × D_Y)
    :param Phi: mixing feature matrix (N × D_V)

    :returns: approximate model probability L(q) + ln p(M)
    """
    N, K = M.shape

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
                                                 b_tau=b_tau)
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

    params = {
        "W": W,
        "Lambda_1": Lambda_1,
        "a_tau": a_tau,
        "b_tau": b_tau,
        "V": V
    }
    ln_p_M = -np.log(np.math.factorial(K))  # (7.3), i.e. p_M \propto 1/K
    return L_q + ln_p_M, params


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
    a_alpha_k, b_alpha_k = A_ALPHA, B_ALPHA
    a_tau_k, b_tau_k = A_TAU, B_TAU
    L_k_q = -np.inf
    delta_L_k_q = DELTA_S_L_K_Q + 1
    # This is constant; Drugowitsch nevertheless puts it into the while loop
    # (probably for readability).
    a_alpha_k = A_ALPHA + D_X * D_Y / 2
    # Drugowitsch reaches convergence usually after 3-4 iterations [PDF p. 237].
    while delta_L_k_q > DELTA_S_L_K_Q:
        E_alpha_alpha_k = a_alpha_k / b_alpha_k
        Lambda_k = np.diag([E_alpha_alpha_k] * D_X) + X_k.T @ X_k
        Lambda_k_1 = np.linalg.inv(Lambda_k)
        W_k = Y_k.T @ X_k @ Lambda_k_1
        a_tau_k = A_TAU + 0.5 * np.sum(m_k)
        b_tau_k = B_TAU + 1 / (2 * D_Y) * (np.sum(Y_k * Y_k)
                                           - np.sum(W_k * (W_k @ Lambda_k)))
        E_tau_tau_k = a_tau_k / b_tau_k
        # D_Y factor in front of trace due to sum over D_Y elements (7.100).
        b_alpha_k = B_ALPHA + 0.5 * (E_tau_tau_k * np.sum(W_k * W_k)
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
                 a_tau: np.ndarray, b_tau: np.ndarray):
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

    V = rng.normal(loc=0, scale=A_BETA / B_BETA, size=(D_V, K))
    a_beta = np.repeat(A_BETA, K)
    b_beta = np.repeat(B_BETA, K)
    L_M_q = -np.inf
    delta_L_M_q = DELTA_S_L_M_Q + 1
    while delta_L_M_q > DELTA_S_L_M_Q:
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

    # Limit all elements of G such that EXP_MIN <= g_nk <= LN_MAX - np.log(K).
    G = np.clip(G, EXP_MIN, LN_MAX - np.log(K))

    G = np.exp(G) * M
    # The sum can be 0 meaning we do 0/0 (== NaN) but we ignore it because it is
    # fixed one line later (this is how Drugowitsch does it).
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
    # The sum can be 0 meaning we do 0/0 (== NaN) but we ignore it because it is
    # fixed one line later (this is how Drugowitsch does it).
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

    # NOTE Not quite sure why Drugowitsch doesn't use his division operator for
    # this expression.
    E_beta_beta = a_beta / b_beta
    G = mixing(M, Phi, V)
    R = responsibilities(X=X,
                         Y=Y,
                         G=G,
                         W=W,
                         Lambda_1=Lambda_1,
                         a_tau=a_tau,
                         b_tau=b_tau)
    KLRG = np.inf
    delta_KLRG = DELTA_S_KLRG + 1
    while delta_KLRG > DELTA_S_KLRG:
        E = Phi.T @ (G - R) + V * E_beta_beta
        e = E.T.reshape((-1))
        H = hessian(Phi=Phi, G=G, a_beta=a_beta, b_beta=b_beta)
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
        # responsibilities performs a `nan_to_num(…, nan=0)`, so we might divide by
        # 0 here. The intended behaviour is to silently get a NaN that can then
        # be replaced by 0 again (this is how Drugowitsch does it [PDF p. 213]).
        # TODO This should actually result in a `divide` error (and not an
        # `invalid`).
        with np.errstate(divide="ignore"):
            KLRG = np.sum(R * np.nan_to_num(np.log(G / R), nan=0))
        # Just to make sure that we don't accidentally get an inf here …
        assert np.isfinite(KLRG)
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
            H[lk:uk:1, lj:uj:1] = -Phi.T @ (Phi * (G[:, [k]] * G[:, [j]]))
            H[lj:uj:1, lk:uk:1] = -Phi.T @ (Phi * (G[:, [k]] * G[:, [j]]))
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
    for k in range(K):
        v_k = V[:, [k]]
        l = k * D_V
        u = (k + 1) * D_V
        # Not that efficient, I think (but very close to [PDF p. 244]).
        # Lambda_V_1_kk = Lambda_V_1[l:u:1, l:u:1]
        # a_beta[k] = A_BETA + D_V / 2
        # b_beta[k] = B_BETA + 0.5 * (np.trace(Lambda_V_1_kk) + v_k.T @ v_k)
        # More efficient.
        a_beta[k] = A_BETA + D_V / 2
        b_beta[k] = B_BETA + 0.5 * (np.sum(Lambda_V_1_diag[l:u:1])
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
    L_k_3_q = -ss.gammaln(A_ALPHA) + A_ALPHA * np.log(B_ALPHA) + ss.gammaln(
        a_alpha_k
    ) - a_alpha_k * np.log(b_alpha_k) + D_X * D_Y / 2 + D_Y / 2 * np.log(
        np.linalg.det(Lambda_k_1))
    L_k_4_q = D_Y * (-ss.gammaln(A_TAU) + A_TAU * np.log(B_TAU) +
                     (A_TAU - a_tau_k) * ss.digamma(a_tau_k)
                     - A_TAU * np.log(b_tau_k) - B_TAU * E_tau_tau_k
                     + ss.gammaln(a_tau_k) + a_tau_k)
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

    L_M1q = K * (-ss.gammaln(A_BETA) + A_BETA * np.log(B_BETA))
    for k in range(K):
        # NOTE this is just the negated form of the update two lines prior?
        L_M1q = L_M1q + ss.gammaln(a_beta[k]) - a_beta[k] * np.log(b_beta[k])

    # L_M2q is the Kullback-Leibler divergence [PDF p. 246].
    # responsibilities performs a `nan_to_num(…, nan=0)`, so we might divide by
    # 0 here. The intended behaviour is to silently get a NaN that can then
    # be replaced by 0 again (this is how Drugowitsch does it [PDF p. 213]).
    with np.errstate(divide="ignore"):
        L_M2q = np.sum(R * np.nan_to_num(np.log(G / R), nan=0))
    L_M3q = 0.5 * np.linalg.slogdet(Lambda_V_1)[1] + K * D_V / 2
    return L_M1q + L_M2q + L_M3q
