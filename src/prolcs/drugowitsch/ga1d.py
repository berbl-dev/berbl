import sys
from copy import deepcopy
from typing import *

import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore
import scipy.stats as sstats  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from ..common import matching_matrix, phi_standard
from ..logging import log_
from ..radialmatch1d import RadialMatch1D
from ..utils import get_ranges
from . import *
from .hyperparams import HyperParams

Individual = RadialMatch1D
Population = List[Individual]


def deterministic_tournament(fits: List[float], size: int,
                             random_state: np.random.RandomState):
    """
    I can only guess, what [PDF p. 249] means by “deterministic tournament
    selection”.

    :param fits: List of fitness values

    :returns: Index of fitness value of selected individual
    """
    tournament = random_state.choice(range(len(fits)), size=size)
    return max(tournament, key=lambda i: fits[i])


def mutate(MM: List, random_state: np.random.RandomState):
    return list(map(lambda i: i.mutate(random_state), MM))


def crossover(M_a: List, M_b: List, random_state: np.random.RandomState):
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
    M_a_ = M_a + M_b
    random_state.shuffle(M_a_)
    # TODO Is this correct: This is how Drugowitsch does it but this way we get
    # many small individuals (higher probability of creating small individuals
    # due to selecting 9 ∈ [0, 10] being equal to selecting 1 ∈ [0, 10]).
    K_b_ = random_state.randint(low=1, high=K_a + K_b)
    # This way we might be able to maintain current individual sizes on average.
    # K_b_ = int(np.clip(np.random.normal(loc=K_a), a_min=1, a_max=K_a + K_b))
    M_b_ = []
    for k in range(K_b_):
        i = random_state.randint(low=0, high=len(M_a_))
        M_b_.append(M_a_[i])
        del M_a_[i]
    assert K_a + K_b - K_b_ == len(M_a_)
    assert K_b_ == len(M_b_)
    assert K_a + K_b == len(M_b_) + len(M_a_)
    return M_a_, M_b_


def cl_count(P):
    return np.sum(np.array(list(map(len, P))))


def avg_ind_size(P):
    return np.mean(np.array(list(map(len, P))))


class DrugowitschGA1D(BaseEstimator):
    def __init__(self,
                 random_state=None,
                 phi: Callable[[np.ndarray], np.ndarray] = phi_standard,
                 n_iter: int = 250,
                 pop_size: int = 20,
                 init_avg_ind_size: int = 10,
                 init: Callable[[np.ndarray, np.ndarray], Population] = None,
                 tnmt_size: int = 5,
                 cross_prob: float = 0.4,
                 muta_prob: float = 0.4,
                 a_alpha: float = 10**-2,
                 b_alpha: float = 10**-4,
                 a_beta: float = 10**-2,
                 b_beta: float = 10**-4,
                 a_tau: float = 10**-2,
                 b_tau: float = 10**-4,
                 delta_s_l_k_q: float = 10**-4,
                 delta_s_l_m_q: float = 10**-2,
                 delta_s_klrg: float = 10**-8,
                 exp_min: float = np.log(np.finfo(None).tiny),
                 ln_max: float = np.log(np.finfo(None).max),
                 logging: str = "mlflow"):
        """


        Parameters are hyper parameters for the LCS as a whole, for the employed
        GA as well as for the Bayesian/variational framework (Table 8.1, PDF p.
        233).

        :param random_state: Used to construct a ``numpy.random.RandomState``
            object for repeatability.
        :param phi: Mixing feature extractor (N × D_X → N × D_V), in LCS usually
            ``phi(X) = np.ones(…)`` [PDF p. 234]. For performance reasons, we
            transform the whole input matrix to the feature matrix at once
            (other than Drugowitsch, who specifies a function operating on a
            single sample).
        :param n_iter: Number of iterations to run the GA for.
        :param pop_size: Population size.
        :param init_avg_ind_size: Average individual size to use for
            initialization (done by drawing individual sizes uniformly from
            ``[1, init_avg_ind_size * 2]``), gets overridden by init
        :param init: Custom function for data-dependent init, receives ``X`` and
            ``y`` as arguments, overrides ``init_avg_ind_size`` and
            ``pop_size``.
        :param tnmt_size: Tournament size.
        :param cross_prob: Crossover probability.
        :param muta_prob: Mutation probability.
        :param a_alpha: Scale parameter of weight vector variance prior.
        :param b_alpha: Shape parameter of weight vector variance prior.
        :param a_beta: Scale parameter of mixing weight vector variance prior.
        :param b_beta: Shape parameter of mixing weight vector variance prior.
        :param a_tau: Scale parameter of noise variance prior.
        :param b_tau: Shape parameter of noise variance prior.
        :param delta_s_l_k_q: Stopping criterion for classifier update.
        :param delta_s_l_m_q: Stopping criterion for mixing model update.
        :param delta_s_klrg: Stopping criterion for mixing weight update.
        :param exp_min: Lowest real number ``x`` on system such that ``exp(x) >
            0``. The default is the logarithm of the smallest positive number of
            the default dtype (as of 2020-10-06, this dtype is float64).
        :param ln_max: ``ln(x)``, where ``x`` is the highest real number on the
            system. The default is the logarithm of the highest number of the
            default dtype.
        :param logging: Whether to side-channel log things, currently only
            "mlflow" is supported any other string results in not performing any
            side-channel logging.
        """

        self.random_state = random_state
        self.phi = phi
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.init_avg_ind_size = init_avg_ind_size
        self.init = init
        self.tnmt_size = tnmt_size
        self.cross_prob = cross_prob
        self.muta_prob = muta_prob
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_beta = a_beta
        self.b_beta = b_beta
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.delta_s_l_k_q = delta_s_l_k_q
        self.delta_s_l_m_q = delta_s_l_m_q
        self.delta_s_klrg = delta_s_klrg
        self.exp_min = exp_min
        self.ln_max = ln_max
        self.logging = logging

        self._P: Population = []
        self._elitist_index: int = None
        self._elitist: Individual = None
        self._p_M_D_elitist: float = -np.inf
        self._params_elitist = None

    def fit(self, X, y, **kwargs):
        self.random_state_ = check_random_state(self.random_state)
        self.ga(X=X,
                Y=y,
                random_state=self.random_state_,
                phi=self.phi,
                n_iter=self.n_iter,
                pop_size=self.pop_size,
                init_avg_ind_size=self.init_avg_ind_size,
                init=self.init,
                tnmt_size=self.tnmt_size,
                cross_prob=self.cross_prob,
                muta_prob=self.muta_prob,
                logging=self.logging)
        return self

    def ga(self, X: np.ndarray, Y: np.ndarray,
           random_state: np.random.RandomState,
           phi: Callable[[np.ndarray], np.ndarray], n_iter: int, pop_size: int,
           init_avg_ind_size: int, init: Callable[[np.ndarray, np.ndarray],
                                                  Population], tnmt_size: int,
           cross_prob: float, muta_prob: float, logging: str):
        """
        [PDF p. 248 ff.]

        “Allele of an individual's genome is given by the representation of a
        single classifier's matching function, which makes the genome's length
        determined by the number of classifiers of the associated model
        structure. As this number is not fixed, the individuals in the
        population can be of variable length.” [PDF p. 249]

        Fitness is model probability (which manages bloat by having lower values
        for overly complex model structures).

        Default values are the ones from [PDF p. 260].

        :param X: input matrix (N × D_X)
        :param Y: output matrix (N × D_Y)
        :param phi: mixing feature extractor (N × D_X → N × D_V), in LCS usually
            ``phi(X) = np.ones(…)`` [PDF p. 234]. For performance reasons, we
            transform the whole input matrix to the feature matrix at once
            (other than Drugowitsch, who specifies a function operating on a
            single sample).
        :param n_iter: iterations to run the GA
        :param pop_size: population size
        :param init_avg_ind_size: average individual size to use for
            initialization (done by drawing individual sizes uniformly from
            ``[1, init_avg_ind_size * 2]``), gets overridden by init
        :param init: custom function for data-dependent init, receives ``X`` and
            ``Y`` as arguments, overrides ``init_avg_ind_size`` and ``pop_size``
        :param tnmt_size: tournament size
        :param cross_prob: crossover probability
        :param muta_prob: mutation probability

        :returns: model structures (list of N × K) with their probabilities
        """
        N, D_X = X.shape

        Phi = phi(X)

        # This might seem ugly (and it certainly is), but this way, we are able
        # to keep the signatures in prolcs.drugowitsch.__init__.py clean and
        # very close to the algorithmic description.
        HyperParams().A_ALPHA = self.a_alpha
        HyperParams().B_ALPHA = self.b_alpha
        HyperParams().A_BETA = self.a_beta
        HyperParams().B_BETA = self.b_beta
        HyperParams().A_TAU = self.a_tau
        HyperParams().B_TAU = self.b_tau
        HyperParams().DELTA_S_L_K_Q = self.delta_s_l_k_q
        HyperParams().DELTA_S_L_M_Q = self.delta_s_l_m_q
        HyperParams().DELTA_S_KLRG = self.delta_s_klrg
        HyperParams().EXP_MIN = self.exp_min
        HyperParams().LN_MAX = self.ln_max
        HyperParams().LOGGING = self.logging

        if init is None:
            Ks = random_state.randint(low=1,
                                      high=2 * init_avg_ind_size,
                                      size=pop_size)
            ranges = get_ranges(X)
            self.P_ = [
                individual(ranges, k, random_state=random_state) for k in Ks
            ]
        else:
            self.P_ = init(X, Y)

        # TODO Parametrize number of elitists
        self.elitist_index_ = None
        self.elitist_ = None
        self.p_M_D_elitist_ = -np.inf
        self.params_elitist_ = None
        for i in range(n_iter):
            sys.stdout.write(
                f"\rStarting iteration {i+1}/{n_iter}, "
                f"best solution of size "
                f"{len(self.elitist_) if self.elitist_ is not None else '?'} "
                f"at p_M(D) = {self.p_M_D_elitist_}\t")
            Ms = map(lambda ind: matching_matrix(ind, X), self.P_)
            # Compute fitness for each individual (i.e. model probabilities).
            # Also: Get params.
            p_M_D_and_params = list(
                map(
                    lambda M: model_probability(M, X, Y, Phi, self.exp_min,
                                                self.ln_max), Ms))
            p_M_D, params = tuple(zip(*p_M_D_and_params))
            p_M_D, params = list(p_M_D), list(params)
            self.elitist_index_ = np.argmax(p_M_D)
            if p_M_D[self.elitist_index_] > self.p_M_D_elitist_:
                self.elitist_ = deepcopy(self.P_[self.elitist_index_])
                self.p_M_D_elitist_ = p_M_D[self.elitist_index_]
                self.params_elitist_ = deepcopy(params[self.elitist_index_])
            P__: List[np.ndarray] = []
            while len(P__) < pop_size:
                i1, i2 = deterministic_tournament(
                    p_M_D, size=tnmt_size,
                    random_state=random_state), deterministic_tournament(
                        p_M_D, size=tnmt_size, random_state=random_state)
                c1, c2 = deepcopy(self.P_[i1]), deepcopy(self.P_[i2])
                if random_state.random() < cross_prob:
                    c1, c2 = crossover(c1, c2, random_state)
                if random_state.random() < muta_prob:
                    c1, c2 = mutate(c1, random_state), mutate(c2, random_state)
                P__.append(c1)
                P__.append(c2)

            self.P_ = P__
            # print("")
            # print(pop_stats(self.P_))
            log_("fitness", self._p_M_D_elitist, step=i)
            log_("cl_count", cl_count(self.P_), step=i)
            log_("avg_ind_size", avg_ind_size(self.P_), step=i)

        return self.phi, self.elitist_, self.p_M_D_elitist_, self.params_elitist_
