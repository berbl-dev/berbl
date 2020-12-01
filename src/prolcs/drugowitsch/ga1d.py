import sys
from copy import deepcopy
from typing import *

import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore
import scipy.stats as sstats  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

from ..common import phi_standard
from ..logging import log_
from ..radialmatch1d import RadialMatch1D
from ..utils import get_ranges
from . import *
from .hyperparams import HParams

Individual = Model
Population = List[Individual]


def deterministic_tournament(models: List[Model], size: int,
                             random_state: np.random.RandomState):
    """
    I can only guess, what [PDF p. 249] means by “deterministic tournament
    selection”.

    :param models: List of individuals (models) to select from

    :returns: A deepcopy of the tournament winner
    """
    tournament = random_state.choice(models, size=size)
    # TODO Think about renaming p_M_D to score
    return deepcopy(max(tournament, key=lambda model: model.p_M_D))


def cl_count(P):
    return np.sum(np.array(list(map(lambda m: m.size(), P))))


def avg_ind_size(P):
    return np.mean(np.array(list(map(lambda m: m.size(), P))))


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
            ``y`` as arguments, generating a list of ``Model``s; overrides
            ``init_avg_ind_size`` and ``pop_size``.
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

        self.P_: Population = []
        self.elitist_: Individual = None

    # Don't try to serialize the init function (when serializing the model,
    # initialization is over anyway, so this is kind of OK, I think).
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["init"]
        return state

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
        HParams().A_ALPHA = self.a_alpha
        HParams().B_ALPHA = self.b_alpha
        HParams().A_BETA = self.a_beta
        HParams().B_BETA = self.b_beta
        HParams().A_TAU = self.a_tau
        HParams().B_TAU = self.b_tau
        HParams().DELTA_S_L_K_Q = self.delta_s_l_k_q
        HParams().DELTA_S_L_M_Q = self.delta_s_l_m_q
        HParams().DELTA_S_KLRG = self.delta_s_klrg
        HParams().EXP_MIN = self.exp_min
        HParams().LN_MAX = self.ln_max
        HParams().LOGGING = self.logging

        if init is None:
            raise Exception("Automatic init not supported yet")
            Ks = random_state.randint(low=1,
                                      high=2 * init_avg_ind_size,
                                      size=pop_size)
            ranges = get_ranges(X)
            self.P_ = [
                individual(ranges, k, random_state=random_state) for k in Ks
            ]
        else:
            self.P_ = init(X, Y)

        def fitness(m):
            return model_probability(m, X, Y, Phi, self.exp_min, self.ln_max)

        # TODO Parametrize number of elitists
        self.elitist_ = None
        for i in range(n_iter):
            sys.stdout.write(
                f"\rStarting iteration {i+1}/{n_iter}, "
                f"best solution of size "
                f"{self.elitist_.size() if self.elitist_ is not None else '?'} "
                f"at p_M(D) = {self.elitist_.p_M_D if self.elitist_ is not None else '?'}\t"
            )

            # Evaluate population and store elitist.
            self.P_ = list(map(fitness, self.P_))
            self.elitist_ = max(
                self.P_ +
                ([self.elitist_] if self.elitist_ is not None else []),
                key=lambda i: i.p_M_D)

            P__: List[np.ndarray] = []
            while len(P__) < pop_size:
                c1, c2 = deterministic_tournament(
                    self.P_, size=tnmt_size,
                    random_state=random_state), deterministic_tournament(
                        self.P_, size=tnmt_size, random_state=random_state)
                if random_state.random() < cross_prob:
                    c1, c2 = c1.crossover(c2, random_state)
                if random_state.random() < muta_prob:
                    c1, c2 = c1.mutate(random_state), c2.mutate(random_state)
                P__.append(c1)
                P__.append(c2)

            self.P_ = P__
            log_("elitist.fitness", self.elitist_.p_M_D, step=i)
            log_("elitist.size", self.elitist_.size(), step=i)
            log_("pop.cl_count", cl_count(self.P_), step=i)
            log_("pop.avg_ind_size", avg_ind_size(self.P_), step=i)

        return self.phi, self.elitist_

    def predict1_elitist_mean_var(self, x):
        """
        [PDF p. 224]

        The mean and variance of the predictive density described by the
        parameters of the GA's elitist.

        “As the mixture of Student’s t distributions might be multimodal, there
        exists no clear definition for the 95% confidence intervals, but a
        mixture density-related study that deals with this problem can be found
        in [118].  Here, we take the variance as a sufficient indicator of the
        prediction’s confidence.” [PDF p. 224]

        :param X: input vector (D_X)

        :returns: mean output vector (D_Y), variance of output (D_Y)
        """
        # TODO It may sometimes be better to do *some* sort of mixture over all
        # individuals since otherwise we throw away knowledge? Especially if
        # elitist is weak (can we then e.g. boost?).
        model = self.elitist_

        X = x.reshape((1, -1))
        M = model.matching_matrix(X)
        N, K = M.shape

        x_ = np.append(1, x)

        Phi = model.phi(X)
        G = mixing(M, Phi, model.V)  # (N=1) × K
        g = G[0]
        D_Y, D_X = model.W[0].shape

        W = np.array(model.W)

        # (\sum_k g_k(x) W_k) x, (7.108)
        # TODO This can probably be vectorized
        gW = 0
        for k in range(len(W)):
            gW += g[k] * W[k]
        # TODO It's probably more efficient to have each classifier predict x_
        # and then mix the predictions afterwards (as it is done in
        # LCSBookCode).
        y = gW @ x_

        var = np.zeros(D_Y)
        # TODO Can this be vectorized?
        for j in range(D_Y):
            for k in range(K):
                var[j] += g[k] * (2 * model.b_tau[k] / (model.a_tau[k] - 1) *
                                  (1 + x_ @ model.Lambda_1[k] @ x_) +
                                  (model.W[k][j] @ x_)**2) - y[j]**2
        return y, var
