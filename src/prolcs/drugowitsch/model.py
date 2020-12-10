from typing import *

import numpy as np  # type: ignore

from ..common import matching_matrix


class Model():
    # def __init__(self, match, W, Lambda_1, a_tau, b_tau, a_alpha, b_alpha):
    def __init__(self, matchs: List, phi):
        """
        :param matchs: A list of matching functions (i.e. objects implementing a
            ``match`` attribute) defining the structure of this model.
        """
        self.phi = phi
        # TODO Rename matchs to mstruct?
        self.matchs = matchs
        self.unfit()

    def size(self):
        return len(self.matchs)

    def fitted(self, p_M_D, W, Lambda_1, a_tau, b_tau, a_alpha, b_alpha, V,
               Lambda_V_1, a_beta, b_beta, L_q, ln_p_M, L_k_q, L_M_q):
        self.p_M_D = p_M_D
        self.W = W
        self.Lambda_1 = Lambda_1
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.V = V
        self.Lambda_V_1 = Lambda_V_1,
        self.a_beta = a_beta,
        self.b_beta = b_beta
        self.L_q = L_q
        self.ln_p_M = ln_p_M
        self.L_k_q = L_k_q
        self.L_M_q = L_M_q
        return self

    def unfit(self):
        self.p_M_D = None
        self.W = None
        self.Lambda_1 = None
        self.a_tau = None
        self.b_tau = None
        self.a_alpha = None
        self.b_alpha = None
        self.V = None
        self.Lambda_V_1 = None
        self.a_beta = None
        self.b_beta = None
        self.L_q = None
        self.ln_p_M = None
        self.L_k_q = None
        self.L_M_q = None
        return self

    def matching_matrix(self, X: np.ndarray):
        """
        :param X: input matrix (N × D_X)

        :returns: matching matrix (N × K)
        """
        # TODO Can we maybe vectorize this?
        return matching_matrix(self.matchs, X)

    def mutate(self, random_state: np.random.RandomState):
        self.matchs = list(map(lambda m: m.mutate(random_state), self.matchs))
        # TODO Can we use unfit more sensibly?
        self.unfit()
        return self

    def crossover(self, other, random_state: np.random.RandomState):
        """
        Drugowitsch's simple diadic crossover operator.

        [PDF p. 250]

        :param other: another model

        :returns: two new (unfitted) models resulting from crossover of the
            inputs
        """
        M_a = self.matchs
        M_b = other.matchs

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

        return Model(M_a_, self.phi), Model(M_b_, other.phi)
