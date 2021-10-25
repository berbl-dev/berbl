import random

import numpy as np  # type: ignore
from deap import base, creator, tools
from sklearn.utils import check_random_state  # type: ignore

from ...mixture import Mixture
from ...utils import randseed
from .. import Search


class GADrugowitsch(Search):
    """
    A DEAP-based implementation of the general GA algorithm found in
    Drugowitsch's book.

    The genotypes aren't fixed to be of the same form as Drugowitsch's (i.e.
    this mimics only the general algorithmic part of the GA which can be applied
    to many different forms of individuals).
    """
    def __init__(self,
                 init,
                 mutate,
                 pop_size=20,
                 cxpb=0.4,
                 mupb=0.4,
                 n_iter=250,
                 tournsize=5,
                 random_state=None,
                 **kwargs):
        """
        Parameters
        ----------
        init : callable receiving a ``RandomState``
            Distribution over lists of matching functions from which
            ``pop_size`` many are drawn for initialization.
        mutate : callable receiving an individual and a ``RandomState``
            Mutation operator to use.
        pop_size : int
            Population size.
        cxpb : float in [0, 1]
            Crossover probability.
        mupb : float in [0, 1]
            Mutation probability.
        n_iter : positive int
            Number of iterations to run.
        tournsize : positive int
            Tournament size.
        random_state : NumPy (legacy) ``RandomState``
            Due to scikit-learn compatibility, we use NumPy's legacy API.
        **kwargs
            Any other keyword parameters are passed through to ``Mixture``,
            ``Rule`` and ``Mixing``.
        """
        # TODO init should be in a parent class
        self.init = init
        self.mutate = mutate
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.mupb = mupb
        self.n_iter = n_iter
        self.tournsize = tournsize
        self.random_state = random_state
        self.__kwargs = kwargs

        creator.create("FitnessMax", base.Fitness, weights=(1., ))
        creator.create("Genotype", list, fitness=creator.FitnessMax)

    def fit(self, X: np.ndarray, y: np.ndarray):
        random_state = check_random_state(self.random_state)
        # DEAP uses the global ``random.random`` RNG.
        seed = randseed(random_state)
        random.seed(seed)

        self.toolbox = base.Toolbox()

        def genotype(random_state):
            # TODO Wrapping this here like this isn't that nice
            return creator.Genotype(self.init(random_state=random_state))

        self.toolbox.register("genotype", genotype, random_state=random_state)

        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.genotype)

        # “We create a new population by selecting two individuals (…) To avoid
        # the influence of fitness scaling, we select individuals from the
        # current population by deterministic tournament selection with
        # tournament size ts.”
        # [PDF p. 249]
        self.toolbox.register("select", tools.selTournament, k=2)

        self.toolbox.register("mate",
                              crossover,
                              random_state=random_state)

        self.toolbox.register("mutate", self.mutate)

        def _evaluate(genotype):
            phenotype = Mixture(matchs=genotype, random_state=random_state)
            # TODO Make retraining after end unnecessary
            phenotype.fit(X, y)
            return (phenotype.p_M_D_, )

        self.toolbox.register("evaluate", _evaluate)

        self.pop_ = self.toolbox.population(n=self.pop_size)

        fitnesses = self.toolbox.map(self.toolbox.evaluate, self.pop_)
        for ind, fit in zip(self.pop_, fitnesses):
            ind.fitness.values = fit

        self.elitist_ = tools.HallOfFame(1)
        self.elitist_.update(self.pop_)

        for i in range(self.n_iter):
            elitist = self.elitist_[0]
            print(
                f"Generation {i}. Elitist of size {len(elitist)} with p(M | D) "
                f"= {elitist.fitness.values[0]:.2}")

            pop_new = []
            # NEXT it would be better to draw once a new population (e.g.
            # random_state.choice) and then go over that? this way we don't need
            # the while and it's more declarative. also it should be the same
            while len(pop_new) < self.pop_size:
                offspring = self.toolbox.select(self.pop_,
                                                tournsize=self.tournsize)
                offspring = [self.toolbox.clone(ind) for ind in offspring]

                for c1, c2 in zip(offspring[::2], offspring[1::2]):
                    if random_state.random() < self.cxpb:
                        self.toolbox.mate(c1, c2, random_state=random_state)
                        del c1.fitness.values
                        del c2.fitness.values

                for c in offspring:
                    self.toolbox.mutate(c, random_state=random_state)
                    del c.fitness.values

                invalids = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = [self.toolbox.evaluate(ind) for ind in invalids]
                for ind, fit in zip(invalids, fitnesses):
                    ind.fitness.values = fit

                pop_new += offspring

            self.pop_[:] = pop_new
            self.elitist_.update(self.pop_)

        self.mixture_ = Mixture(matchs=self.elitist_[0],
                                random_state=random_state)
        # TODO Make retraining after end unnecessary
        self.mixture_.fit(X, y)

def crossover(child1, child2, random_state: np.random.RandomState):
    """
    Drugowitsch's simple diadic crossover operator.

    [PDF p. 250]

    Returns
    -------
    pair of results
        two new (unfitted) models resulting from crossover of the inputs
    """
    M_a = child1
    M_b = child2

    K_a = len(M_a)
    K_b = len(M_b)
    M_a_ = M_a + M_b
    random_state.shuffle(M_a_)
    # TODO Is this correct: This is how Drugowitsch does it but this way we get
    # many small individuals (higher probability of creating small individuals
    # due to selecting 9 ∈ [0, 10] being equal to selecting 1 ∈ [0, 10]).
    K_b_ = random_state.randint(low=1, high=K_a + K_b)
    # This way we might be able to maintain current individual sizes on average.
    # K_b_ = int(np.clip(random_state.normal(loc=K_a), a_min=1, a_max=K_a + K_b))
    M_b_ = []
    for k in range(K_b_):
        i = random_state.randint(low=0, high=len(M_a_))
        M_b_.append(M_a_[i])
        del M_a_[i]
    assert K_a + K_b - K_b_ == len(M_a_)
    assert K_b_ == len(M_b_)
    assert K_a + K_b == len(M_b_) + len(M_a_)

    return M_a_, M_b_
