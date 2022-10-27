import random
from copy import deepcopy
from typing import List

import numpy as np  # type: ignore
from deap import creator, tools  # type: ignore
from mlflow import log_metric  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from tqdm import tqdm, trange  # type: ignore

from ...utils import randseed


class GADrugowitsch:
    """
    A [DEAP](https://github.com/DEAP/deap)-based implementation of the GA
    algorithm found in [Drugowitsch's book](/).

    The genotypes aren't fixed to be of the same form as Drugowitsch's (i.e.
    this mimics only the general algorithmic part of the GA which can be applied
    to many different forms of individuals).

    The exact operator instances used are expected to be given as part of the
    toolbox object (just as it is the case for the algorithms implementations
    that are part of [DEAP](https://github.com/DEAP/deap)).
    """

    def __init__(self,
                 toolbox,
                 random_state,
                 pop_size=20,
                 cxpb=0.4,
                 mupb=0.4,
                 n_iter=250,
                 add_bias=True):
        """
        Parameters
        ----------
        toolbox : object
            A DEAP `Toolbox` object that specifies all the operators required
            by this metaheuristic.
        random_state : int, NumPy (legacy) `RandomState` object
            Due to scikit-learn compatibility, we use NumPy's legacy API.
        pop_size : int
            Population size.
        cxpb : float in [0, 1]
            Crossover probability.
        mupb : float in [0, 1]
            Mutation probability.
        n_iter : positive int
            Number of iterations to run.
        """
        self.toolbox = toolbox
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.mupb = mupb
        self.n_iter = n_iter
        self.random_state = random_state
        self.add_bias = add_bias

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit this to the data.

        This gets called by [berbl.BERBL.fit][].
        """

        random_state = check_random_state(self.random_state)
        # DEAP uses the global ``random.random`` RNG.
        seed = randseed(random_state)
        random.seed(seed)

        self.pop_ = self.toolbox.population(n=self.pop_size)

        fitnesses = [
            self.toolbox.evaluate(i, X, y)
            for i in tqdm(self.pop_, desc="Evaluate initial ", leave=True)
        ]
        for ind, fit in zip(self.pop_, fitnesses):
            ind.fitness.values = fit

        self.elitist_ = tools.HallOfFame(1)
        self.elitist_.update(self.pop_)
        elitist = self.elitist_[0]

        for i in trange(
                self.n_iter,
                desc=("GA (best "
                      f"{len(elitist)}/{elitist.fitness.values[0]:.1})")):
            elitist = self.elitist_[0]

            # TODO Consider a more modular setup for logging
            log_metric("elitist.ln_p_M_D", elitist.fitness.values[0], i)

            pop_new: List = []
            while len(pop_new) < self.pop_size:
                # “We create a new population by selecting two individuals from
                # the current population. … is repeated until the new population
                # again holds P individuals. Then, the new population replaces
                # the current one and the next iteration begins.”
                offspring = self.toolbox.select(self.pop_)
                offspring = [self.toolbox.clone(ind) for ind in offspring]

                offspring_ = []
                for c1, c2 in zip(offspring[::2], offspring[1::2]):
                    # “Apply crossover with probability pc …”
                    if random_state.random() < self.cxpb:
                        # TODO I don't yet modify c1 and c2 in-place. Must I?
                        c1_, c2_ = self.toolbox.mate(c1,
                                                     c2,
                                                     random_state=random_state)
                        c1_ = creator.Genotype(c1_)
                        c2_ = creator.Genotype(c2_)
                        del c1_.fitness.values
                        del c2_.fitness.values
                        offspring_ += [c1_, c2_]

                offspring = offspring_
                for c in offspring:
                    # “… and mutation with probability pm.”
                    if random_state.random() < self.mupb:
                        self.toolbox.mutate(c, random_state=random_state)
                        del c.fitness.values

                invalids = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = [
                    self.toolbox.evaluate(ind, X, y) for ind in tqdm(
                        invalids, desc="Eval generation  ", leave=False)
                ]
                for ind, fit in zip(invalids, fitnesses):
                    ind.fitness.values = fit

                pop_new += offspring

            self.pop_[:] = pop_new
            self.elitist_.update(self.pop_)

        # TODO Doc those
        self.size_ = [len(i) for i in self.elitist_]
        self.ln_p_M_D_ = [i.phenotype.ln_p_M_D_ for i in self.elitist_]

        return self

    def predict(self, X):
        """
        Uses the current elitist to perform a prediction.

        This gets called by [berbl.BERBL.predict][].
        """
        return self.elitist_[0].phenotype.predict(X)

    def predict_mean_var(self, X):
        """
        Uses the current elitist to perform a prediction.

        This gets called by [berbl.BERBL.predict_mean_var][].
        """
        return self.elitist_[0].phenotype.predict_mean_var(X)

    def predicts(self, X):
        """
        Uses the current elitist to perform a prediction.

        This gets called by [berbl.BERBL.predicts][].
        """
        return self.elitist_[0].phenotype.predicts(X)

    def predict_distribution(self, x):
        """
        Uses the current elitist to perform a prediction.

        This gets called by [berbl.BERBL.predict_distribution][].
        """
        return self.elitist_[0].phenotype.predict_distribution(x)

    def frozen(self):
        """
        Returns a picklable copy of this object (we simply remove the toolbox).
        """
        copy = deepcopy(self)
        del copy.toolbox
        return copy
