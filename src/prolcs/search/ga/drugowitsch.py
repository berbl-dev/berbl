import random
from copy import deepcopy

import numpy as np  # type: ignore
from deap import creator, tools
from prolcs.utils import randseed
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from .. import Search


class GADrugowitsch(Search):
    """
    A DEAP-based implementation of the general GA algorithm found in
    Drugowitsch's book.

    The genotypes aren't fixed to be of the same form as Drugowitsch's (i.e.
    this mimics only the general algorithmic part of the GA which can be applied
    to many different forms of individuals).

    The exact operator instances used are expected to be given as part of the
    toolbox object (just as it is the case for the algorithms implementations
    that are part of DEAP).
    """
    def __init__(self,
                 toolbox,
                 pop_size=20,
                 cxpb=0.4,
                 mupb=0.4,
                 n_iter=250,
                 random_state=None,
                 add_bias=True):
        """
        Model training hyperparameters can be changed by assigning values to the
        fields of ``HParams()``; e.g. ``HParams().A_ALPHA = 1e-2``. This might
        seem ugly (and it certainly is), but, this way, we are able to keep the
        signatures in prolcs.literal.__init__.py clean and very close to the
        algorithmic description.

        Parameters
        ----------
        pop_size : int
            Population size.
        cxpb : float in [0, 1]
            Crossover probability.
        mupb : float in [0, 1]
            Mutation probability.
        n_iter : positive int
            Number of iterations to run.
        random_state : NumPy (legacy) ``RandomState``
            Due to scikit-learn compatibility, we use NumPy's legacy API.
        """
        self.toolbox = toolbox
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.mupb = mupb
        self.n_iter = n_iter
        self.random_state = random_state
        self.add_bias = add_bias

    def fit(self, X: np.ndarray, y: np.ndarray):
        random_state = check_random_state(self.random_state)
        # DEAP uses the global ``random.random`` RNG.
        seed = randseed(random_state)
        random.seed(seed)

        self.pop_ = self.toolbox.population(n=self.pop_size)

        fitnesses = [self.toolbox.evaluate(i, X, y) for i in self.pop_]
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
                    self.toolbox.evaluate(ind, X, y) for ind in invalids
                ]
                for ind, fit in zip(invalids, fitnesses):
                    ind.fitness.values = fit

                pop_new += offspring

            self.pop_[:] = pop_new
            self.elitist_.update(self.pop_)

        # TODO Doc those
        self.size_ = [len(i) for i in self.elitist_]
        self.p_M_D_ = [i.phenotype.p_M_D_ for i in self.elitist_]
        return self

    def predict_mean_var(self, X):
        check_is_fitted(self)

        return self.elitist_[0].phenotype.predict_mean_var(X)

    def predicts(self, X):
        check_is_fitted(self)

        return self.elitist_[0].phenotype.predicts(X)

    def frozen(self):
        """
        Returns a picklable of this object.

        Simply removes the toolbox.
        """
        copy = deepcopy(self)
        del copy.toolbox
        return copy
