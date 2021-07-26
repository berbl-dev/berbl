import numpy as np  # type: ignore
from deap import creator, tools  # type: ignore
from prolcs.common import initRepeat_binom
from prolcs.literal.model import Model
from prolcs.match.radial1d_drugowitsch import RadialMatch1D
import prolcs.search.operators.drugowitsch as dop
from sklearn.utils import check_random_state  # type: ignore


class Toolbox(dop.Toolbox):
    def __init__(self, random_state, tournsize=5):
        """
        Creates a toolbox based on
        ``prolcs.search.operators.drugowitsch.Toolbox`` with initialization and
        evaluation as defined in (Drugowitsch, 2007).

        Returns
        -------
        ``deap.base.Toolbox`` object
            To be used e.g. with ``prolcs.search.ga.drugowitsch.GADrugowitsch``.
        """
        super().__init__(tournsize=tournsize)

        random_state = check_random_state(random_state)

        self.register("gene", RadialMatch1D.random, random_state=random_state)

        self.register("genotype",
                        initRepeat_binom,
                        creator.Genotype,
                        self.gene,
                        n=8,
                        p=0.5,
                        random_state=random_state)

        self.register("population", tools.initRepeat, list, self.genotype)

        def _evaluate(genotype, X, y):
            genotype.phenotype = Model(matchs=genotype,
                                    random_state=random_state).fit(X, y)
            return (genotype.phenotype.p_M_D_, )

        self.register("evaluate", _evaluate)
