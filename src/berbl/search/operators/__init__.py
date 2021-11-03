"""Base definitions for all toolboxes."""

from deap import base, creator
from sklearn.utils import check_random_state  # type: ignore

from ...literal.model import Model
from ...mixture import Mixture

# I'm not sure whether it is best practice to execute this whenever importing
# this module but since you get a warning if you do it multiple times (e.g. if
# we did it upon instantiating Toolbox), we leave it as that for now.
creator.create("FitnessMax", base.Fitness, weights=(1., ))
creator.create("Genotype", list, fitness=creator.FitnessMax)


class Toolbox(base.Toolbox):
    """
    Base class for toolboxes that are used to perform model structure search.
    Specifies and registers ``evaluate`` depending on the provided parameters.
    """
    def __init__(self,
                 literal=False,
                 add_bias=True,
                 phi=None,
                 fit_mixing="bouchard",
                 random_state=None):
        """
        Parameters
        ----------
        literal : bool
            Whether to use the literal backend (unoptimized but close to the
            main reference, uses the Laplace approximation–based mixing
            training). May be used to check whether a new
            implementation's/idea's behaviour is still close to the original
            reference.
        add_bias : bool
            Whether to add an all-ones bias column to the input data.
        phi : callable
            Mixing feature extractor (N × D_X → N × DV); if ``None`` uses the
            default LCS mixing feature matrix based on ``phi(x) = 1``.
        fit_mixing : str
            Only applies if ``literal`` is ``False``.  How mixing weights should
            be fitted. One of ``"bouchard"`` (experimental but may be faster and
            better-behaving) and ``"laplace"`` (the original method, very slow
            and possibly suboptimal in terms of the variational bound).
        random_state : int, RandomState instance
        """
        super().__init__()

        self.literal = literal
        self.random_state = check_random_state(random_state)

        if self.literal:

            def _evaluate(genotype, X, y):
                genotype.phenotype = Model(matchs=genotype,
                                           random_state=self.random_state,
                                           add_bias=add_bias,
                                           phi=phi).fit(X, y)
                return (genotype.phenotype.p_M_D_, )
        else:

            def _evaluate(genotype, X, y):
                # kwargs is used because Mixture takes fit_mixing, and kwargs
                # for submodels etc.
                # TODO Get rid of kwargs, make explicit
                genotype.phenotype = Mixture(matchs=genotype,
                                             random_state=self.random_state,
                                             add_bias=add_bias,
                                             phi=phi,
                                             fit_mixing=fit_mixing).fit(X, y)
                return (genotype.phenotype.p_M_D_, )

        self.register("evaluate", _evaluate)
