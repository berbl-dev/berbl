"""Search operators as defined in Drugowitsch's book."""

from typing import List

import numpy as np  # type: ignore
from deap import creator, tools  # type: ignore

from ...match.softinterval1d_drugowitsch import SoftInterval1D
from ...utils import initRepeat_binom
from . import Toolbox


class DefaultToolbox(Toolbox):
    """
    Toolbox specified in [Drugowitsch's book](/).

    Extends the base toolbox (containing `evaluate`) by providing `gene`,
    `genotype`, `population`, `select`, `mate` and `mutate`.
    """
    def __init__(self,
                 matchcls=SoftInterval1D,
                 random_state=None,
                 n=100,
                 p=0.5,
                 literal=False,
                 add_bias=True,
                 phi=None,
                 tournsize=5,
                 fit_mixing="laplace",
                 match_args={},
                 **kwargs):
        """
        Initializes this toolbox by creating and registering operators.

        Individuals are created by drawing their size from a binomial
        distribution and then .

        Parameters
        ----------
        random_state : None, int, NumPy (legacy) RandomState object
            See [berbl.search.operators.Toolbox][].
        matchcls : object
            Matching function class to be used. By default,
            [`SoftInterval1D`][berbl.match.softinterval1d_drugowitsch.SoftInterval1D].
        n : int, > 0
            n parameter (number of independent experiments) of the binomial
            distribution from which initial individual sizes are drawn.
        p : float
            p parameter (success rate) of the binomial distribution from which
            initial individual sizes are drawn.
        literal : bool
            See [berbl.search.operators.Toolbox][].
        add_bias : bool
            See [berbl.search.operators.Toolbox][].
        phi : callable
            See [berbl.search.operators.Toolbox][].
        tournsize : int, > 1
            Size of the tournaments used in the `select` operator.
        fit_mixing : str
            See [berbl.search.operators.Toolbox][].
        """
        super().__init__(literal=literal,
                         add_bias=add_bias,
                         phi=phi,
                         fit_mixing=fit_mixing,
                         random_state=random_state,
                         **kwargs)

        self.register("gene",
                      matchcls.random,
                      random_state=self.random_state,
                      **match_args)
        self.register("genotype",
                      initRepeat_binom,
                      creator.Genotype,
                      self.gene,
                      n=n,
                      p=p,
                      random_state=self.random_state)
        self.register("population", tools.initRepeat, list, self.genotype)

        # “We create a new population by selecting two individuals (…) To avoid
        # the influence of fitness scaling, we select individuals from the
        # current population by deterministic tournament selection with
        # tournament size ts.”
        # [PDF p. 249]
        self.register("select", tools.selTournament, k=2, tournsize=tournsize)

        self.register("mate", crossover)

        self.register("mutate", mutate)


def mutate(matchs, random_state: np.random.RandomState):
    """
    Drugowitsch's individual-level mutation operator.

    See [Drugowitsch's book](/).

    Go once over the individual and call each gene's `mutate` method with the
    provided random state.

    Note that the genes are *not* copied which means that in-place alterations
    of them cannot be ruled out.

    Parameters
    ----------
    matchs : list of objects
        An individual consisting of genes, each gene having a `mutate` method
        that expects an `np.random.RandomState`.

    Returns
    -------
    tuple of object
        The mutated individual wrapped in a one-tuple (DEAP specification).
    """
    return [m.mutate(random_state) for m in matchs],
    # TODO Should extract m.mutate to here as well? Or otherwise mark as
    # Drugowitsch operator.


def crossover(M_a: List, M_b: List, random_state: np.random.RandomState):
    """
    Drugowitsch's simple diadic crossover operator.

    See [Drugowitsch's book](/).

    Draw two new sizes for the offspring individuals and then randomly
    distribute the parent's genes among them.

    Parameters
    ----------
    M_a : list
        First individual to crossover.
    M_b : list
        Second individual to crossover.

    Returns
    -------
    pair of objects
        Two new (unfitted) models resulting from crossover of the inputs.
    """
    # “As two selected individuals can be of different length, we cannot apply
    # standard uniform cross-over but have to use different means: we want the
    # total number of [rules] to remain unchanged, but as the location of
    # the [rules] in the genome of an individual do not provide us with
    # any information, we allow their location to change.”
    K_a = len(M_a)
    K_b = len(M_b)
    M_a_ = M_a + M_b
    random_state.shuffle(M_a_)
    K_b_ = random_state.randint(low=1, high=K_a + K_b)
    M_b_ = []
    for k in range(K_b_):
        i = random_state.randint(low=0, high=len(M_a_))
        M_b_.append(M_a_[i])
        del M_a_[i]
    assert K_a + K_b - K_b_ == len(M_a_)
    assert K_b_ == len(M_b_)
    assert K_a + K_b == len(M_b_) + len(M_a_)

    # TODO To use this with DEAP builtin algorithms, modify in place
    # Use M_a.extend([1] * …) and del M_b[k:], then M_a[:] = M_a_ etc.
    return (M_a_, M_b_)
