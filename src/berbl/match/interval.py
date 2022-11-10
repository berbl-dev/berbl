import numpy as np  # type: ignore
import scipy.stats as st  # type: ignore


def _mirror(x, x_min, x_max, exclude_min=False, exclude_max=False):
    """
    Restrict the given value to the given bounds using the mirror correction
    strategy (see, e.g., Kononova et al.'s 2022 article *Differential Evolution
    Outside the Box*).

    Parameters
    ----------
    exclude_min, exclude_max : bool
        Whether the allowed range excludes the given minimum or maximum.
    """
    x_ = x - (x > x_max - 1 + exclude_max) * 2 * (x - x_max)
    x_ = x_ + (x_ < x_min + 1 - exclude_min) * 2 * (x_min - x_)

    if x_ > x_max or x_ < x_min:
        return mirror(x_, x_min, x_max)
    else:
        return x_


def mirror(a, a_min, a_max, exclude_min=False, exclude_max=False):
    """
    Restrict the given array to the given bounds using the mirror correction
    strategy (see, e.g., Kononova et al.'s 2022 article *Differential Evolution
    Outside the Box*).

    Parameters
    ----------
    exclude_min, exclude_max : bool
        Whether the allowed range excludes the given minimum or maximum.
    """
    # NOTE We have to define this element-wise because we may otherwise easily
    # construct cases where it doesn't terminate (i.e. where two or more
    # entries oscillate forever).
    return np.vectorize(_mirror)(a, a_min, a_max, exclude_min, exclude_max)


class Interval():
    """
    [`self.match`][berbl.match.interval.Interval.match] is a hard interval–based
    matching function which means that its value is 1 within the interval [`l`,
    `u`) and `np.finfo(None).tiny` anywhere else.

    This differs from
    [`SoftInterval1D`][berbl.match.softinterval1d_drugowitsch.SoftInterval1D]
    in that this matching function has a steep drop off at the interval bounds.

    Notes
    -----
    The implied lower and upper bounds may not respect [`x_min`, `x_max`] but
    the lowest possible lower bound is actually `x_min - (x_max - x_min) / 2`
    (and the highest possible upper bound is `x_max + (x_max - x_min) / 2`).
    This is because we don't want to introduce epistasis into the genotype (we'd
    need to make the domain of `center` depend on the current value of `spread`
    in order to ensure [`x_min`, `x_max`] is respected).
    """

    def __init__(
        self,
        center: np.array,
        spread: np.array,
        x_min: np.array = None,
        x_max: np.array = None,
        # TODO Find good defaults here (2**8 may be too small?)
        res_center: int = int(2**8),
        res_spread: int = int(2**8),
        # TODO Find good defaults here (0.1 may be too large for small
        # dimensions and too small for large dimensions?)
        std_factor_center: float = 0.1,
        std_factor_spread: float = 0.1,
    ):
        """
        Parameters
        ----------
        center : array of shape (DX,)
            Array of natural numbers in [0, `res_center`]. See `res_center`.
        spread : array of shape (DX,)
            Array of natural numbers in [0, `res_spread`]. See `res_spread`.
        x_min, x_max : arrays of shape (DX,)
            The expected minimum and maximum values of the inputs. If `None`
            (the default), they are set to `-2.0` for `x_min` and `2.0` for
            `x_max` which corresponds to assuming uniformly distributed inputs
            which have been standardized (i.e. an input range of [-2, 2] is
            assumed which is the range [`-np.sqrt(3)`, `np.sqrt(3)`] of a
            standardized uniform distribution including a little bit of wiggle
            room).
        res_center, res_spread : int
            Resolutions. Center (and spread) genes are represented by natural
            numbers in [0, `res_center`] (and [0, `res_spread`]) and transformed
            to their phenotypes using `x_min` and `x_max`.
        std_factor_center, std_factor_center : float
            Rate of the input space to use as standard deviation for the normal
            distribution–based mutation. A value of `std_factor_center = 0.1`
            means that the expected width of mutation steps is `0.1 *
            res_center` (i.e. a tenth of input space, *not* taking into account
            mirror correction which is applied thereafter).
        """
        self.res_center = res_center
        self.res_spread = res_spread

        assert np.all((0 <= center) & (
            center <= res_center)), "Center resolution not respected"
        assert np.all((0 <= spread) & (
            spread <= res_spread)), "Spread resolution not respected"
        assert len(center) == len(
            spread), "There must be as many centers as spreads"

        self.center = center
        self.spread = spread
        self.DX = len(self.center)

        if x_min is None:
            self.x_min = np.repeat(-2.0, self.DX)
        else:
            self.x_min = x_min
        if x_max is None:
            self.x_max = np.repeat(2.0, self.DX)
        else:
            self.x_max = x_max

        assert np.all(self.x_min < self.x_max), "Input range not well defined"

        self._recompute_bounds()

        std_center = 0.1 * self.res_center
        std_spread = 0.1 * self.res_spread
        self.dist_mut_center = st.norm(loc=0, scale=std_center)
        self.dist_mut_spread = st.norm(loc=0, scale=std_spread)

    def __repr__(self):
        return (f"Interval(center={self.center}, "
                f"spread={self.spread}, "
                f"x_min={self.x_min}, "
                f"x_max={self.x_max}, "
                f"res_center={self.res_center}, "
                f"res_spread={self.res_spread})")

    @classmethod
    def random(
            cls,
            DX: int,
            random_state: np.random.RandomState,
            x_min: np.array = None,
            x_max: np.array = None,
            res_center: int = int(2**8),
            res_spread: int = int(2**8),
    ):
        """
        Parameters
        ----------
        DX : int
            Dimensionality of inputs.
        x_min, x_max : arrays of shape (DX,)
            See constructor documentation for `x_min` and `x_max`.
        res_center, res_spread : int
            Resolutions. Center (and spread) genes are represented by natural
            numbers in [0, `res_center`] (and [0, `res_spread`]) and transformed
            to their phenotypes using `x_min` and `x_max`.
        """
        if x_min is None:
            x_min = np.repeat(-2.0, DX)
        if x_max is None:
            x_max = np.repeat(2.0, DX)

        center = random_state.randint(low=0, high=res_center, size=DX)
        spread = random_state.randint(low=0, high=res_spread, size=DX)
        return Interval(
            center=center,
            spread=spread,
            x_min=x_min,
            x_max=x_max,
            res_center=res_center,
            res_spread=res_spread,
        )

    @classmethod
    def random_at(
            cls,
            center_phen: np.array,
            random_state: np.random.RandomState,
            x_min: np.array = None,
            x_max: np.array = None,
            res_center: int = int(2**8),
            res_spread: int = int(2**8),
    ):
        """
        Parameters
        ----------
        center_phen : array
            Point in input space (i.e. phenotype space) to use as the center of
            this interval.
        x_min, x_max : arrays of shape (DX,)
            See constructor documentation for `x_min` and `x_max`.
        res_center, res_spread : int
            Resolutions. Center (and spread) genes are represented by natural
            numbers in [0, `res_center`] (and [0, `res_spread`]) and transformed
            to their phenotypes using `x_min` and `x_max`.
        """
        DX = center_phen.shape[0]

        if x_min is None:
            x_min = np.repeat(-2.0, DX)
        if x_max is None:
            x_max = np.repeat(2.0, DX)

        center = (center_phen - x_min) * res_center / (x_max - x_min)
        spread = random_state.randint(low=0, high=res_spread, size=DX)
        return Interval(
            center=center,
            spread=spread,
            x_min=x_min,
            x_max=x_max,
            res_center=res_center,
            res_spread=res_spread,
        )

    def _recompute_bounds(self):
        self.center_phen = self.x_min + self.center * (
            self.x_max - self.x_min) / self.res_center
        self.spread_phen = self.spread * (self.x_max
                                          - self.x_min) / (2 * self.res_spread)
        self.l = self.center_phen - self.spread_phen
        self.u = self.center_phen + self.spread_phen

    def mutate(self, random_state: np.random.RandomState):
        """
        Mutate this matching function in-place.

        1. Draw new centers and spreads from a normal centered on the current
           values.
        2. Round the values to the nearest int (we use a fixed resolution and
           natural numbers to represent our genes).
        3. Fix out-of-range values using the mirror correction strategy (see
           [`mirror`][berbl.match.interval.mirror]).
        4. Recompute and cache new lower and upper bounds (for performance
           reasons only).
        """

        self.center = self.center.astype(float)
        self.center += self.dist_mut_center.rvs(self.DX)
        self.center[:] = np.rint(self.center)
        self.center[:] = mirror(self.center,
                                a_min=0,
                                a_max=self.res_center,
                                exclude_max=True).astype(int)

        self.spread = self.spread.astype(float)
        self.spread += self.dist_mut_spread.rvs(self.DX)
        self.spread[:] = np.rint(self.spread)
        self.spread[:] = mirror(self.spread,
                                a_min=0,
                                a_max=self.res_spread,
                                exclude_max=True).astype(int)

        # Recompute and cache lower/upper bounds (so we don't have to do this
        # every time we call match).
        self._recompute_bounds()

        return self

    # TODO Implement __call__ instead
    def match(self, X: np.ndarray) -> np.ndarray:
        """
        Compute matching vector for given input (assumes that the input doesn't
        have a bias column which shouldn't take part in matching).

        Parameters
        ----------
        X : input matrix of shape (N, DX)

        Returns
        -------
        array of shape (N, 1)
            Matching vector of this matching function for the given input.
        """
        return np.where(
            np.all(self.l <= X, axis=1) & np.all(X < self.u, axis=1), 1.0,
            np.finfo(None).tiny).reshape(-1, 1)

    def plot(self, ax):
        import matplotlib.pyplot as plt
        X = np.linspace(self.x_min, self.x_max, 100)
        ax.plot(X, self.match(X))
