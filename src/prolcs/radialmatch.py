from typing import *
import numpy as np  # type: ignore
import scipy.special as ss  # type: ignore


class RadialMatch():
    def __init__(self,
                 mu: np.ndarray,
                 lambd_2: np.ndarray,
                 ranges: np.ndarray = None,
                 # TODO Use random_state (see RadialMatch1D where we only supply
                 # it in mutate and where it's needed)
                 rng: np.random.Generator = np.random.default_rng()):
        """
        Note: The covariance matrix has to be positive definite (cf. e.g.
        `Wikipedia
        <https://en.wikipedia.org/wiki/Gaussian_function#Multi-dimensional_Gaussian_function>`_),
        thus we simply require its inverse right away (also, this way we don't
        have to invert it which is costly and the model structure optimizer
        can very well work on the inverse directly anyway).

        :param mu: Position of the Gaussian
        :param lambd_2: Squared precision matrix (squared inverse covariance
            matrix)
        :param ranges: The value ranges of the problem considered. If ``None``,
            use ``[-inf, inf]`` for each dimension.
        """
        assert mu.shape[0] == lambd_2.shape[0]
        assert mu.shape[0] == lambd_2.shape[1]
        self.mu = mu
        self.lambd_2 = lambd_2
        self.rng = rng
        if ranges is not None:
            assert ranges.shape == (mu.shape[0], 2)
            self.ranges = ranges
        else:
            self.ranges = np.repeat([-np.inf, np.inf], len(mu)).reshape(
                (mu.shape[0], 2))

    def __repr__(self):
        return f"RadialMatch({self.mu}, {self.lambd_2}, {self.ranges})"

    @classmethod
    def random(
        cls,
        ranges: np.ndarray,
        lambd_2_gen: Callable[
            [np.ndarray],
            # TODO Extract this
            np.ndarray] = lambda r: np.repeat(100, r.shape[0]**2).reshape(
                (r.shape[0], r.shape[0])),
        rng: np.random.Generator = np.random.default_rng()):
        """
        Based on [PDF p. 256] but slightly different:

        * We allow multi-dimensional inputs (he only does ``D_X = 1``).
        * We do not introduce helper parameters (for now).
        * We only roughly try to mimic the ranges of mutation used by him.
        * We clip both lambd_2 and mu values to the respective dimension's range.

        This is basically what (Butz, 2005, “Kernel-based, ellipsoidal
        conditions in the real-valued XCS classifier system”) but with soft
        boundaries (i.e. every classifier matches everywhere, even if only
        slightly), that is, without a threshold parameter. Also, we evolve
        ``(sigma**-1)**2`` directly.

        :param ranges: The value ranges for each dimension of the problem
            considered (a matrix of shape ``(mu.shape[0], 2)``).
        :param lambd_2_gen: Generator for deriving values for lambd_2 for each
            dimension from ``ranges``. The default is to simply use a value of
            ``100`` for each dimension (which is pretty arbitrary).
        """
        D_X, _ = ranges.shape
        assert _ == 2
        return RadialMatch(
            mu=rng.uniform(low=ranges[:, [0]].reshape((-1)),
                           high=ranges[:, [1]].reshape((-1)),
                           size=D_X),
            # NOTE rng.uniform(…, size=(D_X, D_X)) does not work in general. Has
            # to be problem and especially ranges-dependent.
            lambd_2=lambd_2_gen(ranges),
            ranges=ranges,
            rng=rng)

    def mutate(self):
        self.mu = np.clip(
            self.rng.normal(
                loc=self.mu,
                # TODO This was chosen to be similar to [PDF p. 228] but isn't
                # probably.
                # TODO Drugowitsch gives variance, but NumPy wants standard
                # deviation.
                scale=0.01 * np.sum(self.ranges * np.array([-1, 1]), 1)),
            # clip to each dimension's range
            self.ranges[:, [0]].reshape((-1)),
            self.ranges[:, [1]].reshape((-1)))
        self.lambd_2 = self.rng.normal(
            loc=self.lambd_2,
            # TODO This was chosen relatively arbitrary (but motivated by [PDF
            # p. 228])
            # TODO Drugowitsch gives variance, but NumPy wants standard
            # deviation.
            scale=0.005 * np.sum(self.ranges * np.array([-1, 1]), 1),
            size=self.lambd_2.shape)
        return self

    def match(self, X: np.ndarray):
        """
        We vectorize the following (i.e. feed the whole input through at once)::

            for n in range(len(X)):
                M[n] = np.exp(-0.5 * (x - mu) @ lambd_2 @ (x - mu))

        :param X: input matrix (N × D_X)
        :returns: matching vector (N) of this matching function (i.e. of this
            classifier)
        """
        mu_ = np.broadcast_to(self.mu, X.shape)
        delta = X - mu_
        # We have to clip this so we don't return 0 here (0 should never be
        # returned because every match function matches everywhere at least a
        # little bit). Also, we clip from above such that this never returns a
        # value larger than 1 (it's a probability, after all).
        m_min = np.log(np.finfo(None).tiny)
        # TODO The maximum negative number might be different than simply the
        # negation of the minimum positive number.
        m_max = -np.finfo(None).tiny
        # NOTE The negative sign in front of 0.5 is not in Drugowitsch's (8.10)
        # [PDF p. 256].  However it can be found e.g. in (Butz et al.,
        # Kernel-based, ellipsoidal conditions in the real-valued XCS classifier
        # system, 2005) and seems reasonable: With delta -> 0, we want to have m
        # -> 1 and without the negative sign, it may be that m > 1.
        m = np.clip(-0.5 * np.sum(delta.T * (self.lambd_2 @ delta.T), 0),
                    m_min, m_max)
        return np.exp(m)[:, np.newaxis]

    def plot(self, ax, **kwargs):
        if self.mu.shape == (1, ):
            l = self.ranges[:,[0]].reshape(1)
            h = self.ranges[:,[1]].reshape(1)
            X = np.arange(l, h, 0.01)[:, np.newaxis]
            M = self.match(X)
            ax.plot(X, M, **kwargs)
            ax.axvline(self.mu, color=kwargs["color"])
        else:
            raise Exception(
                "Can only plot one-dimensional RadialMatch objects")
