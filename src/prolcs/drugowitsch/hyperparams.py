import numpy as np # type: ignore


class HyperParams():
    """
    We use Alex Martelli's Borg pattern for not having to add Drugowitsch's
    hyper parameters to all signatures.

    See `this
    <https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/>`_.
    """
    __shared_state = {
        "A_ALPHA": 10**-2,
        "B_ALPHA": 10**-4,
        "A_BETA": 10**-2,
        "B_BETA": 10**-4,
        "A_TAU": 10**-2,
        "B_TAU": 10**-4,
        "DELTA_S_L_K_Q": 10**-4,
        "DELTA_S_L_M_Q": 10**-2,
        "DELTA_S_KLRG": 10**-8,
        "EXP_MIN": np.log(np.finfo(None).tiny),
        "LN_MAX": np.log(np.finfo(None).max),
        "LOGGING": "mlflow",
        # We use the deprecated API for sklearn compatibility.
        "random_state": np.random.RandomState()
    }

    def __init__(self):
        self.__dict__ = self.__shared_state
