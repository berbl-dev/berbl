import numpy as np # type: ignore


class State():
    """
    We use Alex Martelli's Borg pattern for not having to add the contents of
    this object to too many signatures.

    See `this
    <https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/>`_.
    """
    __shared_state = {
        # We use the deprecated API for sklearn compatibility.
        "random_state": np.random.RandomState(),
        # Only used to generate step-stamped logging.
        "step": 0,
        "oscillation_count": 0
    }

    def __init__(self):
        self.__dict__ = self.__shared_state
