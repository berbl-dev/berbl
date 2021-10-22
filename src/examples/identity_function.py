import numpy as np  # type: ignore
from deap import creator, tools  # type: ignore
from prolcs import ProLCS
from prolcs.search.operators.drugowitsch import DefaultToolbox
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

X = np.arange(300).reshape((-1, 1))
y = np.arange(300).reshape((-1, 1))

random_state = check_random_state(2)

toolbox = DefaultToolbox(literal=False, random_state=random_state)


# Let's assume we want to use a custom mutate that does nothing (i.e. disable
# mutation in a slightly awkward fashion).
def custom_mutate(genotype, random_state):
    return genotype


toolbox.register("mutate", custom_mutate)

pipe = make_pipeline(StandardScaler(), ProLCS(toolbox=toolbox, n_iter=20))
estimator = pipe.fit(X, y)
y_pred = estimator.predict(X)
print("MAE on training data: ", mean_absolute_error(y, y_pred))
