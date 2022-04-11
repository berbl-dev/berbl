import numpy as np  # type: ignore
from berbl import BERBL
from berbl.search.operators.drugowitsch import DefaultToolbox
from sklearn.compose import TransformedTargetRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils import check_random_state  # type: ignore

# Generate some data for the identity function.
X = np.arange(300).reshape((-1, 1))
y = np.arange(300).reshape((-1, 1))

random_state = check_random_state(2)

# Initialize toolbox, let's assume that a population size of 10 suffices for
# this task.
toolbox = DefaultToolbox(n=10, random_state=random_state)


# Let's assume we want to use a custom mutate that does nothing (i.e. disable
# mutation in a slightly awkward fashion).
def custom_mutate(genotype, random_state):
    return genotype


# Override mutation operator in toolbox with our custom one.
toolbox.register("mutate", custom_mutate)

# Instantiate BERBL.
regressor = BERBL(toolbox=toolbox, n_iter=20)

# Next, just some standard scikit-learn stuff: Create a pipeline that
# standardizes inputs.
pipe = make_pipeline(
    StandardScaler(),
    TransformedTargetRegressor(regressor=regressor,
                               transformer=StandardScaler()))

# Fit the pipeline.
estimator = pipe.fit(X, y)

# Make predictions (here, on the training data for simplicities sake).
y_pred = estimator.predict(X)

# Get some metrics.
print("MAE on training data: ", mean_absolute_error(y, y_pred))
