# BERBL: Bayesian Evolutionary Rule-based Learner


This is an implementation of Drugowitsch's Bayesian Learning Classifier System[^1][^2].
BERBL stands for Bayesian Evolutionary Rule-based Learner.


[^1]: Jan Drugowitsch. 2007. Learning Classifier Systems from first principles.
    [PDF](https://drugowitschlab.hms.harvard.edu/files/drugowitschlab/files/thesis2007.pdf).
[^2]: Jan Drugowitsch. 2008. Design and Analysis of Learning Classifier Systems - A Probabilistic Approach.
    [PDF](https://drugowitschlab.hms.harvard.edu/files/drugowitschlab/files/lcsbook2008.pdf).
    
    
[Documentation can be found here.](https://berbl-dev.github.io/berbl/)


## Running the tests


```bash
nix develop
```
drops you into a development shell with all the dependencies.


You can then run all tests the recommended way using
```bash
tox
```


In order to run a selection of tests use (see the `pytest` documentation for
details).
```bash
pytest tests -k FILTEREXP
```
