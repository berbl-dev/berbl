# BERBL: Bayesian Evolutionary Rule-based Learner


This is an implementation of Drugowitsch's Bayesian Learning Classifier System[^1][^2].


[^1]: Jan Drugowitsch. 2007. Learning Classifier Systems from first principles.
    [PDF](https://drugowitschlab.hms.harvard.edu/files/drugowitschlab/files/thesis2007.pdf).
[^2]: Jan Drugowitsch. 2008. Design and Analysis of Learning Classifier Systems - A Probabilistic Approach.
    [PDF](https://drugowitschlab.hms.harvard.edu/files/drugowitschlab/files/lcsbook2008.pdf).
    
## Example usage


Check out [this very simple example](src/examples/identity_function.py).


## Note on nomenclature


In the implementation we try to avoid the overloaded term *classifier* and
instead use *rule*. A *rule* consists of a *matching function* and a (local)
*submodel*. In addition, there is a *mixing weight* associated with each rule
that comes in to play when rules overlap (i.e. when an input is matched by the
matching functions of more than one rule).


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


## Building the docs


In the development shell (i.e. after running `nix develop`):

```
PYTHONPATH=src:$PYTHONPATH mkdocs serve
```
