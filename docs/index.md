# The BERBL library


This is an implementation of Drugowitsch's Bayesian Learning Classifier System[^1][^2].
BERBL stands for Bayesian Evolutionary Rule-based Learner.


[^1]: Jan Drugowitsch. 2007. Learning Classifier Systems from first principles.
    [PDF](https://drugowitschlab.hms.harvard.edu/files/drugowitschlab/files/thesis2007.pdf).
[^2]: Jan Drugowitsch. 2008. Design and Analysis of Learning Classifier Systems - A Probabilistic Approach.
    [PDF](https://drugowitschlab.hms.harvard.edu/files/drugowitschlab/files/lcsbook2008.pdf).
    
    
Note that documentation for this library is still underway. If you have any
questions, feel free to open an
[issue](https://github.com/berbl-dev/berbl/issues).


## Usage examples


You could use defaults everywhere but in the population size and the number of
iterations to run:

```Python
--8<-- "docs/examples/identity_function_defaults.py"
```


You can also override certain operators of the evolutionary algorithm similarly
to the [DEAP API](https://deap.readthedocs.io/en/master/tutorials/basic/part2.html#using-the-toolbox):

<!-- https://facelessuser.github.io/pymdown-extensions/extensions/snippets/#snippets-notation -->
```Python
--8<-- "docs/examples/identity_function.py"
```


## Note on nomenclature


In the implementation we try to avoid the overloaded term *classifier* and
instead use *rule*. A *rule* consists of a *matching function* and a (local)
*submodel*. In addition, there is a *mixing weight* associated with each rule
that comes in to play when rules overlap (i.e. when an input is matched by the
matching functions of more than one rule).
