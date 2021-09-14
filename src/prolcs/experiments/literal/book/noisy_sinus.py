# TODO Need soft interval matching for this
import click
import numpy as np  # type: ignore
from prolcs.tasks.book.noisy_sinus import f, generate

from .experiment import experiment


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-s", "--seed", type=click.IntRange(min=0), default=0)
@click.option("--show/--no-show", type=bool, default=False)
@click.option("-d", "--sample-size", type=click.IntRange(min=1), default=300)
@click.option("--standardize/--no-standardize", type=bool, default=False)
def run_experiment(n_iter, seed, show, sample_size, standardize):
    X, y = generate(sample_size)
    X_test, y_test_true = generate(1000, random_state=12345)

    # generate equidistant, denoised data as well (only for visual reference);
    # note that this doesn't need to be transformed back and forth
    X_denoised = np.linspace(0, 1, 100)[:, np.newaxis]
    y_denoised = f(X_denoised, noise_var=0)

    gaparams = {
        "n": 8,
        "p": 0.5,
        "tournsize": 5,
    }
    experiment(gaparams, X, y, X_test, y_test_true, X_denoised, y_denoised, n_iter, seed,
               show, sample_size, standardize=standardize)


if __name__ == "__main__":
    run_experiment()
