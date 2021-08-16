#! /usr/bin/env nix-shell
#! nix-shell -i python -p "python38.withPackages(ps: with ps; [click])"

import os
import pathlib
import shutil
from subprocess import Popen, PIPE, STDOUT
import tempfile

import click


@click.command()
@click.option("-n", "--n_iter", type=click.IntRange(min=1), default=250)
@click.option("-s",
              "--seed",
              type=click.IntRange(min=0),
              default=0,
              help="Seed of the first repetition to run.")
@click.option("-t",
              "--time",
              type=click.IntRange(min=10),
              default=30,
              help="Slurm's --time in minutes, (default: 30).")
@click.option("--mem",
              type=click.IntRange(min=1),
              default=100,
              help="Slurm's --mem in megabytes, (default: 100).")
@click.option("-r",
              "--reps",
              type=click.IntRange(min=1),
              default=10,
              help="Number of repetitions to run.")
@click.argument("experiment")
def run_experiment(n_iter, seed, time, reps, mem, experiment):
    """
    Run EXPERIMENT on the cluster.
    """

    job_dir = "/data/oc-compute02/hoffmada/prolcs"

    exp_file = f"{job_dir}/src/prolcs/experiments/{experiment}.py"

    if not pathlib.Path(exp_file).is_file():
        print(f"Experiment {experiment} does not exist. Check path ({exp_file}).")
        exit(1)

    experiment = "prolcs/experiments/{experiment}".replace("/", ".")

    sbatch = "\n".join([
        f'#!/usr/bin/env bash',  #
        f'#SBATCH --time={time}',
        f'#SBATCH --mem={mem}',
        f'#SBATCH --partition=cpu',
        f'#SBATCH --output={job_dir}/output/output-%A-%a.txt',
        f'#SBATCH --array=0-{reps}',
        (f'nix-shell "{job_dir}/default.nix" --command '
         f'"PYTHONPATH=\'{job_dir}/src:$PYTHONPATH\' python {experiment} '
         f'--seed $(({seed} + $SLURM_ARRAY_TASK_ID))"')
    ])
    print(sbatch)
    print()

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, "w+") as f:
        f.write(sbatch)
    print(f"Wrote sbatch to {tmp.name}.")
    print()

    p = Popen(["sbatch", f"{tmp.name}"], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    output = p.communicate()
    stdout = output[0].decode("utf-8")
    stderr = output[1].decode("utf-8")
    print(f"stdout:\n{stdout}\n")
    print(f"stderr:\n{stderr}\n")
    jobid = int(stdout.replace("Submitted batch job ", ""))
    print(f"Job ID: {jobid}")
    print()

    sbatch_dir = f"{job_dir}/jobs"
    os.makedirs(sbatch_dir, exist_ok=True)
    tmppath = pathlib.Path(tmp.name)
    fname = pathlib.Path(sbatch_dir, f"{jobid}.sbatch")
    shutil.copy(tmppath, fname)
    print(f"Renamed {tmp.name} to {fname}")


    # after starting the job, rename the sbatch to job id


if __name__ == "__main__":
    run_experiment()

# Local Variables:
# mode: python
# End:
