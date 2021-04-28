#!/usr/bin/env bash
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute02/hoffmada/prolcs/output/output-%A-%a.txt
#SBATCH --array=0-9
job_dir=/data/oc-compute02/hoffmada/prolcs/
experiment=src/prolcs/experiments/drugowitsch/generated_function.py
nix-shell "$job_dir"/default.nix --command "PYTHONPATH=$job_dir/src:$PYTHONPATH python $job_dir/$experiment --seed 123 --sample-size 150"
