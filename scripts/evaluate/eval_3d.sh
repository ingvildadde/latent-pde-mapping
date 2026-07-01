#!/bin/bash
#SBATCH --job-name=eval-3D
#SBATCH --partition=HGXQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=600GB
#SBATCH --time=03-00:00:00
#SBATCH --output=slurm_output/slurm%j.out
#SBATCH --error=slurm_output/slurm%j.err


python --version
which python

current_path=$(pwd)

echo "$current_path"

pipenv run python3 src/evaluate/evaluate_3D_experiments.py

echo "Finished slurm"