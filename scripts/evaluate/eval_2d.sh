#!/bin/bash
#SBATCH --job-name=eval-2D
#SBATCH --partition=HGXQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=30GB
#SBATCH --time=00-01:00:00
#SBATCH --output=slurm_output/slurm%j.out
#SBATCH --error=slurm_output/slurm%j.err

module purge
module load Python/3.11.5-GCCcore-13.2.0

python --version
which python

current_path=$(pwd)

echo "$current_path"

pipenv run python3 src/evaluate/evaluate_2D_experiments.py

echo "Finished slurm"