#!/bin/bash
#SBATCH --job-name=boundary_2D
#SBATCH --partition=HGXQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCh --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=01-05:00:00
#SBATCH --output=slurm_output/slurm%j.out
#SBATCH --error=slurm_output/slurm%j.err

python --version
which python

cd "/cluster/home/ingvild/ep_simulations/"
echo $pwd


pipenv run python3 -m src.analysis.boundary_term_analysis --folder_path "outputs/2D/anisotropic/center/exp_rerun_with_epoch_times_reproduce_1" --train_data
pipenv run python3 -m src.analysis.boundary_term_analysis --folder_path "outputs/2D/anisotropic/center/shear_rerun_with_epoch_times_reproduce_1" --train_data
pipenv run python3 -m src.analysis.boundary_term_analysis --folder_path "outputs/2D/anisotropic/center/nonlin_rerun_with_epoch_times_reproduce_1" --train_data
pipenv run python3 -m src.analysis.boundary_term_analysis --folder_path "outputs/2D/anisotropic/center/rot_rerun_with_epoch_times_reproduce_1" --train_data