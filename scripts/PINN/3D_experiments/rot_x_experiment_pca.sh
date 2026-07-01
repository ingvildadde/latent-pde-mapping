#!/bin/bash
#SBATCH --job-name=3D-rot-x-recreate
#SBATCH --partition=HGXQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCh --cpus-per-task=4
#SBATCH --mem=600GB
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm_output/slurm%j.out
#SBATCH --error=slurm_output/slurm%j.err

module purge
module load Python/3.11.5-GCCcore-13.2.0

python --version
which python

current_path=$(pwd)

# Setup environment variables
export DATASETS_ROOT_FOLDER_PATH="${current_path}/data/raw"


### PINNs Run ###

OUTPUT_PATH="outputs/3D/anisotropic/center/rot_x_pca_reproduce_1"
DATA_CONFIG_FILE="configs/data_configs/3D/rot_x.yaml"

LPM_PINN_CONFIG_FILE="configs/model_configs/PINN/3D/LPM_pinn_pca.yaml"
LG_PINN_CONFIG_FILE="configs/model_configs/PINN/3D/LG_pinn_pca.yaml"
AFFINE_PINN_CONFIG_FILE="configs/model_configs/PINN/3D/affine_pinn_pca.yaml"
BASIC_PINN_CONFIG_FILE="configs/model_configs/PINN/3D/basic_pinn.yaml"

# LPM-PINN
pipenv run python3 -m PINN.main "${LPM_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions --map_pde

# LG-PINN
pipenv run python3 -m PINN.main "${LG_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions

# Affine-PINN
pipenv run python3 -m PINN.main "${AFFINE_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions

# Basic-PINN
pipenv run python3 -m PINN.main "${BASIC_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions