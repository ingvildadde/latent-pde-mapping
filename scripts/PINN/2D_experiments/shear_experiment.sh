#!/bin/bash
#SBATCH --job-name=2D-shear-recreate
#SBATCH --partition=HGXQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCh --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH --time=02-00:00:00
#SBATCH --output=slurm_output/slurm%j.out
#SBATCH --error=slurm_output/slurm%j.err


python --version
which python

current_path=$(pwd)

# Setup environment variables
export DATASETS_ROOT_FOLDER_PATH="${current_path}/data/raw"


### PINN runs ###

OUTPUT_PATH="outputs/2D/anisotropic/center/shear"
DATA_CONFIG_FILE="configs/data_configs/2D/shear.yaml"

LPM_PINN_CONFIG_FILE="configs/model_configs/PINN/2D/LPM_pinn.yaml"
LG_PINN_CONFIG_FILE="configs/model_configs/PINN/2D/LG_pinn.yaml"
PA_PINN_CONFIG_FILE="configs/model_configs/PINN/2D/PA_pinn.yaml"
BASIC_PINN_CONFIG_FILE="configs/model_configs/PINN/2D/basic_pinn.yaml"

# LPM-PINN
pipenv run python3 -m PINN.main "${LPM_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions --map_pde

# LG-PINN
pipenv run python3 -m PINN.main "${LG_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions

# PA-PINN
pipenv run python3 -m PINN.main "${PA_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions

# Basic-PINN
pipenv run python3 -m PINN.main "${BASIC_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions
