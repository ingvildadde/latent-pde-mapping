#!/bin/bash
#SBATCH --job-name=2D-nonlin-recreate
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

### DeepONet runs ###
LG_DEEPONET_MODEL_CONFIG="./configs/model_configs/DeepONet/LG_deeponet.yaml"
LPM_DEEPONET_MODEL_CONFIG="./configs/model_configs/DeepONet/LPM_deeponet.yaml"

SIM_CONFIG="./configs/system_dynamics.yaml"
DATA_CONFIG="./configs/data_configs/2D/nonlin.yaml"
OUTPUT_FOLDER_PATH="./outputs/DeepONets/2D/anisotropic/center/nonlin_14_sensors_rerun_with_epoch_times_reproduce_1"
NUM_SENSORS=14
FAMILY_NAME="nonlin_family"
DIM=2


pipenv run python3 DeepONet/main.py \
    --model_config "$LG_DEEPONET_MODEL_CONFIG" \
    --sim_config "$SIM_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --output_folder_path "$OUTPUT_FOLDER_PATH" \
    --num_sensors "$NUM_SENSORS" \
    --family_name "$FAMILY_NAME" \
    --dim "$DIM" \

pipenv run python3 DeepONet/main.py \
    --model_config "$LPM_DEEPONET_MODEL_CONFIG" \
    --sim_config "$SIM_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --output_folder_path "$OUTPUT_FOLDER_PATH" \
    --num_sensors "$NUM_SENSORS" \
    --family_name "$FAMILY_NAME" \
    --dim "$DIM" \