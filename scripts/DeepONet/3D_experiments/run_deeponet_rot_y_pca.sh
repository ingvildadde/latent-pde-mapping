#!/bin/bash
#SBATCH --job-name=deeponet
#SBATCH --partition=HGXQ
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCh --cpus-per-task=4
#SBATCH --mem=600GB
#SBATCH --time=02-00:00:00
#SBATCH --output=slurm_output/slurm%j.out
#SBATCH --error=slurm_output/slurm%j.err

module purge
module load Python/3.11.5-GCCcore-13.2.0

python --version
which python

current_path=$(pwd)

echo "$current_path"

### DeepONet Runs ###

LG_DEEPONET_MODEL_CONFIG="./configs/model_configs/DeepONet/3D/LG_deeponet_pca.yaml"
LPM_DEEPONET_MODEL_CONFIG="./configs/model_configs/DeepONet/3D/LPM_deeponet_pca.yaml"

SIM_CONFIG="./configs/system_dynamics.yaml"
DATA_CONFIG="./configs/data_configs/3D/rot_y.yaml"
OUTPUT_FOLDER_PATH="./outputs/DeepONets/3D/anisotropic/center/rot_y_42_sensors_pca_reproduce_1"
NUM_SENSORS=42 #28
FAMILY_NAME="rot_y_family"
DIM=3

pipenv run python3 DeepONet/main.py \
    --model_config "$LPM_DEEPONET_MODEL_CONFIG" \
    --sim_config "$SIM_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --output_folder_path "$OUTPUT_FOLDER_PATH" \
    --num_sensors "$NUM_SENSORS" \
    --family_name "$FAMILY_NAME" \
    --dim "$DIM" \

pipenv run python3 DeepONet/main.py \
    --model_config "$LG_DEEPONET_MODEL_CONFIG" \
    --sim_config "$SIM_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --output_folder_path "$OUTPUT_FOLDER_PATH" \
    --num_sensors "$NUM_SENSORS" \
    --family_name "$FAMILY_NAME" \
    --dim "$DIM" \