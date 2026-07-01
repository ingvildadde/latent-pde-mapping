# Latent PDE Mapping

This repository contains the code implementation for the paper:

> **Latent PDE mapping for physics-informed learning across geometries with limited data**

Latent PDE mapping enables accurate predictions on novel, unseen geometries by pulling back
geometry-specific PDE residuals and boundary conditions to a predefined latent geometry via the
deformation gradient, thereby enabling the automated calculation of geometry-consistent shape
gradients.

<p align="center">
  <img src="./CMAME_overview_fig.png" alt="Overview of the Latent PDE Mapping framework" width="500">
</p>

## Installation

**Requirements:** Python 3.11, [pipenv](https://pipenv.pypa.io/)

Clone the repository and install dependencies:

```bash
git clone https://github.com/ingvildadde/latent-pde-mapping.git
cd latent-pde-mapping
pipenv install
```

Activate the environment:

```bash
pipenv shell
```

Key dependencies include PyTorch, scikit-learn, and h5py (see [`Pipfile`](Pipfile) for the full list).


## Data

Download the finite element HDF5 data files from [https://doi.org/10.5281/zenodo.20928054](https://doi.org/10.5281/zenodo.20928054) and place them in the `data/` folder.


## Usage

Example of how to train LPM-PINN on the 2D rotational family with deformation parameters as geometric descriptor:

```bash
OUTPUT_PATH="outputs/2D/anisotropic/center/rot" # Modify to your prefered output folder name
DATA_CONFIG_FILE="configs/data_configs/2D/rot.yaml"
    
LPM_PINN_CONFIG_FILE="configs/model_configs/PINN/2D/LPM_pinn.yaml"

pipenv run python3 -m PINN.main "${LPM_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" --output_path "${OUTPUT_PATH}" --save --make_internal_predictions --make_external_predictions --map_pde
```

LG-PINN, PA-PINN and Basic-PINN can be trained using a similar bash command with the appropriate data config from the `data_configs` folder and model config from the `model_configs` folder without the `--map_pde` flag.


Example of how to train LPM-DON on the 2D rotational family with deformation parameters as geometric descriptor:

```bash
LPM_DEEPONET_MODEL_CONFIG="./configs/model_configs/DeepONet/LPM_deeponet.yaml"

SIM_CONFIG="./configs/system_dynamics.yaml"
DATA_CONFIG="./configs/data_configs/2D/rot.yaml"
OUTPUT_FOLDER_PATH="./outputs/DeepONets/2D/anisotropic/center/rot_14_sensors" # Modify to your prefered output folder name
NUM_SENSORS=14
FAMILY_NAME="rot_family"
DIM=2


pipenv run python3 DeepONet/main.py \
    --model_config "$LPM_DEEPONET_MODEL_CONFIG" \
    --sim_config "$SIM_CONFIG" \
    --data_config "$DATA_CONFIG" \
    --output_folder_path "$OUTPUT_FOLDER_PATH" \
    --num_sensors "$NUM_SENSORS" \
    --family_name "$FAMILY_NAME" \
    --dim "$DIM" \
```

The LG-DON can be run in a similar manner by using the appropriate model config file.