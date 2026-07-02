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

### Physics-Informed Neural Networks (PINNs)

Four PINN variants are available:

| Model | Config (2D) | Config (3D) |
|---|---|---|
| LPM-PINN | `configs/model_configs/PINN/2D/LPM_pinn.yaml` | `configs/model_configs/PINN/3D/LPM_pinn.yaml` |
| LG-PINN | `configs/model_configs/PINN/2D/LG_pinn.yaml` | `configs/model_configs/PINN/3D/LG_pinn.yaml` |
| PA-PINN | `configs/model_configs/PINN/2D/PA_pinn.yaml` | `configs/model_configs/PINN/3D/PA_pinn.yaml` |
| Basic-PINN | `configs/model_configs/PINN/2D/basic_pinn.yaml` | `configs/model_configs/PINN/3D/basic_pinn.yaml` |

Append `_pca` to the config filename to use PCA-based geometric descriptors instead of deformation parameters (e.g. `LPM_pinn_pca.yaml`).

**Example** — train LPM-PINN on the 2D rotational family:

```bash
OUTPUT_PATH="outputs/2D/anisotropic/center/rot"  # adjust as needed
DATA_CONFIG_FILE="configs/data_configs/2D/rot.yaml"
LPM_PINN_CONFIG_FILE="configs/model_configs/PINN/2D/LPM_pinn.yaml"

pipenv run python3 -m PINN.main "${LPM_PINN_CONFIG_FILE}" "${DATA_CONFIG_FILE}" \
    --output_path "${OUTPUT_PATH}" \
    --save \
    --make_internal_predictions \
    --make_external_predictions
```

Key flags:

| Flag | Description |
|---|---|
| `--save` | Save the trained model to `--output_path` |
| `--make_internal_predictions` | Evaluate on interpolation (internal) test domains after training |
| `--make_external_predictions` | Evaluate on extrapolation (external) test domains after training |

---

### Physics-Informed Deep Operator Networks (PI-DONs)

Two PI-DONs variants are available:

| Model | Config (2D) | Config (3D) |
|---|---|---|
| LPM-DON | `configs/model_configs/DeepONet/2D/LPM_deeponet.yaml` | `configs/model_configs/DeepONet/3D/LPM_deeponet.yaml` |
| LG-DON | `configs/model_configs/DeepONet/2D/LG_deeponet.yaml` | `configs/model_configs/DeepONet/3D/LG_deeponet.yaml` |

Append `_pca` to the config filename to use PCA-based geometric descriptors (e.g. `LPM_deeponet_pca.yaml`).

**Example** — train LPM-DON on the 2D rotational family with 14 sensor points:

```bash
LPM_DON_CONFIG_FILE="configs/model_configs/DeepONet/2D/LPM_deeponet.yaml"
DATA_CONFIG="configs/data_configs/2D/rot.yaml"
OUTPUT_PATH="outputs/DeepONets/2D/anisotropic/center/rot_14_sensors"  # adjust as needed
NUM_SENSORS=14
FAMILY_NAME="rot_family"
DIM=2

pipenv run python3 DeepONet/main.py \
    --model_config "$LPM_DON_CONFIG_FILE" \
    --data_config "$DATA_CONFIG" \
    --output_folder_path "$OUTPUT_PATH" \
    --num_sensors "$NUM_SENSORS" \
    --family_name "$FAMILY_NAME" \
    --dim "$DIM"
```

---
