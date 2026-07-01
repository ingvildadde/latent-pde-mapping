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

Shell scripts for all experiments are provided in the `scripts/` folder:

- **PINNs** — `scripts/PINN/`
- **DeepONets** — `scripts/DeepONet/`

Each script submits a Slurm job and trains multiple model variants by default. Modify the script to target a specific model, and update the Slurm directives to match your cluster configuration.

**Example** — train PINNs on the 2D rotational family using deformation parameters as geometric descriptor:

```bash
sbatch scripts/PINN/2D_experiments/rot_experiment.sh
```

**Example** — train a physics-informed DeepONet on the same family:

```bash
sbatch scripts/DeepONet/2D_experiments/run_deeponet_rot.sh
```
