import random
import torch
import h5py
import os
import numpy as np
from pathlib import Path
from time import time

from PINN.model import PINN
from src.data.data_handlers import InputData
from src.data.domain import Domain
from src.data.domain_family import DomainFamily, CombinedFamily
from src.utils.file_utils import save_dict_to_hdf5, load_config, load_dict_from_hdf5, load_trained_model


def random_predictions(domain: Domain,
                       random_seed: int,
                       num_random_samples: int,
                       model: PINN,
                       device: torch.device,
                       mapped_mode: bool,
                       dim: int):
    
    random.seed(random_seed)

    spatial_data = domain.x_ref if mapped_mode else domain.x
    time = domain.tau
    targets = domain.V

    test_targets = []

    random_test_idx = np.random.choice(spatial_data.shape[0], num_random_samples, replace=False)

    test_samples = spatial_data[random_test_idx].to(device)
    test_targets = targets[random_test_idx].to(device)

    model.to(device)

    test_data = InputData(test_samples, time, D=domain.D, F=domain.F, u=domain.u, device=device, geometric_descriptor=domain.geometric_descriptor, dim=dim)
    V_preds = model.predict(test_data)

    return V_preds, test_targets, test_samples, time



def multiple_predictions(model: PINN, domains: list[Domain], mapped_mode: bool, device: torch.device, dim: int):

    all_predictions = {}

    for i, domain in enumerate(domains):

        all_predictions[i] = {}
        begin_time = time()
        all_predictions[i]["preds"] = predict(model=model, domain=domain, mapped_mode=mapped_mode, device=device, dim=dim)
        end_time = time()
        all_predictions[i]["inference_time"] = end_time - begin_time
        all_predictions[i]["gt"] = domain.V

    return all_predictions



def predict(model: PINN, domain: Domain, mapped_mode: bool, device: torch.device, dim: int):

    x = domain.x_ref if mapped_mode else domain.x
    normals = domain.x_ref_bc_normals if mapped_mode else domain.x_bc_normals

    input_data = InputData(x=x, t=domain.tau, D=domain.D, F=domain.F, u=domain.u, normal=normals, device=device, geometric_descriptor=domain.geometric_descriptor, dim=dim)

    model.to(device)
    predictions = model.predict(input_data=input_data)

    return predictions.cpu()


def predict_with_single_model(experiment_path: Path, family: DomainFamily | CombinedFamily, use_all_domains: bool, mapped_mode: bool, device: torch.device):
    
    config = load_config(os.path.join(experiment_path, "model_config.yaml"))
    data_config = config["data"]
    data_config["root_path"] = '../.' + data_config["root_path"]
        
    domains = family.domains if use_all_domains else family.test_domains

    model = load_trained_model(experiment_path, device=device)
    return multiple_predictions(model, domains, mapped_mode=mapped_mode, device=device)



def save_predictions_to_hdf5(hdf5_filename: str, output_folder_path: Path, predictions: dict):
    """
    Saves predictions to a HDF5 file in an output_folder_path
    """
    file_path = os.path.join(output_folder_path, hdf5_filename)

    with h5py.File(file_path, "w") as f:
        save_dict_to_hdf5(f, "", predictions)


def load_predictions_from_hdf5(hdf5_filename: str, folder_path: Path, downsample_factor: int = None):
    """
    Saves predictions to a HDF5 file in an output_folder_path   
    """
    file_path = os.path.join(folder_path, hdf5_filename)

    with h5py.File(file_path, "r") as f:
        loaded_dict = load_dict_from_hdf5(f, downsample_factor=downsample_factor)

    return loaded_dict



def make_prediction(trained_model: PINN,
                    family: DomainFamily|CombinedFamily,
                    extrapolation: bool,
                    model_config: dict,
                    device: torch.device,
                    dim: int,
                    output_path: str):

    domains = family.domains if extrapolation else family.test_domains
    pred_name = "internal" if not extrapolation else "external"

    name = model_config["model"]["name"]

    predictions_dict = {}
    print("\nMaking predictions...")
    predictions_dict['output'] = multiple_predictions(trained_model, domains, mapped_mode=model_config['use_LPM'], device=device, dim=dim)
    predictions_dict["family"] = family.name if hasattr(family, "name") else "Combined"
    predictions_dict["name"] = name

    print("Saving predictions...")
    save_predictions_to_hdf5(f"{name}_{pred_name}_test_predictions_with_time.h5", output_path, predictions_dict)
    print(f"Saved predictions to {name}_{pred_name}_test_predictions_with_time.h5")