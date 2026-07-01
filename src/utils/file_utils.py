import yaml
import h5py
import numpy as np
from pathlib import Path
import os
import torch

from PINN.model import PINN


def load_config(config_path: Path):
    """Loads a .yaml file."""
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def load_trained_model(model_folder_path, device):

    MODEL_PATH = os.path.join(model_folder_path, 'model.pth')
    CONFIG_PATH = os.path.join(model_folder_path, 'model_config.yaml')

    config = load_config(CONFIG_PATH)
    model = PINN(config=config["pinn"], conductivities=config["sim_coeff"], device=device)
    
    model.load_state_dict(torch.load(MODEL_PATH))

    return model


def save_dict_to_hdf5(h5file, path, dictionary):
    """Recursively saves a nested dictionary to an HDF5 file."""
    for key, value in dictionary.items():
        full_path = f"{path}/{key}" if path else key
        if isinstance(value, dict):
            # Create a group and recurse
            _ = h5file.create_group(full_path)
            save_dict_to_hdf5(h5file, full_path, value)
        else:
            # Handle strings and basic data
            if isinstance(value, str):
                h5file.create_dataset(full_path, data=np.bytes_(value))
            else:
                h5file.create_dataset(full_path, data=value)


def load_dict_from_hdf5(h5file, path="/", downsample_factor=None):
    """Recursively load a nested dictionary from an HDF5 file."""
    result = {}
    for key in h5file[path]:
        item = h5file[f"{path}/{key}"]
        if isinstance(item, h5py.Group):
            # Recurse into groups
            result[key] = load_dict_from_hdf5(h5file, f"{path}/{key}", downsample_factor=downsample_factor)
        elif isinstance(item, h5py.Dataset):
            data = item[()]  # Load the data
            # Only downsample if it's an array with shape (not a scalar)
            if downsample_factor is not None and hasattr(data, 'shape') and data.shape:
                data = data[::downsample_factor]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            result[key] = data
    return result



def load_data_family_from_hdf5(load_path):
    """
    Load a dictionary from an HDF5 file.
    """
    with h5py.File(load_path, "r") as f:
        def recursively_load_dict_from_group(h5file, path):
            ans = {}
            for key, item in h5file[path].items():
                key_str = str(key)
                if isinstance(item, h5py.Group):
                    ans[key_str] = recursively_load_dict_from_group(h5file, path + key_str + "/")
                else:
                    ans[key_str] = item[()]  # load as numpy array

            for key, val in h5file[path].attrs.items():
                key_str = str(key)
                ans[key_str] = val
            return ans
        
        return recursively_load_dict_from_group(f, "/")
