from pathlib import Path
import torch
import yaml
import pandas as pd

from PINN.model import PINN
from src.training.loggers import TrainLogger, ValLogger

def save_dict_data(results: dict, folder_path: Path, save_file_name: str):
    """Saves a training result dict as .csv in the given folder path."""

    RESULTS_SAVE_PATH = str(Path.joinpath(folder_path, save_file_name + '.csv'))

    # Filter out any keys with empty arrays
    results = {k: v for k, v in results.items() if len(v) > 0}
    headers = list(results.keys())

    df = pd.DataFrame.from_dict(results)
    print(f"Saving model {save_file_name} to: {RESULTS_SAVE_PATH}")
    df.to_csv(RESULTS_SAVE_PATH, header=headers)


def save_model_info(model_name: str,
                    training_time: float,
                    split_indices: dict,
                    folder_path: Path):
    """Saves information about the model in a .csv file."""

    model_info = {"model_name": [model_name],
                  "training_time": [training_time],
                  "train_idx": split_indices["train_idx"],
                  "val_idx": split_indices["val_idx"],
                  "test_idx": split_indices["test_idx"]}
    
    headers = list(model_info.keys())
    
    INFO_SAVE_PATH = str(Path.joinpath(folder_path, 'model_info.csv'))
    df = pd.DataFrame.from_dict(model_info, orient='index').transpose()
    print(f"Saving model info to: {INFO_SAVE_PATH}")
    df.to_csv(INFO_SAVE_PATH, header=headers)


def save_model(model: torch.nn.Module,
               folder_path: Path):
    """Saves model features as .pth in the given folder (folder_name)."""

    MODEL_SAVE_PATH = str(Path.joinpath(folder_path, 'model.pth'))

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)


def save_predictions(predictions: torch.Tensor,
                     folder_path: Path):
    """Saves predicted data as a .csv file"""

    predictions = predictions.squeeze().cpu()

    PREDICTIONS_SAVE_PATH = str(Path.joinpath(folder_path, 'predictions.csv'))
    
    df = pd.DataFrame(predictions)
    print(f"Saving model predictions to: {PREDICTIONS_SAVE_PATH}")
    df.to_csv(PREDICTIONS_SAVE_PATH)


def save_model_config(config: dict,
                      folder_path: Path):
    
    CONFIG_SAVE_PATH = str(Path.joinpath(folder_path, 'model_config.yaml'))
    
    with open(CONFIG_SAVE_PATH, "w") as file:
        yaml.dump(config, file)
    
    file.close()
    print(f"Saving model config to: {CONFIG_SAVE_PATH}")


def save_all(model: PINN,
             outputs_folder_path: str,
             folder_name: str,
             config_settings: dict,
             split_indices: dict,
             training_results: TrainLogger = None,
             val_results: ValLogger = None,
             prediction_results: torch.Tensor = None):
    """
    Saves:
        - model_info.csv
        - training_loss.csv
        - model.pth
    """
    # Create folder for storing results
    ROOT_FOLDER = Path(outputs_folder_path)
    ROOT_FOLDER.mkdir(parents=True,
                      exist_ok=True)
    
    FOLDER_NAME = Path.joinpath(ROOT_FOLDER, folder_name)
    FOLDER_NAME.mkdir(parents=True,
                      exist_ok=True)
    
    # Save all information and data related to the model
    save_model_info(model_name=model.name if hasattr(model, 'name') else '',
                    training_time=model.training_time,
                    split_indices=split_indices,
                    folder_path=FOLDER_NAME)
    

    save_model(model=model,
               folder_path=FOLDER_NAME)
    
    save_model_config(config=config_settings,
                      folder_path=FOLDER_NAME)
    
    if training_results is not None:
        training_results = vars(training_results)
        save_dict_data(results=training_results, folder_path=FOLDER_NAME, save_file_name="training_loss")
    
    if val_results is not None:
        val_results = vars(val_results)
        save_dict_data(results=val_results, folder_path=FOLDER_NAME, save_file_name="val_loss")

    
    if prediction_results is not None:
        save_predictions(predictions=prediction_results,
                         folder_path=FOLDER_NAME)
