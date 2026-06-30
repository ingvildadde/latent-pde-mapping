import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import random
import argparse
from datetime import datetime
from time import time
from timeit import default_timer as timer
import matplotlib.font_manager as fm
import matplotlib as mpl

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference.metrics import relative_L2
from src.utils.file_utils import load_config
from src.data.domain_family import DomainFamily
from src.data.data_handlers import InputData
from src.utils.save_utils import save_all
from src.inference.predictions import save_predictions_to_hdf5

from DeepONet.model import DeepONet
from DeepONet.pi_trainer import pi_train_DON


try:
    font_path = os.path.abspath('./fonts/Times_New_Roman.ttf')
    times_new_roman = fm.FontProperties(fname=font_path)

    # Register it with matplotlib
    fm.fontManager.addfont(font_path)

    # Now matplotlib knows about the name
    mpl.rcParams['font.family'] = times_new_roman.get_name()

    print("Using:", mpl.rcParams['font.family'])
except:
    print("Using standard font")


# Set matplotlib font size
font_size = 20
ticks_label_size = 18
plt.rcParams.update({'font.size': font_size, 'xtick.labelsize': ticks_label_size, 'ytick.labelsize': ticks_label_size})

def main(
        model_config: str,
        sim_config: str,
        data_config: str,
        output_folder_path: str,
        num_sensors: int,
        family_name: str = '',
        dim: int = 2,
        checkpoint_path: str = None,
        recreate_sensor_points: bool = False,
        visualize_sensor_points_flag: bool = False
    ):

    data_config = load_config(os.path.abspath(data_config))
    model_config = load_config(os.path.abspath(model_config))
    sim_config = load_config(os.path.abspath(sim_config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    torch.cuda.empty_cache()

    # Set random seed for reproducibility
    RANDOM_SEED = model_config["random_seed"]

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    print(f"\nTraining DeepONet with {num_sensors} sensor points and {dim}D data")
    print(f"Recreating sensor points: {recreate_sensor_points}")

    family = DomainFamily(name=family_name, model_config=model_config, data_config=data_config, dim=dim, use_sensor_data=True, recreate_sensor_points=recreate_sensor_points, num_sensors=num_sensors) 

    exp_name = model_config["model"]["name"]
    
    experiment_name = exp_name + '-' + datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
    slurm_id = os.getenv('SLURM_JOB_ID')
    experiment_name += f"_jobid_{slurm_id}" if slurm_id is not None else ''
    checkpoint_path_name = os.path.join("./outputs/checkpoints", experiment_name) if not checkpoint_path else checkpoint_path

    if visualize_sensor_points_flag:
        print("Visualizing sensor points...")
        visualize_sensor_points(family.train_domains, dim=dim, save_folder_path=os.path.join(output_folder_path, "figures/train_sensor_points"))
    
    model = DeepONet(model_config, sim_config, device=device)

    print("Test domains:", len(family.test_domains))
    print("Validation domains:", len(family.val_domains))
    print("Training domains:", len(family.train_domains))
    
    start_time = timer()
    train_logs, val_logs, best_model = pi_train_DON(model, model_config, family, device, start_epoch=0, checkpoint_path=checkpoint_path_name, dim=dim)
    end_time = timer()

    val_logs.max_total_loss = np.nan_to_num(val_logs.max_total_loss, nan=np.inf)
    best_epoch = np.argmin(val_logs.max_total_loss)*model_config['training']['val_epochs']
    models = os.listdir(checkpoint_path_name)
    last_saved_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Last saved model: {last_saved_model}")
    print(f"Best saved model: {best_epoch}")

    best_model = DeepONet(config=model_config, conductivities=sim_config, device=device)
    best_model.load_state_dict(torch.load(os.path.join(checkpoint_path_name, f'checkpoint_{str(best_epoch)}') + '.pth')['model_state_dict'])
    best_model.training_time = end_time-start_time

    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    config = {"data": data_config, "pinn": model_config, "sim_coeff": sim_config}

    # Save training logs using save_dict_data utility
    save_all(best_model,
             output_folder_path,
             config_settings=config,
             split_indices=family.split_indices,
             folder_name=experiment_name,
             training_results=train_logs,
             val_results=val_logs)
    
    # Internal test predictions
    make_predictions(family, model, output_folder_path, num_sensors=num_sensors, family_name=family_name, internal=True, name=model_config["model"]["name"], dim=dim)
    del family

    # External test predictions
    print("\nLoading external data for predictions...")
    external_family = DomainFamily(name=family_name, model_config=model_config, data_config=data_config, dim=dim, use_external_family=True, use_sensor_data=True, num_sensors=num_sensors)
    make_predictions(external_family, model, output_folder_path, num_sensors=num_sensors, family_name=family_name, internal=False, name=model_config["model"]["name"], dim=dim)
    



def make_predictions(family: DomainFamily, model: DeepONet, output_folder_path: str, num_sensors: int, family_name: str = '', internal: bool = True, name: str = '', dim: int = 2):

    pred_name = "internal" if internal else "external"

    test_domains = family.test_domains if internal else family.domains

    predictions_dict = {}

    test_losses = {}
    test_mse_losses = {}
    test_outputs = {}

    with torch.no_grad():
        for i, test_domain in enumerate((test_domains)):
            
            print(f"Making predictions for test domain {i+1}/{len(test_domains)}...")

            begin_time = time()

            sensor_functions = test_domain.V_sensor_points[:, 0]
            test_input = InputData(x=test_domain.x_ref, t=test_domain.tau, geometric_descriptor=test_domain.geometric_descriptor, normal=test_domain.x_ref_bc_normals, dim=dim, device=model.device, V=sensor_functions)
            V_preds = model.forward(test_input)[0]
            test_loss = relative_L2(V_preds, test_domain.V.to(model.device))
            test_mse_loss = nn.MSELoss()(V_preds, test_domain.V.to(model.device))
            test_losses[i] = test_loss.cpu().item()
            test_mse_losses[i] = test_mse_loss.cpu().item()

            # Move output to CPU for saving
            test_outputs[i] = {}
            test_outputs[i]['preds'] = V_preds.cpu()
            test_outputs[i]['gt'] = test_domain.V.cpu()
            end_time = time()
            test_outputs[i]['inference_time'] = end_time - begin_time
    
    predictions_dict['family'] = family_name
    predictions_dict['output'] = test_outputs
    predictions_dict['name'] = name
    predictions_dict['test_losses'] = test_losses

    print(f"\n{pred_name.capitalize()} L2 test losses: \n")
    print(test_losses)
    print(f"\n{pred_name.capitalize()} MSE test losses: \n")
    print(test_mse_losses)

    print("Saving predictions...")
    save_predictions_to_hdf5(f"{name}_{pred_name}_test_predictions.h5", output_folder_path, predictions_dict)
    print(f"Saved predictions to {name}_{pred_name}_test_predictions.h5")



def visualize_sensor_points(domains, dim, save_folder_path: str):

    # Create folder if it doesn't exist
    os.makedirs(save_folder_path, exist_ok=True)

    time_idx = 0
    
    for idx, domain in enumerate(domains):
        # ax = plt.figure().add_subplot(111)
        V = (domain.V[:, time_idx]*100) - 80
        print(domain.V_sensor_points.shape)
        if dim == 3:
            from mpl_toolkits.mplot3d import Axes3D 
            ax = plt.figure().add_subplot(111, projection='3d')
            _ = ax.scatter(domain.x_ref[:, time_idx, 0].detach().cpu().numpy(), domain.x_ref[:, time_idx, 1].detach().cpu().numpy(), domain.x_ref[:, time_idx, 2].detach().cpu().numpy(), c=V.detach().cpu().numpy(), alpha=0.1, vmin=-80, vmax=20)
            sc2 = ax.scatter(domain.sensor_points[:, time_idx, 0].detach().cpu().numpy(), domain.sensor_points[:, time_idx, 1].detach().cpu().numpy(), domain.sensor_points[:, time_idx, 2].detach().cpu().numpy(), c=(domain.V_sensor_points[:, time_idx].detach().cpu().numpy()*100)-80, marker='s', vmin=-80, vmax=20)
        else:
            ax = plt.figure().add_subplot(111)
            _ = plt.scatter(domain.x_ref[:, time_idx, 0].detach().cpu().numpy(), domain.x_ref[:, time_idx, 1].detach().cpu().numpy(), c=V.detach().cpu().numpy(), alpha=0.1, vmin=-80, vmax=20)
            # sc2 = plt.scatter(domain.sensor_points[:, time_idx, 0].detach().cpu().numpy(), domain.sensor_points[:, time_idx, 1].detach().cpu().numpy(), c=(domain.V_sensor_points[:, time_idx].detach().cpu().numpy()*100)-80, marker='s', vmin=-80, vmax=20)
            sc2 = plt.scatter(domain.sensor_points[:, time_idx, 0].detach().cpu().numpy(), domain.sensor_points[:, time_idx, 1].detach().cpu().numpy(), marker='s', c="red")
        
        cbar = plt.colorbar(sc2, ax=ax)
        cbar.set_label('V [mV]')
        plt.savefig(os.path.join(save_folder_path, f"sensor_points_t{time_idx}_{idx}.pdf"), bbox_inches='tight')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    
    # implement args parsing
    parser = argparse.ArgumentParser(description="Train DeepONet model for system dynamics.")
    parser.add_argument("--model_config", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("--sim_config", type=str, required=True, help="Path to the simulation configuration file.")
    parser.add_argument("--data_config", type=str, required=True, help="Path to the data configuration file.")
    parser.add_argument("--output_folder_path", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint folder.")
    parser.add_argument("--num_sensors", type=int, required=True, help="Number of sensor points to use.")
    parser.add_argument("--family_name", type=str, default="", help="Name of the domain family.")
    parser.add_argument("--dim", type=int, default=2, help="Dimensionality of the problem.")
    parser.add_argument("--recreate_sensor_points", action="store_true", help="Whether to recreate sensor points.")
    parser.add_argument("--visualize_sensor_points", action="store_true", help="Whether to visualize sensor points.")
    args = parser.parse_args()

    main(
        model_config=args.model_config,
        sim_config=args.sim_config,
        data_config=args.data_config,
        output_folder_path=args.output_folder_path,
        num_sensors=args.num_sensors,
        checkpoint_path=args.checkpoint_path,
        family_name=args.family_name,
        dim=args.dim,
        recreate_sensor_points=args.recreate_sensor_points,
        visualize_sensor_points_flag=args.visualize_sensor_points
    )
