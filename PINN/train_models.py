import os
import torch
import numpy as np
from timeit import default_timer as timer
from datetime import datetime
import matplotlib.pyplot as plt

from PINN.model import PINN
from PINN.pinn_trainer import PINNTrainer
from src.data.domain_family import DomainFamily
from src.utils.save_utils import save_all


def train_pinn(model_config: dict,
               sim_config: dict,
               data_config: dict,
               family: DomainFamily,
               device: torch.device,
               save_results: bool,
               noise_level: float,
               output_path: str = "",
               checkpoint_folder_path: str = None,
               start_epoch: int = 0,
               map_pde: bool = False):
    
    include_scaling_layer = model_config.get("include_scaling_layer", False)
    val_eval_method = model_config.get("val_method", "max_mse_loss")
    print(f"Validation evaluation method: {val_eval_method}")

    if include_scaling_layer:
        x_offset, x_scale = family.get_x_scaling_params()
        t_offset, t_scale = family.get_t_scaling_params()
        geom_offset, geom_scale = family.get_geometric_descriptor_scaling_params()
        model_config["scaling_params"] = {
            "x_offset": x_offset,
            "x_scale": x_scale,
            "t_offset": t_offset,
            "t_scale": t_scale,
            "geom_offset": geom_offset,
            "geom_scale": geom_scale
        }

    if model_config['geometric_descriptor'] == "pca":
        # Visualize pca cumulative variance for training data
        fig = plt.figure()
        plt.bar(np.arange(1, len(family.pca_variance_percentage) + 1), family.pca_variance_percentage, alpha=0.5, align='center', label='Individual variance')
        plt.step(np.arange(1, len(np.cumsum(family.pca_variance_percentage)) + 1), np.cumsum(family.pca_variance_percentage), where='mid', color='red', label='Cumulative variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.show()

        os.makedirs(output_path, exist_ok=True)
        fig.savefig(os.path.abspath(os.path.join(output_path, "pca_variance_plot.png")))


    loss_fn = torch.nn.MSELoss()

    PINN_model = PINN(config=model_config, conductivities=sim_config, device=device).to(device)

    if checkpoint_folder_path is not None:
        start_checkpoint = os.path.join(checkpoint_folder_path, f'checkpoint_{str(start_epoch)}') + '.pth'
        print(f"Loading model from checkpoint: {start_checkpoint}")
        PINN_model.load_state_dict(torch.load(start_checkpoint)['model_state_dict'])
        print("Model loaded successfully.")
        print("Loading optimizer state from checkpoint.")
        optimizer = torch.optim.Adam(params=PINN_model.parameters(), lr=model_config["optimizer"]["learning_rate"])
        optimizer.load_state_dict(torch.load(start_checkpoint)['optimizer_state_dict'])
        print("Optimizer state loaded successfully.")
        print(f"Resuming training from loaded checkpoint (epoch {start_epoch}).")
    else:
        optimizer = torch.optim.Adam(params=PINN_model.parameters(), lr=model_config["optimizer"]["learning_rate"])

    trainer = PINNTrainer(
        optimizer=optimizer,
        epochs=model_config["training"]["epochs"],
        data_loss_fn=loss_fn,
        device=device,
        map_pde=map_pde
        )
    
    exp_name = model_config["model"]["name"]
    
    experiment_name = exp_name + '-' + datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
    slurm_id = os.getenv('SLURM_JOB_ID')
    experiment_name += f"_jobid_{slurm_id}" if slurm_id is not None else ''
    checkpoint_path_name = os.path.join("./outputs/checkpoints", experiment_name) if not checkpoint_folder_path else checkpoint_folder_path

    print("Training PINN model:", exp_name)
    # Start timer
    start_time = timer()

    # Train model
    PINN_model_train_results, PINN_model_val_results = trainer.train(model=PINN_model,
                                                                    model_config=model_config,
                                                                    domain_family=family,
                                                                    show_worst_val=False,
                                                                    mapped_mode=model_config['use_LPM'],
                                                                    checkpoint_path=checkpoint_path_name,
                                                                    start_epoch=start_epoch
                                                                    )


    # End timer and print training time
    end_time = timer()

    if val_eval_method == "max_l2_loss":
        PINN_model_val_results.max_l2_loss = np.nan_to_num(PINN_model_val_results.max_l2_loss, nan=np.inf)
        best_epoch = np.argmin(PINN_model_val_results.max_l2_loss)*model_config['training']['val_epochs'] + start_epoch
    else:
        PINN_model_val_results.max_total_loss = np.nan_to_num(PINN_model_val_results.max_total_loss, nan=np.inf)
        best_epoch = np.argmin(PINN_model_val_results.max_total_loss)*model_config['training']['val_epochs'] + start_epoch


    models = os.listdir(checkpoint_path_name)
    last_saved_model = max(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Last saved model: {last_saved_model}")
    print(f"Best saved model: {best_epoch}")
    
    best_model = PINN(config=model_config, conductivities=sim_config, device=device)
    best_model.load_state_dict(torch.load(os.path.join(checkpoint_path_name, f'checkpoint_{str(best_epoch)}') + '.pth')['model_state_dict'])
    best_model.training_time = end_time-start_time
    
    print(f"Total training time: {best_model.training_time:.3f} seconds")

    if save_results:
        print("Saving results")

        config = {"data": data_config, "pinn": model_config, "sim_coeff": sim_config}

        output_path = output_path if output_path else "./outputs"

        save_all(model=best_model,
                outputs_folder_path=output_path,
                config_settings=config,
                split_indices=family.split_indices,
                folder_name=experiment_name,
                training_results=PINN_model_train_results,
                val_results=PINN_model_val_results
                )
        
    return best_model, PINN_model_train_results, PINN_model_val_results