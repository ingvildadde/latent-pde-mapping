import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from itertools import chain
import os
from time import time

from DeepONet.model import DeepONet

from src.training.loggers import TrainLogger, ValLogger
from src.data.domain_family import DomainFamily
from src.data.data_handlers import InputData

from src.training.loss_functions import data_loss, boundary_condition_loss, pde_loss, initial_data_loss
from src.inference.metrics import relative_L2

def pi_train_DON(model: DeepONet,
              model_config: dict,
              domain_family: DomainFamily,
              device: torch.device,
              checkpoint_path: str,
              start_epoch: int = 0,
              dim: int = 2):
    """
        trains a deep operator network

        Parameters:
            model    (DeepONet)     : the network to be trained
            x_branch (torch.tensor) : the branch input data
            x_trunk  (torch.tensor) : the trunk input data
            y        (torch.tensor) : the targets
    """

    train_logger = TrainLogger()
    val_logger = ValLogger()

    lr = model_config['optimizer']['learning_rate']
    epochs = model_config['training']['epochs']
    mapped_mode = model_config['use_LPM']
    val_epochs = model_config['training']['val_epochs']
    dim = model_config['dimensions']
    grad_clip_norm = model_config['training'].get('grad_clip_norm', None)
    map_pde = model_config.get('map_pde', False)

    end_epoch = start_epoch + epochs

    data_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model = model
    best_score = float('inf')

    model.to(device)
    
    for epoch in range(start_epoch, end_epoch+1):

        start_time_epoch = time()

        model.train()

        total_loss_value, data_loss_value, initial_loss_value, pde_loss_value, ode_loss_value, bc_loss_value, F_loss_value = 0, 0, 0, 0, 0, 0, 0

        for i, (domain, col_inputs, bc_inputs, init_inputs, init_targets) in enumerate(domain_family.iter_train()):

            sensor_functions = domain.V_sensor_points[:, 0]

            domain_splits = domain.mapped_train_data if mapped_mode else domain.train_data

            supervised_data = InputData(**domain_splits.sensor_inputs.__dict__, dim=dim, device=device, V=sensor_functions)
            data_batch_loss_unweighted = data_loss(model, supervised_data, data_loss_fn, domain.V_sensor_points)

            if mapped_mode:
                boundary_data = InputData(**bc_inputs.__dict__, dim=dim, device=device, V=sensor_functions)
                bc_batch_loss_unweighted = boundary_condition_loss(model, boundary_data, normals=boundary_data.normals, loss_fn=data_loss_fn, mapped_mode=map_pde, dim=dim)

                collocation_data = InputData(**col_inputs.__dict__, dim=dim, device=device, V=sensor_functions)
                pde_batch_loss_unweighted, ode_batch_loss_unweighted, _, _ = pde_loss(model, collocation_data, mapped_mode=map_pde, loss_fn=data_loss_fn, dim=dim, predict_F=False)

                initial_data = InputData(**init_inputs.__dict__, device=device, dim=dim, initial_data=True, V=sensor_functions)
                initial_batch_loss_unweighted = initial_data_loss(model, initial_data, data_loss_fn, init_targets.V)

                loss = data_batch_loss_unweighted + bc_batch_loss_unweighted + pde_batch_loss_unweighted + ode_batch_loss_unweighted + initial_batch_loss_unweighted

                total_loss_value += loss.item()
                data_loss_value += data_batch_loss_unweighted.item()
                bc_loss_value += bc_batch_loss_unweighted.item()
                pde_loss_value += pde_batch_loss_unweighted.item()
                ode_loss_value += ode_batch_loss_unweighted.item()
                initial_loss_value += initial_batch_loss_unweighted.item()
            else:
                loss = data_batch_loss_unweighted

                total_loss_value += loss.item()
                data_loss_value += data_batch_loss_unweighted.item()

            optimizer.zero_grad()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm) if grad_clip_norm else None # Apply gradient clipping if specified


            # Update parameters
            optimizer.step()

            end_time_epoch = time()
            epoch_duration = end_time_epoch - start_time_epoch

        if np.isnan(total_loss_value):
                print("NaN detected in data!")
                return train_logger, val_logger

        if epoch == 2000:
            lr = lr*0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print(f"\nNew learning rate: {lr} \n")

        log_train_metrics(
                train_logger,
                epoch,
                data_loss_value,
                initial_loss_value,
                pde_loss_value,
                ode_loss_value,
                bc_loss_value,
                total_loss_value,
                F_loss_value,
                epoch_duration,
                loss_weights=None,
                grad_norms=None
            )

        ### Test on validation data ###
        if ((val_epochs != 0) and (epoch % val_epochs == 0)) or (val_epochs % epochs == 0):
            
            model.eval()

            val_eval = {}
            val_preds = {}

            total_val_loss, val_data_loss, val_init_loss, val_pde_loss, val_ode_loss, val_bc_loss, val_F_loss = 0, 0, 0, 0, 0, 0, 0
            l2_loss, l2_values = 0, []

            for (i, val_domain) in enumerate(domain_family.val_domains):

                val_sensor_functions = val_domain.V_sensor_points[:, 0]
                
                x = val_domain.x_ref if mapped_mode else val_domain.x
                x_bc = val_domain.x_ref_bc if mapped_mode else val_domain.x_bc
                bc_normals = val_domain.x_ref_bc_normals if mapped_mode else val_domain.x_bc_normals

                skip = 100
                temp_normal = torch.zeros_like(x)
                val_input_data = InputData(x=x[::skip, :], t=val_domain.tau[::skip, :], D=val_domain.D[::skip, :, :], F=val_domain.F[::skip, :, :], u=val_domain.u[::skip, :, :], geometric_descriptor=val_domain.geometric_descriptor[::skip, :, :], normal=temp_normal[::skip, :, :], dim=dim, device=device, V=val_sensor_functions) # Downsampled input data
                data_loss_domain = data_loss(model, input_data=val_input_data, data_loss_fn=torch.nn.MSELoss(), true_y=val_domain.V[::skip, :])
                val_data_loss += data_loss_domain.item()

                if mapped_mode:
                    skip_bc = 20 if dim == 3 else 1
                    input_data_bc = InputData(x=x_bc[::skip_bc, :], t=val_domain.tau[:x_bc[::skip_bc, :].shape[0]], D=val_domain.D_bc[::skip_bc, :], F=val_domain.F_bc[::skip_bc, :], u=val_domain.u_bc[::skip_bc, :], geometric_descriptor=val_domain.geometric_descriptor[:x_bc[::skip_bc, :].shape[0]], normal=bc_normals[::skip_bc, :], dim=dim, device=device, V=val_sensor_functions)
                    val_bc_loss_domain = boundary_condition_loss(model, input_data=input_data_bc, normals=input_data_bc.normals, loss_fn=torch.nn.MSELoss(), mapped_mode=map_pde, dim=dim)
                    val_bc_loss += val_bc_loss_domain.item()

                    val_pde_loss_domain, val_ode_loss_domain, _, _ = pde_loss(model, input_data=val_input_data, mapped_mode=map_pde, loss_fn=torch.nn.MSELoss(), dim=dim)

                    val_pde_loss += val_pde_loss_domain.item()
                    val_ode_loss += val_ode_loss_domain.item()

                    val_input_data_init = InputData(x=x[::skip, :], t=val_domain.tau[::skip, :], D=val_domain.D[::skip, :], F=val_domain.F[::skip, :], u=val_domain.u[::skip, :], geometric_descriptor=val_domain.geometric_descriptor[::skip, :], normal=temp_normal[::skip, :], device=device, dim=dim, initial_data=True, V=val_sensor_functions)
                    initial_loss_domain = initial_data_loss(model, input_data=val_input_data_init, data_loss_fn=torch.nn.MSELoss(), true_y=val_domain.V_init[::skip])

                    val_init_loss += initial_loss_domain.item()

                    total_loss_domain = data_loss_domain.item() + val_bc_loss_domain.item() + val_pde_loss_domain.item() + val_ode_loss_domain.item() + initial_loss_domain.item()
                
                else:
                    total_loss_domain = data_loss_domain.item()

                total_val_loss += total_loss_domain
                val_eval[i] = [total_loss_domain]

                V_preds = model(val_input_data)[0].detach()
                l2_loss_domain = relative_L2(V_preds, torch.tensor(val_domain.V[::skip, :]).to(model.device))
                l2_loss += l2_loss_domain.item()
                l2_values.append(l2_loss_domain.item())
                
                if mapped_mode:
                    del val_pde_loss_domain, val_ode_loss_domain, val_input_data, val_bc_loss_domain, input_data_bc, data_loss_domain, initial_loss_domain
                else:
                    del val_input_data, data_loss_domain

            val_eval_values = list(chain(*val_eval.values()))
            

            mean_data_loss = val_data_loss / len(domain_family.val_domains)
            mean_bc_loss = val_bc_loss / len(domain_family.val_domains)
            mean_pde_loss = val_pde_loss / len(domain_family.val_domains)
            mean_ode_loss = val_ode_loss / len(domain_family.val_domains)
            mean_init_loss = val_init_loss / len(domain_family.val_domains)
            mean_total_loss = total_val_loss / len(domain_family.val_domains)
            max_total_loss = np.max(val_eval_values)

            mean_l2_loss = l2_loss / len(domain_family.val_domains)
            max_l2_loss = np.max(l2_values)

            log_val_metrics(
                val_logger,
                max_total_loss,
                mean_total_loss,
                mean_data_loss,
                mean_init_loss,
                mean_pde_loss,
                mean_ode_loss,
                mean_bc_loss,
                val_F_loss,
                max_l2_loss,
                mean_l2_loss
                )

            if max_total_loss < best_score:
                best_score = max_total_loss
                tqdm.write(f"\nNew best max loss={best_score:.5f}\n", file=sys.stderr)

            checkpoint(model, checkpoint_path, epoch, optimizer)

    return train_logger, val_logger, best_model


def checkpoint(model: DeepONet, folder_path: str, epoch: int, optimizer):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    checkpoint_path = os.path.join(folder_path, f'checkpoint_{epoch}.pth') 

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.abspath(checkpoint_path))
    
    tqdm.write(f"Checkpoint saved at epoch {epoch}")


def log_train_metrics(logger: TrainLogger, epoch: int, data_loss, initial_loss, pde_loss, ode_loss, bc_loss, total_loss, F_loss, epoch_duration, loss_weights=None, grad_norms=None):

        if F_loss == 0.0:
            log_msg = f"Epoch: {epoch} | Data train loss: {data_loss:.4e} | Initial data train loss: {initial_loss:.4e} | PDE train loss {pde_loss:.4e} | ODE train loss {ode_loss:.4e} | BC train loss {bc_loss:.4e} | Total train loss {total_loss:.4e} | Epoch duration: {epoch_duration:.4e} sec"
        else:
            log_msg = f"Epoch: {epoch} | Data train loss: {data_loss:.4e} | Initial data train loss: {initial_loss:.4e} | PDE train loss {pde_loss:.4e} | ODE train loss {ode_loss:.4e} | BC train loss {bc_loss:.4e} | F train loss {F_loss:.4e} | Total train loss {total_loss:.4e} | Epoch duration: {epoch_duration:.4e} sec"
    
        # Add weight information if using dynamic weights
        if loss_weights is not None and epoch % 50 == 0:  # Log weights every 50 epochs
            weight_str = " | Weights: " + ", ".join([f"{k}={v:.3f}" for k, v in loss_weights.items()])
            log_msg += weight_str
        if grad_norms is not None:  # Log gradient norms
            grad_norm_str = " | Grad Norms: " + ", ".join([f"{k}={v:.3f}" for k, v in grad_norms.items()])
            log_msg += grad_norm_str
        
        tqdm.write(log_msg, file=sys.stderr)

        logger.data_loss.append(data_loss)
        logger.initial_loss.append(initial_loss)
        logger.pde_loss.append(pde_loss)
        logger.ode_loss.append(ode_loss)
        logger.bc_loss.append(bc_loss)
        logger.total_loss.append(total_loss)
        if F_loss != 0.0:
            logger.F_loss.append(F_loss)
        
        # Log weights if dynamic
        if loss_weights is not None:
            logger.loss_weights.append(loss_weights.copy())
        if grad_norms is not None:
            logger.grad_norms.append(grad_norms.copy())
        logger.epoch_durations.append(epoch_duration)


def log_val_metrics(logger: ValLogger, max_total_loss, mean_total_loss, data_loss, init_loss, pde_loss, ode_loss, bc_loss, F_loss, max_l2_loss, mean_l2_loss):

        if F_loss == 0.0:
            tqdm.write(f"Validation | Max total loss: {max_total_loss:.4e} | Mean total loss: {mean_total_loss:.4e} | Mean data loss: {data_loss:.4e} | Mean init loss: {init_loss:.4e} | Mean PDE loss: {pde_loss:.4e} | Mean ODE loss: {ode_loss:.4e} | Mean BC loss: {bc_loss:.4e}", file=sys.stderr)
        else:
            tqdm.write(f"Validation | Max total loss: {max_total_loss:.4e} | Mean total loss: {mean_total_loss:.4e} | Mean data loss: {data_loss:.4e} | Mean init loss: {init_loss:.4e} | Mean PDE loss: {pde_loss:.4e} | Mean ODE loss: {ode_loss:.4e} | Mean BC loss: {bc_loss:.4e} | Mean F loss: {F_loss:.4e}", file=sys.stderr)

        tqdm.write(f"Validation | Max L2 loss: {max_l2_loss:.4e} | Mean L2 loss: {mean_l2_loss:.4e}", file=sys.stderr)

        logger.max_total_loss.append(max_total_loss)
        logger.mean_total_loss.append(mean_total_loss)
        logger.data_loss.append(data_loss)
        logger.init_loss.append(init_loss)
        logger.pde_loss.append(pde_loss)
        logger.ode_loss.append(ode_loss)
        logger.bc_loss.append(bc_loss)
        logger.max_l2_loss.append(max_l2_loss)
        logger.mean_l2_loss.append(mean_l2_loss)
        if F_loss != 0.0:
            logger.F_loss.append(F_loss)