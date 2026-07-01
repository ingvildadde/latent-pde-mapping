import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import os
import sys
from time import time

from src.inference.metrics import relative_L2
from PINN.model import PINN
from src.data.data_handlers import InputData
from src.training.loggers import TrainLogger, ValLogger
from src.data.domain_family import DomainFamily

from src.training.loss_functions import *
from src.training.loss_weight_scheduler import GradNormLossWeightScheduler, StaticLossWeightScheduler

class PINNTrainer():
    """Class used for PINN training."""
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 epochs: int,
                 data_loss_fn: torch.nn.Module,
                 device: torch.device = "cuda",
                 use_dynamic_weights: bool = False,
                 map_pde: bool = False
                 ):

        self.device = device
        self.optimizer = optimizer
        self.epochs = epochs
        self.data_loss_fn = data_loss_fn
        self.use_dynamic_weights = use_dynamic_weights
        self.loss_weight_scheduler = None
        self.use_lbfgs = False
        self.map_pde = map_pde

        print("PINN Trainer initialized with map_pde =", self.map_pde)


    def train(self,
              model: PINN,
              model_config: dict,
              domain_family: DomainFamily,
              mapped_mode: bool,
              checkpoint_path: str,             
              show_worst_val: bool = False,
              start_epoch: int = 0
              ):
        """Trains a model over a given set of epochs. In each epoch a training step and test step is performed."""

        mapped_mode = model_config["use_LPM"]

        # Create loggers to store results
        train_logger = TrainLogger()
        val_logger = ValLogger()

        val_epochs = model_config["training"]["val_epochs"]
        best_score = float('inf')
        
        # Send model to device
        model.to(self.device)

        if mapped_mode:
            print("Running mapped mode.")

        training_config = model_config["training"]
        loss_weights = training_config["loss_weights"]
        dim = model_config["dimensions"]
        lr = model_config["optimizer"]["learning_rate"]
        lr_scheduler = model_config["optimizer"]["lr_scheduler"]
        lr_scheduler_gamma = model_config["optimizer"].get("lr_scheduler_gamma")
        grad_clip_norm = training_config.get("grad_clip_norm", None)
        include_F_loss = model_config.get("include_F_loss", False)
        end_epoch = start_epoch + self.epochs

        tqdm.write("\n", file=sys.stderr)
        tqdm.write(f"Starting training for {model.name} in {dim}D for {self.epochs} epochs.", file=sys.stderr)
        tqdm.write(f"Loss weights: {loss_weights}", file=sys.stderr)
        tqdm.write(f"Using gradient clipping: {grad_clip_norm}", file=sys.stderr)
        tqdm.write("\n", file=sys.stderr)   

        # Initialize loss weight scheduler
        dynamic_weight_config = training_config.get("dynamic_loss_weights", {})
        use_dynamic_weights = dynamic_weight_config.get("enabled", False) or self.use_dynamic_weights

        if lr_scheduler == "exponential_decay":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_scheduler_gamma)
            tqdm.write(f"Using ExponentialLR with gamma: {lr_scheduler_gamma}", file=sys.stderr)
        elif lr_scheduler == "multistep":
            milestones = [1000, 50000, 100000]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
            tqdm.write(f"Using MultiStepLR with milestones at epochs: {milestones}", file=sys.stderr)
        else:
            scheduler = None
            tqdm.write("No learning rate scheduler used (lr dropped to 0.0001 at epoch 100).", file=sys.stderr)
        
        loss_names = ['data', 'init', 'pde', 'ode', 'bc']
        if include_F_loss:
            loss_names.append('F')
        
        if use_dynamic_weights:
            print("Using dynamic gradient norm-based loss weight adaptation")
            self.loss_weight_scheduler = GradNormLossWeightScheduler(
                loss_names=loss_names,
                initial_weights=loss_weights,
                alpha=dynamic_weight_config.get("alpha", 1.5),
                update_frequency=dynamic_weight_config.get("update_frequency", 10),
                adaptation_method=dynamic_weight_config.get("method", "grad_norm"),
                warmup_steps=dynamic_weight_config.get("warmup_steps", 100),
                device=self.device
            )
        else:
            self.loss_weight_scheduler = StaticLossWeightScheduler(weights=loss_weights)

        # Initialize grad norm dict
        grad_norms = {}
        
        for loss_name in loss_names:
            grad_norms[loss_name] = 0.0

        # Get train data
        train_data = domain_family.mapped_train_dataloader if mapped_mode else domain_family.train_dataloader

        for epoch in (range(start_epoch, end_epoch+1)):

            start_time_epoch = time()
            
            ### Train loop
            model.train()

            # Compute losses and optimize
            total_loss_value, data_loss_value, initial_loss_value, pde_loss_value, ode_loss_value, bc_loss_value, F_loss_value, grad_norms, unweighted_losses = self.compute_losses(model, train_data, dim, loss_weights, use_dynamic_weights, grad_norms, grad_clip_norm, include_F_loss)
    
            if use_dynamic_weights:
                mean_grad_norms = {k: v / len(train_data) for k, v in grad_norms.items()}
                loss_weights = self.loss_weight_scheduler.update_weights(mean_grad_norms, unweighted_losses, epoch)
                current_weights = self.loss_weight_scheduler.get_weights()

                for loss_name in loss_names:
                    grad_norms[loss_name] = 0.0

            # Learning rate scheduler step
            if scheduler is not None:
                scheduler.step()
                new_lr = scheduler.get_last_lr()[0]
                if lr != new_lr:
                    tqdm.write(f"\nEpoch {epoch}: Learning rate adjusted to {scheduler.get_last_lr()[0]:.6f}\n", file=sys.stderr)
                    lr = new_lr
            else:
                if epoch == 100:
                    lr = lr*0.1
                    self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    print(f"\nNew learning rate: {lr} \n")
            
            
            end_time_epoch = time()
            epoch_duration = end_time_epoch - start_time_epoch
            
            self.log_train_metrics(
                train_logger, epoch, 
                data_loss_value.item(), 
                initial_loss_value.item(), 
                pde_loss_value.item(), 
                ode_loss_value.item(), 
                bc_loss_value.item(), 
                total_loss_value.item(), 
                F_loss_value,
                epoch_duration,
                loss_weights=current_weights if use_dynamic_weights else None,
                grad_norms=mean_grad_norms if use_dynamic_weights else None
            )
            

            ### Test on validation data ###
            if ((val_epochs != 0) and (epoch % val_epochs == 0)) or (val_epochs % self.epochs == 0):
                
                model.eval()

                val_eval = {}
                val_preds = {}

                total_val_loss, val_data_loss, val_init_loss, val_pde_loss, val_ode_loss, val_bc_loss, val_F_loss = 0, 0, 0, 0, 0, 0, 0
                l2_loss, l2_values = 0.0, []

                for (i, val_domain) in enumerate(domain_family.val_domains):

                    use_mapped_pde = self.map_pde #True #False
                    
                    x = val_domain.x_ref if mapped_mode else val_domain.x
                    x_bc = val_domain.x_ref_bc if mapped_mode else val_domain.x_bc
                    bc_normals = val_domain.x_ref_bc_normals if mapped_mode else val_domain.x_bc_normals

                    skip = 100
                    temp_normal = torch.zeros_like(x)
                    val_input_data = InputData(x=x[::skip, :], t=val_domain.tau[::skip, :], D=val_domain.D[::skip, :, :], F=val_domain.F[::skip, :, :], u=val_domain.u[::skip, :, :], geometric_descriptor=val_domain.geometric_descriptor[::skip, :, :], normal=temp_normal[::skip, :, :], dim=dim, device=self.device) # Downsampled input data
                    data_loss_domain = data_loss(model, input_data=val_input_data, data_loss_fn=torch.nn.MSELoss(), true_y=val_domain.V[::skip, :])
                    
                    val_data_loss += data_loss_domain.item()

                    val_input_data_init = InputData(x=x[::skip, :], t=val_domain.tau[::skip, :], D=val_domain.D[::skip, :], F=val_domain.F[::skip, :], u=val_domain.u[::skip, :], geometric_descriptor=val_domain.geometric_descriptor[::skip, :], normal=temp_normal[::skip, :], device=self.device, dim=dim, initial_data=True)
                    initial_loss_domain = initial_data_loss(model, input_data=val_input_data_init, data_loss_fn=torch.nn.MSELoss(), true_y=val_domain.V_init[::skip])

                    val_init_loss += initial_loss_domain.item()

                    val_pde_loss_domain, val_ode_loss_domain, _, _ = pde_loss(model, input_data=val_input_data, mapped_mode=use_mapped_pde, loss_fn=torch.nn.MSELoss(), dim=dim)

                    val_pde_loss += val_pde_loss_domain.item()
                    val_ode_loss += val_ode_loss_domain.item()

                    skip_bc = 20 if dim == 3 else 1
                    input_data_bc = InputData(x=x_bc[::skip_bc, :], t=val_domain.tau[:x_bc[::skip_bc, :].shape[0]], D=val_domain.D_bc[::skip_bc, :], F=val_domain.F_bc[::skip_bc, :], u=val_domain.u_bc[::skip_bc, :], geometric_descriptor=val_domain.geometric_descriptor[:x_bc[::skip_bc, :].shape[0]], normal=bc_normals[::skip_bc, :], dim=dim, device=self.device)
                    val_bc_loss_domain = boundary_condition_loss(model,input_data=input_data_bc, normals=input_data_bc.normals, loss_fn=torch.nn.MSELoss(), mapped_mode=use_mapped_pde, dim=dim)

                    val_bc_loss += val_bc_loss_domain.item()

                    if include_F_loss:
                        F_loss_domain = u_loss(model, val_input_data, data_loss_fn=torch.nn.MSELoss(), true_u=val_input_data.u).item()
                        val_F_loss += F_loss_domain
                    else:
                        F_loss_domain = 0.0

                    total_loss_domain = (data_loss_domain.item() + initial_loss_domain.item() + val_pde_loss_domain.item() + val_ode_loss_domain.item() + val_bc_loss_domain.item() + F_loss_domain)
                    total_val_loss += total_loss_domain
                    val_eval[i] = [total_loss_domain]

                    V_preds = model(val_input_data)[0].detach()
                    l2_loss_domain = relative_L2(V_preds, torch.tensor(val_domain.V[::skip, :]).to(model.device))
                    l2_loss += l2_loss_domain.item()
                    l2_values.append(l2_loss_domain.item())

                    if show_worst_val:
                        with torch.inference_mode():
                            validation_preds = model.predict(val_input_data)
                        
                        val_preds[i] = validation_preds

                    del val_pde_loss_domain, val_ode_loss_domain, val_input_data, val_bc_loss_domain, input_data_bc, data_loss_domain, initial_loss_domain, F_loss_domain

                val_eval_values = list(chain(*val_eval.values()))

                mean_pde_loss = val_pde_loss / len(domain_family.val_domains)
                mean_ode_loss = val_ode_loss / len(domain_family.val_domains)
                mean_bc_loss = val_bc_loss / len(domain_family.val_domains)
                mean_data_loss = val_data_loss / len(domain_family.val_domains)
                mean_init_loss = val_init_loss / len(domain_family.val_domains)
                mean_F_loss = val_F_loss / len(domain_family.val_domains)
                mean_total_loss = total_val_loss / len(domain_family.val_domains)
                max_total_loss = np.max(val_eval_values)

                mean_l2_loss = l2_loss / len(domain_family.val_domains)
                max_l2_loss = np.max(l2_values)


                self.log_val_metrics(val_logger, max_total_loss, mean_total_loss, mean_data_loss, mean_init_loss, mean_pde_loss, mean_ode_loss, mean_bc_loss, mean_F_loss, max_l2_loss=max_l2_loss, mean_l2_loss=mean_l2_loss)

                if torch.isnan(total_loss_value):
                    print("NaN detected in data!")
                    return train_logger, val_logger
                
                if max_total_loss < best_score:
                    best_score = max_total_loss
                    tqdm.write(f"\nNew best max loss={best_score:.5f}\n", file=sys.stderr)

                self.checkpoint(model, checkpoint_path, epoch)
                

                if show_worst_val:
                    print("Prediction on validation data with highest total loss")
                    max_key = max(val_eval, key=val_eval.get)
                    max_idx = np.argmax(val_eval[max_key])
                    x = domain_family.val_domains[max_key].x_ref if mapped_mode else domain_family.val_domains[max_key].x
                    
                    plt.figure()
                    plt.scatter(domain_family.tau[0, :].detach().numpy(), domain_family.val_domains[max_key].V[max_idx, :].detach().numpy(), s=10, label='Ground truth')
                    plt.plot(domain_family.tau[0, :].detach().numpy(), val_preds[max_key][max_idx, :].cpu().detach().numpy(), c="darkorange", label='Prediction')
                    plt.legend()
                    plt.title(f"Max loss: Domain {max_key}, (x, y) = ({x[max_idx, 0, 0]:.2f}, {x[max_idx, 0, 1]:.2f})")
                    plt.xlabel(r"$\tau$ [T.U.]")
                    plt.ylabel("V [A.U.]")
                    plt.show()

        return train_logger, val_logger
    

    def log_train_metrics(self, logger: TrainLogger, epoch: int, data_loss, initial_loss, pde_loss, ode_loss, bc_loss, total_loss, F_loss, epoch_duration, loss_weights=None, grad_norms=None):

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

    def log_val_metrics(self, logger: ValLogger, max_total_loss, mean_total_loss, data_loss, init_loss, pde_loss, ode_loss, bc_loss, F_loss, max_l2_loss, mean_l2_loss):

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


    def checkpoint(self, model: PINN, folder_path: str, epoch: int):

        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        checkpoint_path = os.path.join(folder_path, f'checkpoint_{epoch}.pth') 

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler is not None else None
            }, os.path.abspath(checkpoint_path))
        

    
    def compute_losses(self, model: PINN, train_data, dim, loss_weights, use_dynamic_weights, grad_norms, grad_clip_norm, include_F_loss):

        total_loss_value, data_loss_value, initial_loss_value, pde_loss_value, ode_loss_value, bc_loss_value, F_loss_value = 0, 0, 0, 0, 0, 0, 0

        for _, (s_batch, i_batch, c_batch, b_batch) in enumerate(train_data):

            use_mapped_pde = self.map_pde

            # Compute unweighted losses
            boundary_data = InputData(**b_batch['input'], dim=dim, device=self.device)
            bc_batch_loss_unweighted = boundary_condition_loss(model, boundary_data, normals=boundary_data.normals, loss_fn=self.data_loss_fn, mapped_mode=use_mapped_pde, dim=dim)

            supervised_data = InputData(**s_batch['input'], dim=dim, device=self.device)
            data_batch_loss_unweighted = data_loss(model, supervised_data, self.data_loss_fn, s_batch['target'])

            initial_data = InputData(**i_batch['input'], device=self.device, dim=dim, initial_data=True)
            initial_batch_loss_unweighted = initial_data_loss(model, initial_data, self.data_loss_fn, i_batch['target'])
            
            collocation_data = InputData(**c_batch['input'], dim=dim, device=self.device)
            pde_batch_loss_unweighted, ode_batch_loss_unweighted, _, _ = pde_loss(model, collocation_data, mapped_mode=use_mapped_pde, loss_fn=self.data_loss_fn, dim=dim, predict_F=include_F_loss)

            if include_F_loss:
                F_batch_loss_unweighted = u_loss(model, supervised_data, data_loss_fn=self.data_loss_fn, true_u=supervised_data.u)
            else:
                F_batch_loss_unweighted = torch.tensor(0.0, device=self.device)

            # Prepare unweighted losses dict for weight scheduler
            unweighted_losses = {
                'data': data_batch_loss_unweighted,
                'init': initial_batch_loss_unweighted,
                'pde': pde_batch_loss_unweighted,
                'ode': ode_batch_loss_unweighted,
                'bc': bc_batch_loss_unweighted,
            }
            if include_F_loss:
                unweighted_losses['F'] = F_batch_loss_unweighted
            
            # Apply weights
            bc_batch_loss = loss_weights["bc"] * bc_batch_loss_unweighted
            data_batch_loss = loss_weights["data"] * data_batch_loss_unweighted
            initial_batch_loss = loss_weights["init"] * initial_batch_loss_unweighted
            pde_batch_loss = loss_weights["pde"] * pde_batch_loss_unweighted
            ode_batch_loss = loss_weights["ode"] * ode_batch_loss_unweighted
            
            if include_F_loss:
                F_batch_loss = loss_weights.get("F", 1.0) * F_batch_loss_unweighted
                F_batch_loss_item = F_batch_loss.item()
            else:
                F_batch_loss = torch.tensor(0.0, device=self.device)
                F_batch_loss_item = 0.0

            loss = data_batch_loss + initial_batch_loss + pde_batch_loss + ode_batch_loss + bc_batch_loss + F_batch_loss

            total_loss_value += loss
            data_loss_value += data_batch_loss
            initial_loss_value += initial_batch_loss
            pde_loss_value += pde_batch_loss
            ode_loss_value += ode_batch_loss
            bc_loss_value += bc_batch_loss
            F_loss_value += F_batch_loss_item

            if use_dynamic_weights:
                grad_norms = self.loss_weight_scheduler.compute_gradient_norms(model, unweighted_losses, grad_norm_dict=grad_norms, grad_clip_norm=grad_clip_norm)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm) if grad_clip_norm else None # Apply gradient clipping if specified
            self.optimizer.step()

        return total_loss_value, data_loss_value, initial_loss_value, pde_loss_value, ode_loss_value, bc_loss_value, F_loss_value, grad_norms, unweighted_losses