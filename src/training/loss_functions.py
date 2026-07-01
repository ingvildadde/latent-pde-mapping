import torch

from src.data.data_handlers import InputData
from PINN.model import PINN
from DeepONet.model import DeepONet

def data_loss(model: PINN | DeepONet, input_data: InputData, data_loss_fn, true_y):
    """
    Computes data loss with the given data_loss_fn.
    """
    y_pred = model.forward(input_data=input_data)
    vm = y_pred[0].to(model.device)
    data_loss = data_loss_fn(vm, true_y.to(model.device))
    return data_loss

def u_loss(model: PINN, input_data: InputData, data_loss_fn, true_u):
    """
    Computes data loss with the given data_loss_fn.
    """
    raise NotImplementedError("u_loss is not implemented for current models.")
    y_pred = model.forward(input_data=input_data)
    u_pred = y_pred[:, :, 2:].to(model.device)
    u_loss = data_loss_fn(u_pred, true_u.to(model.device))
    return u_loss


def initial_data_loss(model: PINN, input_data: InputData, data_loss_fn, true_y):
    """
    Computes initial data loss, i.e. data loss with the given data_loss_fn at t = 0.
    """
    y_pred = model.forward(input_data=input_data)
    vm = y_pred[0].to(model.device)
    data_loss = data_loss_fn(vm, true_y.to(model.device))
    return data_loss


def pde_loss(model: PINN, input_data: InputData, mapped_mode: bool, dim: int, loss_fn, predict_F: bool = False):
    """
    Computes residuals of the Aliev-Panfilov model with predicted values of V and W from the PINN.
    """
    # PDE specific parameters
    k = 8.0
    a = 0.15
    epsilon_0 = 0.002
    mu_1 = 0.2
    mu_2 = 0.3

    outputs = model.forward(input_data=input_data)
    V, W = outputs[0], outputs[1]

    if predict_F:
        u_pred = outputs[:, :, 2:]
        diffusion_term, J = compute_diffusion_term(model, input_data, mapped_mode, dimension=dim, predicted_F=u_pred)
    else:
        diffusion_term, J = compute_diffusion_term(model, input_data, mapped_mode, dimension=dim)

    if dim == 2:
        dV_dt, dW_dt, final_diffusion_term = compute_pde_loss_2d(input_data, V, W, diffusion_term, J)
    else:
        dV_dt, dW_dt, final_diffusion_term = compute_pde_loss_3d(input_data, V, W, diffusion_term, J)
    
    # Compute residuals
    I_ion = -((k*V*(V-a)*(V-1)+V*W))
    rhs_PDE = final_diffusion_term + I_ion
    rhs_ODE = -(epsilon_0 + (mu_1*W)/(V+mu_2))*(W+k*V*(V-a-1))
        
    PDE_residual = dV_dt - rhs_PDE
    ODE_residual = dW_dt - rhs_ODE

    # Compute mean-squared loss of residuals
    pde_loss = loss_fn(PDE_residual, torch.zeros_like(PDE_residual))
    ode_loss = loss_fn(ODE_residual, torch.zeros_like(ODE_residual))

    return pde_loss, ode_loss, PDE_residual, ODE_residual


def compute_pde_loss_2d(input_data: InputData, V, W, diffusion_term, J):

    # Compute gradients
    dV_dt = torch.autograd.grad(V, input_data.t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dW_dt = torch.autograd.grad(W, input_data.t, grad_outputs=torch.ones_like(W), create_graph=True)[0]
    dV = torch.autograd.grad(V, (input_data.x, input_data.y), grad_outputs=torch.ones_like(V), create_graph=True)
    dV_dx, dV_dy = dV[0], dV[1]

    # Compute diffusion term
    first_row_diffusion_term = diffusion_term[:, :, 0, 0]*dV_dx + diffusion_term[:, :, 0, 1]*dV_dy
    second_row_diffusion_term = diffusion_term[:, :, 1, 0]*dV_dx + diffusion_term[:, :, 1, 1]*dV_dy
    first_row_diffusion_term_dx = torch.autograd.grad(first_row_diffusion_term, input_data.x, grad_outputs=torch.ones_like(first_row_diffusion_term), create_graph=True)[0]
    second_row_diffusion_term_dy = torch.autograd.grad(second_row_diffusion_term, input_data.y, grad_outputs=torch.ones_like(second_row_diffusion_term), create_graph=True)[0]

    final_diffusion_term = (1/J)*(first_row_diffusion_term_dx + second_row_diffusion_term_dy)

    return dV_dt, dW_dt, final_diffusion_term


def compute_pde_loss_3d(input_data: InputData, V, W, diffusion_term, J):

    # Compute gradients
    dV_dt = torch.autograd.grad(V, input_data.t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dW_dt = torch.autograd.grad(W, input_data.t, grad_outputs=torch.ones_like(W), create_graph=True)[0]
    dV = torch.autograd.grad(V, (input_data.x, input_data.y, input_data.z), grad_outputs=torch.ones_like(V), create_graph=True)
    dV_dx, dV_dy, dV_dz = dV[0], dV[1], dV[2]

    # Compute diffusion term
    first_row_diffusion_term = diffusion_term[:, :, 0, 0]*dV_dx + diffusion_term[:, :, 0, 1]*dV_dy + diffusion_term[:, :, 0, 2]*dV_dz
    second_row_diffusion_term = diffusion_term[:, :, 1, 0]*dV_dx + diffusion_term[:, :, 1, 1]*dV_dy + diffusion_term[:, :, 1, 2]*dV_dz
    third_row_diffusion_term = diffusion_term[:, :, 2, 0]*dV_dx + diffusion_term[:, :, 2, 1]*dV_dy + diffusion_term[:, :, 2, 2]*dV_dz
    
    first_row_diffusion_term_dx = torch.autograd.grad(first_row_diffusion_term, input_data.x, grad_outputs=torch.ones_like(first_row_diffusion_term), create_graph=True)[0]
    second_row_diffusion_term_dy = torch.autograd.grad(second_row_diffusion_term, input_data.y, grad_outputs=torch.ones_like(second_row_diffusion_term), create_graph=True)[0]
    third_row_diffusion_term_dz = torch.autograd.grad(third_row_diffusion_term, input_data.z, grad_outputs=torch.ones_like(third_row_diffusion_term), create_graph=True)[0]

    final_diffusion_term = (1/J)*(first_row_diffusion_term_dx + second_row_diffusion_term_dy + third_row_diffusion_term_dz)

    return dV_dt, dW_dt, final_diffusion_term


def boundary_condition_loss(model: PINN, input_data: InputData, normals: torch.Tensor, mapped_mode: bool, dim: int, loss_fn):
    """
    Computes residuals of the no-flux Neumann boundary condition with predicted values of V from the PINN.
    """
    outputs = model.forward(input_data=input_data)
    V_boundary = outputs[0]

    if dim == 2:
        BC_residual = compute_boundary_condition_loss_2d(model, input_data, normals, mapped_mode, V_boundary)
    else:
        BC_residual = compute_boundary_condition_loss_3d(model, input_data, normals, mapped_mode, V_boundary)

    BC_loss = loss_fn(BC_residual, torch.zeros_like(BC_residual))
        
    return BC_loss

def compute_boundary_condition_loss_2d(model: PINN, input_data: InputData, normals: torch.Tensor, mapped_mode: bool, V_boundary):
    
    dV_dx = torch.autograd.grad(V_boundary, input_data.x, grad_outputs=torch.ones_like(V_boundary), create_graph=True)[0]
    dV_dy = torch.autograd.grad(V_boundary, input_data.y, grad_outputs=torch.ones_like(V_boundary), create_graph=True)[0]

    diffusion_term, _ = compute_diffusion_term(model, input_data, mapped_mode, dimension=2)
    diffusion_term = diffusion_term.reshape(dV_dx.shape[0], dV_dx.shape[1], 2, 2)

    first_row_diffusion_term = (diffusion_term[:, :, 0, 0]*dV_dx + diffusion_term[:, :, 0, 1]*dV_dy)
    second_row_diffusion_term = (diffusion_term[:, :, 1, 0]*dV_dx + diffusion_term[:, :, 1, 1]*dV_dy)

    BC_residual_x = normals[:, :, 0]*first_row_diffusion_term
    BC_residual_y = normals[:, :, 1]*second_row_diffusion_term

    BC_residual = BC_residual_x + BC_residual_y

    return BC_residual


def compute_boundary_condition_loss_3d(model: PINN, input_data: InputData, normals: torch.Tensor, mapped_mode: bool, V_boundary):

    dV_dx = torch.autograd.grad(V_boundary, input_data.x, grad_outputs=torch.ones_like(V_boundary), create_graph=True)[0]
    dV_dy = torch.autograd.grad(V_boundary, input_data.y, grad_outputs=torch.ones_like(V_boundary), create_graph=True)[0]
    dV_dz = torch.autograd.grad(V_boundary, input_data.z, grad_outputs=torch.ones_like(V_boundary), create_graph=True)[0]

    diffusion_term, _ = compute_diffusion_term(model, input_data, mapped_mode, dimension=3)
    diffusion_term = diffusion_term.reshape(dV_dx.shape[0], dV_dx.shape[1], 3, 3)

    first_row_diffusion_term = (diffusion_term[:, :, 0, 0]*dV_dx + diffusion_term[:, :, 0, 1]*dV_dy + diffusion_term[:, :, 0, 2]*dV_dz)
    second_row_diffusion_term = (diffusion_term[:, :, 1, 0]*dV_dx + diffusion_term[:, :, 1, 1]*dV_dy + diffusion_term[:, :, 1, 2]*dV_dz)
    third_row_diffusion_term = (diffusion_term[:, :, 2, 0]*dV_dx + diffusion_term[:, :, 2, 1]*dV_dy + diffusion_term[:, :, 2, 2]*dV_dz)

    BC_residual_x = normals[:, :, 0]*first_row_diffusion_term
    BC_residual_y = normals[:, :, 1]*second_row_diffusion_term
    BC_residual_z = normals[:, :, 2]*third_row_diffusion_term

    BC_residual = BC_residual_x + BC_residual_y + BC_residual_z

    return BC_residual



def compute_diffusion_term(model: PINN, input_data: InputData, mapped_mode: bool, dimension: int, predicted_F = None):
    """
    Computes the diffusion term based on wether or not the mapped pde form is used.
    """
    J = 1.0
    if mapped_mode:
        if predicted_F is not None:
            u_1_x = torch.autograd.grad(predicted_F[:, :, 0], input_data.x, grad_outputs=torch.ones_like(predicted_F[:, :, 0]), create_graph=True)[0]
            u_2_x = torch.autograd.grad(predicted_F[:, :, 1], input_data.x, grad_outputs=torch.ones_like(predicted_F[:, :, 1]), create_graph=True)[0]
            u_1_y = torch.autograd.grad(predicted_F[:, :, 0], input_data.y, grad_outputs=torch.ones_like(predicted_F[:, :, 0]), create_graph=True)[0]
            u_2_y = torch.autograd.grad(predicted_F[:, :, 1], input_data.y, grad_outputs=torch.ones_like(predicted_F[:, :, 1]), create_graph=True)[0]

            F = torch.zeros((predicted_F.shape[0], predicted_F.shape[1], 2, 2), dtype=predicted_F.dtype, device=predicted_F.device)
            F[:, :, 0, 0] = u_1_x.sum()
            F[:, :, 0, 1] = u_2_x.sum()
            F[:, :, 1, 0] = u_1_y.sum()
            F[:, :, 1, 1] = u_2_y.sum()

            F = F + torch.eye(2, dtype=predicted_F.dtype, device=predicted_F.device)
        else:
            F = input_data.F[:, :, :dimension*dimension].reshape(input_data.F.shape[0], input_data.F.shape[1], dimension, dimension)
        F_inverse = torch.inverse(F)
        F_inverse_transpose = torch.transpose(F_inverse, 2, 3)
        J = torch.det(F)
        J_ext = J.unsqueeze(dim=2).unsqueeze(dim=3).expand(F_inverse.shape[0], F_inverse.shape[1], 1, 1).to(model.device)
        if model.diffusion_tensor is None:
            diffusion_term = J_ext*torch.matmul(torch.matmul(F_inverse, input_data.D), F_inverse_transpose).float()
        else:
            diffusion_term = J_ext*torch.matmul(torch.matmul(F_inverse, model.diffusion_tensor), F_inverse_transpose)
            diffusion_term = J_ext*model.diffusion_tensor.unsqueeze(dim=0).unsqueeze(dim=1).expand(input_data.x.shape[0], input_data.t.shape[1], dimension, dimension)
    else:
        if model.diffusion_tensor is None:
            diffusion_term = input_data.D
        else:
            diffusion_term = model.diffusion_tensor.unsqueeze(dim=0).unsqueeze(dim=1).expand(input_data.x.shape[0], input_data.t.shape[1], dimension, dimension)
        J = torch.Tensor([J]).unsqueeze(dim=0).expand(input_data.x.shape[0], input_data.t.shape[1]).to(model.device)
    return diffusion_term, J