import os
import torch
import numpy as np
import argparse

from PINN.model import PINN
from src.data.domain_family import DomainFamily, Domain
from src.utils.file_utils import load_config
from src.data.data_handlers import InputData
from src.training.loss_functions import *


def compute_boundary_term_for_all_checkpoints(args):

    folder_path = args.folder_path
    use_training_data = args.train_data

    ds = 10e-6

    extr_file = True if len(args.extr_file) > 0 else False

    if use_training_data:
        output_file_name = "_training_boundary_terms.npy"
    elif extr_file:
        output_file_name = "_external_test_boundary_terms.npy"
    else:
        output_file_name = "_internal_test_boundary_terms.npy"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for folder in os.listdir(folder_path):

        if not folder.startswith("Affine") or not os.path.isdir(os.path.join(folder_path, folder)):
            continue

        checkpoints_path = os.path.join(os.path.abspath("outputs/checkpoints"), folder)

        model_config_path = os.path.join(folder_path, folder, "model_config.yaml")

        config = load_config(model_config_path)
        model_config = config['pinn']
        data_config = config['data']

        dim = model_config['dimensions']

        cond_config_path = os.path.abspath("configs/system_dynamics.yaml")
        system_config = load_config(cond_config_path)

        if extr_file:
            data_config['family_file'] = args.extr_file

        family = DomainFamily("", model_config, data_config, dim=dim)

        
        if extr_file:
            domains = family.domains
        else:
            domains = family.train_domains if use_training_data else family.test_domains

        all_terms = {}

        for i, model in enumerate(os.listdir(checkpoints_path)):

            term = {}
            
            if not model.endswith('.pth'):
                continue
            
            model_path = os.path.join(checkpoints_path, model)

            model = PINN(model_config, system_config, device=device)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])
            model.eval()

            model.to(device)

            Iv = compute_volume_term(model, domains, ds=ds, dim=dim, device=device)
            Ib = compute_missing_boundary_term(model, domains, ds=ds, dim=dim, device=device)

            term["Ib"] = Ib
            term["Iv"] = Iv

            all_terms[i] = term

        output_dir = os.path.join(folder_path, "boundary_terms")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.save(os.path.join(output_dir, model_config["model"]["name"] + output_file_name), all_terms, allow_pickle=True)

    return all_terms


def compute_missing_boundary_term(model: PINN, domains: list[Domain], ds: float, dim: int, device):

    downsample_factor = 10 if dim == 2 else 3

    boundary_terms = {}

    for i, domain in enumerate(domains):

        b_term = {}

        # Make small pertubation to boundary points
        perturbed_bc, bc = get_perturbed_points(domains[i].x_ref_bc[::downsample_factor], domain.geometric_descriptor, ds=ds, dim=dim)
        num_points = domains[i].x_ref_bc[::downsample_factor].shape[0]

        bc = (torch.tensor(bc)).unsqueeze(1).expand(bc.shape[0], domain.tau[::downsample_factor].shape[-1], bc.shape[1])


        # Get boundary points
        bc_points = InputData(x=bc,
                        t=domain.tau[:num_points],
                        D=domain.D_bc[::downsample_factor],
                        F=domain.F_bc[::downsample_factor],
                        u=domain.u_bc[::downsample_factor],
                        geometric_descriptor=domain.geometric_descriptor[:num_points],
                        normal=domain.x_bc_normals[::downsample_factor],
                        dim=dim,
                        device=device)

        # Compute R
        _, _, R_pde, R_ode = pde_loss(model=model, input_data=bc_points, mapped_mode=False, loss_fn=torch.nn.MSELoss(), dim=dim)


        R_2 = (torch.sum(R_pde**2, dim=1) + torch.sum(R_ode**2, dim=1)).detach().cpu().numpy()

        for s in perturbed_bc.keys():

            b_s_term = {}

            bc_forward, bc_backward = perturbed_bc[s]
            B_i, Vn_i, boundary_change_i = boundary_term(R_2, domain.x_bc_normals[::downsample_factor], ds, bc_forward, bc_backward, device=device)

            b_s_term["B"] = (B_i) / (num_points)
            b_s_term["f"] = R_2
            b_s_term["Vn"] = Vn_i
            b_s_term["boundary_change"] = boundary_change_i

            b_term[f"B_{s}"] = b_s_term

        boundary_terms[f"domain_{i}"] = b_term

    return boundary_terms



def boundary_term(R_2, boundary_normals,
                  ds, boundary_points_forward, boundary_points_backward, device):

    boundary_points_forward = torch.tensor(boundary_points_forward, device=device)
    boundary_points_backward = torch.tensor(boundary_points_backward, device=device)
    boundary_normals = boundary_normals[:, 0, :].to(device)

    # Normal velocities computed with central forward differences in s
    Vb = (boundary_points_forward - boundary_points_backward) / (2*ds)
    Vn = torch.sum(Vb * boundary_normals, dim=1).detach().cpu().numpy()

    # Boundary integral
    B = np.sum(R_2 * Vn)
    boundary_change = np.sum(Vn).item()

    return B.item(), Vn, boundary_change


def compute_volume_term(model: PINN, domains: list[Domain], ds: float, dim: int, device):

    downsample_factor = 10 if dim == 2 else 10

    volume_terms = {}

    for i, domain in enumerate(domains):

        v_term = {}

        perturbed_points, _ = get_perturbed_points(domains[i].x_ref[::downsample_factor], domain.geometric_descriptor, ds=ds, dim=dim)
        num_points = domains[i].x_ref[::downsample_factor].shape[0]

        for s in perturbed_points.keys():

            points_forward, points_backward = perturbed_points[s]

            # Expand to all time points
            points_forward = (torch.tensor(points_forward)).unsqueeze(1).expand(points_forward.shape[0], domain.tau[::downsample_factor].shape[-1], points_forward.shape[1])
            points_backward = (torch.tensor(points_backward)).unsqueeze(1).expand(points_backward.shape[0], domain.tau[::downsample_factor].shape[-1], points_backward.shape[1])

            input_forward = InputData(x=points_forward,
                            t=domain.tau[::downsample_factor],
                            D=domain.D[::downsample_factor],
                            F=domain.F[::downsample_factor],
                            u=domain.u[::downsample_factor],
                            geometric_descriptor=domain.geometric_descriptor[::downsample_factor],
                            normal=domain.x_bc_normals[::downsample_factor],
                            dim=dim,
                            device=device)

            input_backward = InputData(x=points_backward,
                            t=domain.tau[::downsample_factor],
                            D=domain.D[::downsample_factor],
                            F=domain.F[::downsample_factor],
                            u=domain.u[::downsample_factor],
                            geometric_descriptor=domain.geometric_descriptor[::downsample_factor],
                            normal=domain.x_bc_normals[::downsample_factor],
                            dim=dim,
                            device=device)

            _, _, R_pde_forward, R_ode_forward = pde_loss(model=model, input_data=input_forward, mapped_mode=False, loss_fn=torch.nn.MSELoss(), dim=dim)
            _, _, R_pde_backward, R_ode_backward = pde_loss(model=model, input_data=input_backward, mapped_mode=False, loss_fn=torch.nn.MSELoss(), dim=dim)
            
            R_2_forward = (torch.sum(R_pde_forward**2, dim=1) + torch.sum(R_ode_forward**2, dim=1)).detach().cpu().numpy()
            R_2_backward = (torch.sum(R_pde_backward**2, dim=1) + torch.sum(R_ode_backward**2, dim=1)).detach().cpu().numpy()
        
            f_i = (R_2_forward - R_2_backward) / (2*ds)

            Iv_i = np.sum(f_i) / num_points
            v_term[f"Iv_{s}"] = Iv_i.item()

        volume_terms[f"domain_{i}"] = v_term

    return volume_terms



def mapping(A_matrix, M_matrix, vector, non_linear=False):
    if non_linear:
        return A_matrix @ vector + (vector.T @ M_matrix @ vector)
    return A_matrix @ vector

def get_perturbed_points(x: torch.Tensor, affine_params: torch.Tensor, ds: float, dim: int):

    slice_factor = dim*dim

    A = affine_params[0, 0, :slice_factor]

    M = affine_params[0, 0, slice_factor:]

    x_perturbed = {}
    
    # filter out zero entries in A and M
    s_vec_a = A[(A != 0) & (A != 1)].numpy()
    s_vec_m = M[(M != 0) & (M != 1)].numpy()

    if s_vec_a.shape[0] > 0:
        for i in range(0, s_vec_a.shape[0]):
            s_i_forward = s_vec_a[i] + ds
            s_i_backward = s_vec_a[i] - ds

            A_forward = A.clone()
            A_forward[A == s_vec_a[i]] = s_i_forward

            A_backward = A.clone()
            A_backward[A == s_vec_a[i]] = s_i_backward

            x_i_forward = np.array([mapping(A_forward.reshape(dim, dim), M, v) for v in x[:, 0, :]])
            x_i_backward = np.array([mapping(A_backward.reshape(dim, dim), M, v) for v in x[:, 0, :]])

            x_perturbed[f"a_{i}"] = (x_i_forward, x_i_backward)
    
    if s_vec_m.shape[0] > 0:
        for j in range(0, s_vec_m.shape[0]):

            s_j_forward = s_vec_m[j] + ds
            s_j_backward = s_vec_m[j] - ds

            M_forward = M.clone()
            M_forward[M == s_vec_m[j]] = s_j_forward

            M_backward = M.clone()
            M_backward[M == s_vec_m[j]] = s_j_backward

            x_j_forward = np.array([mapping(A.reshape(dim, dim), torch.diag(M_forward), v, non_linear=True) for v in x[:, 0, :]])
            x_j_backward = np.array([mapping(A.reshape(dim, dim), torch.diag(M_backward), v, non_linear=True) for v in x[:, 0, :]])

            x_perturbed[f"m_{j}"] = (x_j_forward, x_j_backward)

    A = A.reshape(dim, dim)
    M = torch.diag(M)
    x_s = np.array([mapping(A, M, v) for v in x[:, 0, :]])

    return x_perturbed, x_s



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_path", type=str, default="", help="Path to folder with models", required=True)
    parser.add_argument("--train_data", action="store_true", help="Use training data domains. If not provided, test data domains are used.")

    parser.add_argument("--extr_file", type=str, default="", help="File name of extrapolation file", required=False)

    args = parser.parse_args()

    b_terms = compute_boundary_term_for_all_checkpoints(args)
