import torch
import torch.nn as nn
import numpy as np

from DeepONet.MLP import MLP
from src.enums.activation_functions import ActivationFunctions
from src.data.data_handlers import InputData

class DeepONet(nn.Module):
    """
        Implementation of the Deep Operator Network
    """
    def __init__(self,
                 config: dict,
                 conductivities: dict, 
                 device: torch.device
                 ):
        """
            Creates the DON using the following parameters

            Parameters:
            n_branch (int) : the input size of the branch network
            n_trunk  (int) : the input size of the trunk network
            depth    (int) : number of layers in each network 
            width.   (int) : number of nodes at each layer
            p        (int) : output dimension of network
            act            : the activation function to be used
        """
        super(DeepONet, self).__init__()

        # Read config parameters
        n_branch = config["model"]["branch_input_size"]
        n_trunk = config["model"]["trunk_input_size"]
        depth = config["model"]["hidden_layers"]
        width = config["model"]["hidden_units"]
        p = config["model"]["branches_out_dim"]
        act = config["model"]["activation_function"]

        self.device = device
        self.training_time = None
        self.include_geometric_descriptor = config.get("include_geometric_descriptor", False)
        self.dim = config["dimensions"]
        self.output_size = config["model"]["output_size"]

        self.p = p
        self.t_norm = 12.9

        if act == ActivationFunctions.TANH.value:
            act= nn.Tanh()
        elif act == ActivationFunctions.RELU.value:
            act = nn.ReLU()
        else:
            raise Exception(f"Could not recognize '{act}' as activation function.")

        self.conductivities = conductivities['conductivities']
        self.diffusion_tensor = self.__construct_D() if config['use_global_D'] else None    # Uses global diffusion tensor given by conductivities in system_dynamics.yaml if local conductivities are not provided.

        # Create branch network
        self.branch_net = MLP(input_size=n_branch, hidden_size=width, num_classes=p, depth=depth, act=act)
        self.branch_net.float()

        # Create trunk network
        self.trunk_net = MLP(input_size=n_trunk, hidden_size=width, num_classes=p*self.output_size, depth=depth, act=act)
        self.trunk_net.float()
        
        self.bias = nn.Parameter(torch.ones((1,)),requires_grad=True)
        
        # Move the entire model to the specified device
        self.to(device)
    
    def convert_np_to_tensor(self, array):
        if isinstance(array, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            tensor = torch.from_numpy(array)
            return tensor.to(self.device, torch.float32)
        else:
            return array.to(self.device, torch.float32)

    
    def forward(self, input_data: InputData):
        """
            evaluates the operator

            x_branch : input_function
            x_trunk : point evaluating at

            returns a scalar
        """

        observed_functions = input_data.V.unsqueeze(0)
        B, _ = observed_functions.shape
        trunk_spatial_loc, trunk_times  = input_data.x.shape

        trunk_inputs = [input_data.x, input_data.y, input_data.t] if self.dim == 2 else [input_data.x, input_data.y, input_data.z, input_data.t]
        branch_inputs = [input_data.V]

        if self.include_geometric_descriptor:
            for desc in input_data.geometric_descriptor:
                branch_inputs.append(desc[0:1, 0])

        x_trunk = torch.stack(trunk_inputs, dim=-1).view(B, trunk_inputs[0].shape[0]*trunk_inputs[0].shape[1], len(trunk_inputs))
        x_branch = torch.cat(branch_inputs, dim=-1).view(B, -1)
        

        branch_out = self.branch_net.forward(x_branch)
        trunk_out = self.trunk_net.forward(x_trunk)

        trunk_out = trunk_out.view(trunk_out.shape[0], trunk_out.shape[1], self.output_size, self.p)
        branch_out = branch_out.unsqueeze(1)

        output = torch.einsum('bip,bnmp->bnm', branch_out, trunk_out) + self.bias
        output = output.reshape(B, trunk_spatial_loc, trunk_times, self.output_size).squeeze(0)

        V = torch.sigmoid(output[:, :, 0])
        W = torch.nn.functional.softplus(output[:, :, 1])

        return V, W

    
    def __construct_D(self):
        print("Constructing global diffusion tensor D...")
        # Set conductivities
        sigma_il = self.conductivities["sigma_il"]
        sigma_el = self.conductivities["sigma_el"]
        sigma_it = self.conductivities["sigma_it"]
        sigma_et = self.conductivities["sigma_et"]

        if self.dim == 3:
            sigma_in = self.conductivities["sigma_in"]
            sigma_en = self.conductivities["sigma_en"]
            D = torch.diag(torch.tensor([(sigma_il*sigma_el)/(sigma_il + sigma_el),
                                         (sigma_it*sigma_et)/(sigma_it + sigma_et),
                                         (sigma_in*sigma_en)/(sigma_in + sigma_en)]))
        else:
            D = torch.diag(torch.tensor([(sigma_il*sigma_el)/(sigma_il + sigma_el),
                                         (sigma_it*sigma_et)/(sigma_it + sigma_et)]))
        
        D = D*self.t_norm
        D = (D.to(self.device)).requires_grad_(True)
        return D