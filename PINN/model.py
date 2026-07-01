import torch.nn as nn
import torch

from src.enums.activation_functions import ActivationFunctions
from src.data.data_handlers import InputData

class PINN(nn.Module):

    def __init__(self,
                 config: dict,
                 conductivities: dict,
                 device: torch.device
                 ):
        super().__init__()

        self.device = device

        # Read config parameters
        self.dim = config["dimensions"]
        self.include_geometric_descriptor = config.get("include_geometric_descriptor", None)
        self.input_shape = config["model"]["input_size"]
        hidden_units = config["model"]["hidden_units"]
        hidden_layers = config["model"]["hidden_layers"]
        output_shape = config["model"]["output_size"]
        include_scaling_layer = config.get("include_scaling_layer", False)

        activation_function = config["model"]["activation_function"]

        if activation_function == ActivationFunctions.TANH.value:
            activation_function = nn.Tanh()
        elif activation_function == ActivationFunctions.SIN.value:
            activation_function = SinActivation()
        else:
            raise Exception(f"Could not recognize '{activation_function}' as activation function.")

        if include_scaling_layer:
            scaling_config = config.get("scaling_params", {})
            self.scaling_layer = ScalingLayer(
                x_offset=scaling_config.get("x_offset"),
                x_scale=scaling_config.get("x_scale"),
                t_offset=scaling_config.get("t_offset"),
                t_scale=scaling_config.get("t_scale"),
                geom_offset=scaling_config.get("geom_offset"),
                geom_scale=scaling_config.get("geom_scale"),
            )



        self.input_layer = [nn.Linear(in_features=self.input_shape, out_features=hidden_units),
                            activation_function]
        

        self.hidden_layers = [layer for _ in range(hidden_layers)
                              for layer in (nn.Linear(
                                  in_features=hidden_units,
                                  out_features=hidden_units
                                  ),
                              activation_function)]

        self.output_layer = [nn.Linear(in_features=hidden_units, out_features=output_shape)]

        self.layer_stack = self.input_layer + self.hidden_layers + self.output_layer
        self.layer_stack = nn.Sequential(*self.layer_stack)

        for i, layer in enumerate(self.layer_stack):
            if isinstance(layer, nn.Linear):
                if activation_function == SinActivation():
                    n = layer.in_features
                    if i == 0:
                        layer.weight.data.uniform_(-1/n, 1/n)
                    else:
                        w_0 = 30
                        std = torch.sqrt(6 / n) / w_0
                        layer.weight.data.uniform_(-std, std)
                else:
                    nn.init.xavier_uniform_(layer.weight)
            #     # nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize bias to zero for stability

        self.t_norm = 12.9

        # Set model information
        self.name = config["model"]["name"] if len(config["model"]["name"]) > 0 else self.__class__.__name__
        self.training_time = None

        # Construct diffusion tensor
        self.conductivities = conductivities["conductivities"]
        self.diffusion_tensor = self.__construct_D() if config['use_global_D'] else None


    def forward(self, input_data: InputData):
        inputs = [input_data.x, input_data.y, input_data.t] if self.dim == 2 else [input_data.x, input_data.y, input_data.z, input_data.t]

        if self.include_geometric_descriptor:
            inputs.extend(input_data.geometric_descriptor)
        
        if hasattr(self, 'scaling_layer'):
            inputs = self.scaling_layer(inputs)

        X = torch.stack(inputs, dim=len(input_data.x.shape))
        outputs = self.layer_stack(X)

        V = torch.sigmoid(outputs[:, :, 0])
        W = torch.nn.functional.softplus(outputs[:, :, 1])

        return V, W
    

    def predict(self, input_data: InputData):
        self.eval()
        with torch.inference_mode():
            y_pred = self.forward(input_data=input_data)
        # return y_pred[:, :, 0]
        return y_pred[0]


    def __construct_D(self):
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




class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
    


class ScalingLayer(nn.Module):
    def __init__(self, x_offset, x_scale, t_offset, t_scale, geom_offset, geom_scale):
        super().__init__()
        self.register_buffer('x_offset', x_offset)
        self.register_buffer('x_scale', x_scale)
        self.register_buffer('t_offset', t_offset)
        self.register_buffer('t_scale', t_scale)

        if geom_offset is not None and geom_scale is not None:
            self.register_buffer('geom_offset', geom_offset)
            self.register_buffer('geom_scale', geom_scale)
        else:
            self.geom_offset = None
            self.geom_scale = None


    def forward(self, inputs: list[InputData]):

        x_scaled = (inputs[0] - self.x_offset[0]) / self.x_scale[0]
        y_scaled = (inputs[1] - self.x_offset[1]) / self.x_scale[1]
        t_scaled = (inputs[2] - self.t_offset) / self.t_scale
        if self.geom_scale is not None:
            geom_scaled = []
            for i in range(self.geom_scale.shape[0]):
                if self.geom_scale[i] != 0:
                    geom_scaled.append((inputs[i+3]- self.geom_offset[i]) / self.geom_scale[i])
                else:
                    geom_scaled.append(torch.zeros_like(inputs[i+3]))
            return [x_scaled, y_scaled, t_scaled] + geom_scaled

        return [x_scaled, y_scaled, t_scaled]