import torch
import math
import numpy as np
from torch.utils.data import DataLoader

from src.data.custom_dataset import CustomDataset
from src.data.helper_functions import select_random_indices_from_data


class Input():
    def __init__(self, x, t, F = None, D = None, u = None, normal = None, geometric_descriptor = None):
        self.x = x
        self.t = t
        self.F = torch.zeros_like(x) if F is None else F
        self.D = torch.zeros_like(x) if D is None else D
        self.u = torch.zeros_like(x) if u is None else u
        self.normal = torch.zeros_like(x) if normal is None else normal
        self.geometric_descriptor = torch.zeros_like(x) if geometric_descriptor is None else geometric_descriptor



class Target():
    def __init__(self, V):
        self.V = V


class InputData():
    def __init__(self, x: torch.Tensor, t: torch.Tensor, geometric_descriptor: torch.Tensor, normal: torch.Tensor, device: torch.device, dim: int,  D: torch.Tensor = None, F: torch.Tensor = None, u: torch.Tensor = None, initial_data: bool = False, V: torch.Tensor = None):

        self.x = x[:, :, 0].to(device, dtype=torch.float32).requires_grad_(True) 
        self.y = x[:, :, 1].to(device, dtype=torch.float32).requires_grad_(True)

        if dim == 3:
            self.z = x[:, :, 2].to(device, dtype=torch.float32).requires_grad_(True)

        self.t = t.to(device, dtype=torch.float32).requires_grad_(True)
        self.normals = normal.to(device, dtype=torch.float32).requires_grad_(True)

        self.geometric_descriptor = [geometric_descriptor[:, :, i].to(device, dtype=torch.float32).requires_grad_(True) for i in range(geometric_descriptor.shape[-1])]
        
        if F is not None:
            self.F = F.to(device, dtype=torch.float32).requires_grad_(True)
        if u is not None:
            self.u = u.to(device, dtype=torch.float32).requires_grad_(True)
        if D is not None:
            self.D = D.to(device, dtype=torch.float32).requires_grad_(True)
        if V is not None:
            self.V = V.to(device, dtype=torch.float32).requires_grad_(True)

        if initial_data:
            self.x = self.x[:, 0].unsqueeze(-1)
            self.y = self.y[:, 0].unsqueeze(-1)

            if dim == 3:
                self.z = self.z[:, 0].unsqueeze(-1)
            
            self.t = self.t[:, 0].unsqueeze(-1)
            self.normals = self.normals[:, 0, :]
            self.geometric_descriptor = [geometric_descriptor[:, 0, i].unsqueeze(-1).to(device, dtype=torch.float32).requires_grad_(True) for i in range(geometric_descriptor.shape[-1])]
            
            if hasattr(self, 'F'):
                self.F = self.F[:, 0, :]
            if hasattr(self, 'u'): 
                self.u = self.u[:, 0, :]
            if hasattr(self, 'D'):
                self.D = self.D[:, 0, :]


class DataSplit():
    def __init__(self):

        self.supervised_inputs = [Input]
        self.supervised_targets = [Target]

        self.initial_inputs = [Input]
        self.initial_targets = [Target]

        self.collocation_inputs = [Input]
        self.collocation_targets = [Target]

        self.boundary_inputs = [Input]
        self.boundary_targets = [Target]

        self.sensor_inputs = [Input]
        self.sensor_targets = [Target]

    
    def print_data_split_shapes(self):

        print(f"Supervised input shape: {self.supervised_inputs.shape}")
        print(f"Supervised output shape: {self.supervised_targets.shape}")

        print("\n")
        print(f"Supervised initial input shape: {self.initial_inputs.shape}")
        print(f"Supervised initial output shape: {self.initial_targets.shape}")

        print("\n")
        print(f"Collocation input shape: {self.collocation_inputs.shape}")
        print(f"Collocation output shape: {self.collocation_targets.shape}")

        print("\n")
        print(f"Boundary input shapes: {self.boundary_inputs.shape}")
        print(f"Boundary output shapes: {self.boundary_targets.shape}")
        print(f"Boundary normals shapes: {self.boundary_normals.shape}")

        print("\n")
        mask = (self.boundary_inputs[:, 0, :2].unsqueeze(1) == self.supervised_inputs[:, 0, :2]).all(dim=-1).any(dim=-1)
        overlap = mask[mask == True]
        print(f"Number of supervised points at boundary: {overlap.shape}")


class DataLoaderGroup():
    def __init__(self, data_split: DataSplit, batch_size: int, num_collocation_points: int, num_bc_points: int, num_initial_points: int, seed: int, use_sensors: bool = False):
        
        self.data_split = data_split
        supervised_inputs = data_split.supervised_inputs if not use_sensors else data_split.sensor_inputs
        supervised_targets = data_split.supervised_targets if not use_sensors else data_split.sensor_targets

        self.num_collocation_points = num_collocation_points
        self.num_bc_points = num_bc_points
        self.num_initial_points = num_initial_points

        self.seed = seed
        self.epoch = 0

        self.g = torch.Generator()
        self.g.manual_seed(seed)

        data_size = supervised_inputs.shape[0] + num_collocation_points + num_bc_points + num_initial_points
        num_batches = math.ceil(data_size / batch_size)
        self.num_batches = num_batches

        if use_sensors:
            self.supervised_loader = DataLoader(CustomDataset(supervised_inputs, supervised_targets),
                                    batch_size=14,
                                    shuffle=True, generator=self.g)
        else:
            self.supervised_loader = DataLoader(CustomDataset(supervised_inputs, supervised_targets),
                                    batch_size=math.floor(supervised_inputs.shape[0] / num_batches),
                                    shuffle=True, generator=self.g)
        
        if self.num_collocation_points > 0 and self.num_bc_points > 0 and self.num_initial_points > 0:
            self.update_collocation_loader(self.epoch, generator=self.g)
     

    def __iter__(self):
        if self.num_collocation_points > 0 and self.num_bc_points > 0 and self.num_initial_points > 0:
            self.update_collocation_loader(self.epoch, generator=self.g)
            self.epoch += 1
            return zip(self.supervised_loader, self.initial_loader, self.collocation_loader, self.boundary_loader)
        else:
            return iter(self.supervised_loader)
    
    def __len__(self):
        return len(self.supervised_loader)
    
    def update_collocation_loader(self, epoch: int, generator: torch.Generator):
        # Resample initial, collocation, and boundary points for each epoch
        seed = self.seed + epoch
        initial_indices = select_random_indices_from_data(input_data=self.data_split.initial_inputs,
                                                                                    number_of_points=self.num_initial_points,
                                                                                    random_seed=seed)

        collocation_indices = select_random_indices_from_data(input_data=self.data_split.collocation_inputs,
                                                                                    number_of_points=self.num_collocation_points,
                                                                                    random_seed=seed)
        
        boundary_indices = select_random_indices_from_data(input_data=self.data_split.boundary_inputs,
                                                                                    number_of_points=self.num_bc_points,
                                                                                    random_seed=seed)
        
        collocation_inputs, collocation_targets = self.data_split.collocation_inputs[collocation_indices], self.data_split.collocation_targets[collocation_indices]
        boundary_inputs, boundary_targets = self.data_split.boundary_inputs[boundary_indices], self.data_split.boundary_targets[boundary_indices]
        initial_inputs, initial_targets = self.data_split.initial_inputs[initial_indices], self.data_split.initial_targets[initial_indices]
        
        self.collocation_loader = DataLoader(CustomDataset(collocation_inputs, collocation_targets),
                                        batch_size=math.floor(collocation_inputs.shape[0] / self.num_batches),
                                        shuffle=True, generator=generator)
        
        self.boundary_loader = DataLoader(CustomDataset(boundary_inputs, boundary_targets),
                                        batch_size=math.floor(boundary_inputs.shape[0] / self.num_batches),
                                        shuffle=True, generator=generator)
        
        self.initial_loader = DataLoader(CustomDataset(initial_inputs, initial_targets),
                                        batch_size=math.floor(initial_inputs.shape[0] / self.num_batches),
                                        shuffle=True, generator=generator)