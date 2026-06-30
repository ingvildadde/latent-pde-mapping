import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, depth, act, final_act = False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Activation functions
        self.act = act 
        self.final_act = final_act

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, num_classes))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)

        if self.final_act == False:
            return x
        else:
            return torch.relu(x)