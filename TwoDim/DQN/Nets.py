import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN2Layers(nn.Module):
    def __init__(self, dim_state, num_actions, intermediate):
        super(DQN2Layers, self).__init__()
        self.dim_state = dim_state
        self.W1 = nn.Linear(self.dim_state, out_features=intermediate)
        self.W2 = nn.Linear(intermediate, out_features=num_actions)

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), self.len)
        x = self.W1(x)
        x = F.relu(x)
        x = self.W2(x)
        return x


class DQN1Layer(nn.Module):
    def __init__(self, dim_state, num_actions, init_values=None, output_activation=F.tanh):
        super(DQN1Layer, self).__init__()
        self.dim_state = dim_state
        self.W1 = nn.Linear(self.dim_state, out_features=num_actions)
        if init_values is not None:
            if type(init_values) is dict:
                # Given a dictionary, initializing according to the dict.
                self.W1.weight.data = torch.Tensor(init_values["weight"])
                self.W1.bias.data = torch.Tensor(init_values["bias"])
            elif isinstance(init_values, str) and init_values.lower()=="zeros":
                # Initialize to zeros.
                self.W1.weight = nn.Parameter(torch.zeros(num_actions, dim_state.item()))
                self.W1.bias = nn.Parameter(torch.zeros(num_actions))
        self.output_activation = output_activation

    def forward(self, x):
        # flatten
        # x = x.view(x.size(0), self.len)
        x = self.W1(x)
        x = self.output_activation(x)
        return x.view(x.size(0), -1)

