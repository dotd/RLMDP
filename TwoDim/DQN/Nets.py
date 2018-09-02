import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, dim_state, num_actions, init_values=None):
        super(DQN1Layer, self).__init__()
        self.dim_state = dim_state
        self.W1 = nn.Linear(self.dim_state, out_features=num_actions)
        if init_values is not None:
            if type(init_values) is dict:
                self.W1.weight.data = torch.Tensor(init_values["weight"])
                self.W1.bias.data = torch.Tensor(init_values["bias"])
            else:
                self.W1.weight = nn.Parameter(torch.zeros(dim_state, num_actions))
                self.W1.bias = torch.nn.Parameter(torch.ones(dim_state))

    def forward(self, x):
        # flatten
        # x = x.view(x.size(0), self.len)
        x = self.W1(x)
        x = F.relu(x)
        return x


def test_DQN1Layer():
    dqn = DQN1Layer(dim_state=2, num_actions=2, init_values=True)
    print("Weights are:\n{}".format(dqn.W1.weight))
    print("Bias is:\n{}".format(dqn.W1.bias))

    dqn = DQN1Layer(dim_state=2, num_actions=2, init_values={"weight": [[0, 1], [2, 3]], "bias": [-4, -5]})
    print("Weights are:\n{}".format(dqn.W1.weight))
    print("Bias is:\n{}".format(dqn.W1.bias))

    input_vec = torch.Tensor([[0.5, 1], [1, 2]])
    output_vec = dqn.forward(input_vec)
    print("if input is:\n{}\nthen, output is:\n{}".format(input_vec, output_vec))

if __name__ == "__main__":
    test_DQN1Layer()