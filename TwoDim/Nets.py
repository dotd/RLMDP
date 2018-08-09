import torch.nn as nn
import torch.nn.functional as F


class DQN2(nn.Module):
    def __init__(self, sizes, num_actions=4, intermediate = 40):
        super(DQN2, self).__init__()
        self.len = sizes[0] * sizes[1]
        self.W1 = nn.Linear(self.len, out_features=intermediate)
        self.W2 = nn.Linear(intermediate, out_features=num_actions)

    def forward(self, x):
        # flatten
        x = x.view(x.size(0),self.len)
        x = self.W1(x)
        x = F.relu(x)
        x = self.W2(x)
        return x