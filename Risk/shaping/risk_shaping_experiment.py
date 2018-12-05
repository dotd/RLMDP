import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class FunctionApproximator(nn.Module):
    def __init__(self, intermediate):
        super(FunctionApproximator, self).__init__()
        self.W1 = nn.Linear(1, out_features=intermediate)
        self.W3 = nn.Linear(intermediate, out_features=intermediate)
        self.W2 = nn.Linear(intermediate, out_features=1)

    def forward(self, x):
        # flatten
        x = self.W1(x)
        x = F.relu(x)
        x = self.W3(x)
        x = F.relu(x)
        x = self.W2(x)
        #x = F.sigmoid(x)
        return x


f = lambda x: math.sqrt(x)
N = 1000
# create samples
x = np.linspace(0, 2, N)
y = np.array([f(i) for i in x])

model = FunctionApproximator(100)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
random = np.random.RandomState(0)

x_orig = x.reshape((-1,1))
y_orig = y.reshape((-1,1))
y_pred = None

for t in range(500):
    x = np.copy(x_orig)
    y = np.copy(y_orig)
    perm = random.permutation(N)
    x = x[perm]
    y = y[perm]
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(torch.Tensor(x))
    #print(y_pred)

    # Compute and print loss
    loss = criterion(y_pred, torch.Tensor(y))
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_pred = model(torch.Tensor(x_orig))

    plt.plot(x_orig, y_pred.detach().numpy(),'r')
    plt.draw()
plt.plot(x_orig, y_pred.detach().numpy(),'g')

"""
random = np.random.RandomState(0)
for rep in range(10):
    # permute the data
    idx = random.permutation(N)
    x = x[idx]
    y = y[idx]
    # doing the fit

    pass
"""



plt.plot(x_orig, y_orig,'b')
plt.show()