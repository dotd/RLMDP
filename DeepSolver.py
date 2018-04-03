import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

def deep_simple_solver(mdp):
    pass

#def get_J_as_deep(trajectory, gamma, X=None):
#    X = np.array([vec[0] for vec in trajectory]).max()+1 if X is None else X

class DeepSolver():
    def __init__(self, X, gamma, layer_sizes=(2, 2, 1), layer_activations=(nn.ReLU, None), input_mode="one_hot"):
        self.X = X
        self.gamma = gamma
        self.input_mode=input_mode

        layers = OrderedDict()

        if self.input_mode=="one_hot":
            embed = nn.Embedding(self.X,self.X)
            embed.weight.data.copy_(torch.from_numpy(np.identity(self.X)))
            layers["embedding_static"] = embed

        for i in range(len(layer_sizes)-1):
            layers["Layer" + str(i)] = nn.Linear(layer_sizes[i],layer_sizes[i+1])
            if layer_activations[i]!=None:
                layers["Activation" + str(i)] = layer_activations[i](inplace=True)
        self.model = nn.Sequential(layers)
        print(self.model)

        self.loss_fn = nn.MSELoss(size_average=False)
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)



DeepSolver(X=2, gamma=0.5)