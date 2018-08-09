import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

def build_network(layer_sizes, layer_activations):
    layers = OrderedDict()
    for i in range(len(layer_sizes ) -1):
        layers["Layer" + str(i)] = nn.Linear(layer_sizes[i] ,layer_sizes[ i +1])
        if layer_activations[i ] != None:
            layers["Activation" + str(i)] = layer_activations[i](inplace=True)
    model = nn.Sequential(layers)
    return model


def build_base(X, first_layer_size, base_type, random = None):
    if base_type=="identity":
        if first_layer_size!=X:
            raise Exception('identity mode and first layer is not equal to the number of states')
        base = np.identity(X)
        return base
    elif base_type=="random":
        if random==None:
            random = np.random.RandomState(0)
        base = random.normal(size=(X,first_layer_size))
        return base
    else:
        raise Exception('build base: mode not supported ' + str(base_type))