import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import ReplayMemory as rm
from ReplayMemory import Transition
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

def get_batch_from_trajectory(trajectory, batcvh_size):
    pass

class DeepJSolver():
    def __init__(self, X, gamma, layer_sizes=(2, 2, 1), layer_activations=(nn.Hardtanh, None), input_mode="one_hot", base = "identity"):
        self.X = X
        self.gamma = gamma
        self.input_mode=input_mode
        self.replay_memory = None

        layers = OrderedDict()

        if base==None:
            self.base = np.random.normal(shape=(X, layer_sizes[0]))
        elif base=="identity":
            self.base = np.identity(self.X)
        else:
            self.base = base


        for i in range(len(layer_sizes)-1):
            layers["Layer" + str(i)] = nn.Linear(layer_sizes[i],layer_sizes[i+1])
            if layer_activations[i]!=None:
                layers["Activation" + str(i)] = layer_activations[i](inplace=True)
        self.model = nn.Sequential(layers)
        print(self.model)

        #self.loss_fn = nn.MSELoss(size_average=False)
        self.learning_rate = 1e-2

        # filter to remove the parameters that doesn’t require gradients
        # Here is the embbedding layer
        self.parameters = self.model.parameters()
        self.optimizer = torch.optim.Adagrad(self.parameters, lr=self.learning_rate)
        #self.optimizer = optim.RMSprop(self.model.parameters())
        self.loss_fn = torch.nn.L1Loss(size_average=True)

    def add_trajectory(self, trajectory):
        # We initialize replay memory
        # it is a tuple of x,u,r,y
        self.replay_memory = rm.ReplayMemory(capacity=len(trajectory))
        for i in range(len(trajectory)):
            #print(trajectory[i])
            x = trajectory[i][0]
            u = trajectory[i][1]
            r = trajectory[i][2]
            y = trajectory[i][3]
            self.replay_memory.push(x,u,y,r)

    def step(self, num_samples=None):
        # Get the transitions
        num_samples = self.replay_memory.capacity if not num_samples else num_samples
        self.transitions = self.replay_memory.sample(num_samples)

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*self.transitions))

        state_base = self.base[list(batch.state)]
        state_batch = Variable(torch.Tensor(state_base), requires_grad=True)
        reward_batch = Variable(torch.Tensor(list(batch.reward)), requires_grad=False)
        next_state_base = self.base[list(batch.next_state)]
        next_state_batch = Variable(torch.Tensor(next_state_base), requires_grad=True)

        # Compute J(s_t)
        Jt = self.model(state_batch)
        Jt_next_state = self.model(next_state_batch)
        expected_reward = Jt - self.gamma * Jt_next_state
        # Compute Huber loss
        loss = self.loss_fn(expected_reward, reward_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def infer(self):
        np_samples = self.base[range(self.X)]
        input = Variable(torch.Tensor(np_samples))
        self.J = self.model(input)
        return self.J

    def step_the_batch(self, batch_size):
        transitions = self.replay_memory.sample(batch_size)
        pass


#DeepSolver(X=2, gamma=0.5)
