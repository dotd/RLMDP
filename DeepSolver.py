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

class DeepSolver():
    def __init__(self, X, gamma, layer_sizes=(2, 2, 1), layer_activations=(nn.ReLU, None), input_mode="one_hot"):
        self.X = X
        self.gamma = gamma
        self.input_mode=input_mode
        self.replay_memory = None

        layers = OrderedDict()

        if self.input_mode=="one_hot":
            embed = nn.Embedding(self.X,self.X)
            embed.weight.data.copy_(torch.from_numpy(np.identity(self.X)))
            # To make this a constant mapping
            embed.weight.requires_grad = False
            layers["embedding_static"] = embed

        for i in range(len(layer_sizes)-1):
            layers["Layer" + str(i)] = nn.Linear(layer_sizes[i],layer_sizes[i+1])
            if layer_activations[i]!=None:
                layers["Activation" + str(i)] = layer_activations[i](inplace=True)
        self.model = nn.Sequential(layers)
        print(self.model)

        self.loss_fn = nn.MSELoss(size_average=False)
        self.learning_rate = 1e-4

        # filter to remove the parameters that doesn’t require gradients
        # Here is the embbedding layer
        if self.input_mode=="one_hot":
            self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            self.parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        #self.optimizer = optim.RMSprop(self.model.parameters())

    def add_trajectory(self, trajectory):
        # We initialize replay memory
        self.replay_memory = rm.ReplayMemory(capacity=len(trajectory))
        for i in range(len(trajectory)):
            x = trajectory[i][0]
            u = trajectory[i][1]
            r = trajectory[i][2]
            y = trajectory[i][3]
            self.replay_memory.push(x,u,y,r)

    def step(self):
        # Get the transitions
        self.transitions = self.replay_memory.sample()

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        '''
        batch = Transition(*zip(*self.transitions))
        print(batch.state)
        print(batch.next_state)
        print([int(s) for s in batch.next_state])

        # torch.cat
        state_batch = Variable(torch.cat([s for s in batch.next_state]))
        #action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(batch.reward)
        next_state_batch = Variable(batch.next_state)
        '''

        batch = Transition(*zip(*self.transitions))
        print(batch.state)
        print(batch.next_state)
        print([int(s) for s in batch.next_state])

        # torch.cat
        state_batch = Variable(torch.LongTensor([int(round(s)) for s in batch.state]))
        #action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.Tensor([s for s in batch.reward]))
        next_state_batch = Variable(torch.LongTensor([int(round(s)) for s in batch.next_state]))

        # Compute J(s_t)
        Jt = self.model(state_batch)
        Jtp1 = self.model(next_state_batch)
        expected_Jtp0 = (Jtp1 * self.gamma) + reward_batch
        # Compute Huber loss
        loss = F.l1_loss(Jt, expected_Jtp0)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            print(param)
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def infer(self):
        input = Variable(torch.LongTensor(range(self.X)))
        self.pred = self.model(input)
        return self.pred

    def step_the_batch(self, batch_size):
        transitions = self.replay_memory.sample(batch_size)
        pass


DeepSolver(X=2, gamma=0.5)
