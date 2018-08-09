import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import ReplayMemory as rm
from ReplayMemory import Transition
from torch.autograd import Variable
import DeepAgents.DeepUtils as du
import math

class DQN:
    def __init__(self,
                 X,
                 U,
                 gamma,
                 replay_capacity,
                 layer_sizes,
                 layer_activations,
                 batch_size = 128,
                 base_type = "identity",
                 random=None):
        self.X = X
        self.U = U
        self.gamma = gamma
        self.replay_memory = None
        if random ==None:
            random = np.random.RandomState(0)
        self.random = random

        self.base = du.build_base(X,layer_sizes[0],base_type = base_type,random = random)
        if layer_sizes[-1] != self.U:
            raise Exception("U and last layer are not the same")
        self.model = du.build_network(layer_sizes, layer_activations)
        print(self.model)

        self.learning_rate = 1e-1

        # filter to remove the parameters that doesn’t require gradients
        # Here is the embbedding layer
        self.parameters = self.model.parameters()
        self.optimizer = torch.optim.Adagrad(self.parameters, lr=self.learning_rate)
        self.loss_fn = nn.MSELoss(size_average=True)
        self.replay_memory = rm.ReplayMemory(replay_capacity)

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.batch_size = batch_size
        self.steps_done = 0

    def update(self, x,u,r,y):
        self.replay_memory.push(x,u,y,r)
        self.optimize_model()

    def infer(self):
        np_samples = self.base[range(self.X)]
        input = Variable(torch.Tensor(np_samples))
        self.J = self.model(input)
        return self.J

    def select_action(self, state):
        sample = self.random.uniform(0,1)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            # Note: returns a Variable of size 1x1
            action = self.model(Variable(torch.Tensor(self.base[state]))).data.max(0)[1].view(-1,1)
        else:
            # Note: returns a LongTensor of size 1x1
            action = torch.LongTensor([[self.random.choice(self.U)]])
        # get the value of the action from LongTensor/Variable
        action = action[0][0]
        return action

    def optimize_model(self):
        if len(self.replay_memory) < self.batch_size:
            return
        transitions = self.replay_memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.FloatTensor(self.base[batch.state,:]))
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1))
        reward_batch = Variable(torch.FloatTensor(batch.reward)).view(-1,1)
        
        next_state_batch = Variable(torch.FloatTensor(self.base[batch.next_state,:]))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # next_state_values = Variable(torch.zeros(self.batch_size).type(torch.FloatTensor))
        next_state_values = self.model(next_state_batch).max(1)[0].view(-1,1)
        # Compute the expected Q values
        # expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        # expected_state_action_values = Variable(expected_state_action_values.data)

        # Compute Huber loss
        loss = self.loss_fn(state_action_values - next_state_values * self.gamma, reward_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return None

    def get_policy(self):
        policy = np.zeros(shape=(self.X, self.U))
        idx = self.model(Variable(torch.FloatTensor(self.base))).max(1)[1].data
        for i,index in enumerate(idx):
            policy[i][index] = 1.0
        return policy

    def get_Q(self):
        Q = self.model(Variable(torch.FloatTensor(self.base))).data
        return Q
