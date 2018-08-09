import torch
import torch.optim as optim
import torch.nn.functional as F

from TwoDim.Nets import DQN2
from TwoDim.TwoDimUtils import ReplayMemory
from TwoDim.TwoDimUtils import Transition

from Filters.OnlineUtils import OnlineFilter2

import random
import math

from collections import namedtuple

FilterSample = namedtuple('FilterSample', ('state', 'action','value'))

class AgentFilter():
    def __init__(self, grid_sizes, num_actions, **kwargs):
        self.memory_replay_len = kwargs.get("memory_replay_len", 1000)
        self.filter_replay_len = kwargs.get("filter_replay_len", 1000)
        self.lr = kwargs.get("lr")
        self.EPS_START = kwargs.get("EPS_START", 0.9)
        self.EPS_END = kwargs.get("EPS_END", 0.05)
        self.EPS_DECAY = kwargs.get("EPS_DECAY", 10)
        self.BATCH_SIZE = kwargs.get("BATCH_SIZE", 100)
        self.GAMMA = kwargs.get("GAMMA")
        self.num_actions = num_actions
        self.device = torch.device(kwargs.get("device","cpu"))
        self.intermediate = kwargs.get("intermediate",60)

        self.policy_net = DQN2(sizes=grid_sizes, intermediate=self.intermediate).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayMemory(self.memory_replay_len)
        self.filter_replay = ReplayMemory(self.filter_replay_len, tuple_type=FilterSample)
        self.steps_done = 0
        self.episode_durations = []

        # Filter section
        filter_in = kwargs.get("filter_in",[1] * 3)
        sampler_len = kwargs.get("sampler_len",1)
        self.online_filter = OnlineFilter2(filter_in, sampler_len=sampler_len)

        # Checking conditions.
        if self.BATCH_SIZE > self.memory_replay_len or self.BATCH_SIZE>self.filter_replay_len:
            raise NameError('BATCH_SIZE > min(memory_replay_len, filter_replay_len)')

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                vec = self.policy_net(state.float())
                action_selected = vec.max(1)[1].view(1, 1)
                return action_selected, True
        else:
            action_random = torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)
            return action_random, eps_threshold

    def optimize_model(self):
        if len(self.memory) < self.memory_replay_len:
            return False
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        '''
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        '''
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values = self.policy_net(next_state_batch.float()).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.float()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def optimize_filter(self):
        if len(self.filter_replay) < self.filter_replay_len:
            return False
        filter_samples = self.filter_replay.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = FilterSample(*zip(*filter_samples))

        '''
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        '''
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        value_batch = torch.cat(batch.value)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.float(), value_batch.unsqueeze(1).float())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def update_filters(self, state, action, reward):
        # We begin with adding a sample
        result = self.online_filter.add(reward, (state, action))
        if result is None:
            return
        # A result is ready. We add it to the filter_replay
        ready_state = result[2][0]
        ready_action = result[2][1]
        value = torch.tensor([result[0]], device=self.device, dtype=torch.float64)
        self.filter_replay.push(ready_state, ready_action, value)
