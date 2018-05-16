
########### IMPORTS #################
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from TwoDim.TwoDimMDP import TwoDimSparseMDPSimulator
from collections import  deque

########### ENVIRONMENT #################
sizes = (5, 4)
rewards = {(sizes[0]-1, sizes[1]-1): 1}
terminal_states = [(sizes[0]-1, sizes[1]-1)]
# start_states = list(itertools.product(range(sizes[0]),range(sizes[1])))
start_states = [(0, 0)]
np_random = np.random.RandomState(0)
env = TwoDimSparseMDPSimulator(sizes=sizes, noise=0, rewards=rewards, random=np_random, start_states=start_states,
                             terminal_states=terminal_states)
actions = env.get_actions_list()
num_actions = len(actions)


# if gpu is to be used
device = torch.device("cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

'''
class DQN(nn.Module):

    def __init__(self, z_dim_in=1, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=z_dim_in, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.head = nn.Linear(in_features=448, out_features=num_actions)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
'''

class DQN2(nn.Module):
    def __init__(self, sizes, num_actions=4, intermediate = 10):
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

def get_screen(state):
    screen_2d_np = env.transform_into_screen(state)
    screen_4d_torch = torch.from_numpy(screen_2d_np)
    screen_4d_torch = torch.reshape(screen_4d_torch, (1,1,screen_2d_np.shape[0],screen_2d_np.shape[1]))
    return screen_4d_torch

BATCH_SIZE = 50
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 1

policy_net = DQN2(sizes=sizes).to(device)
target_net = DQN2(sizes=sizes).to(device)
initial_net = DQN2(sizes=sizes).to(device)
target_net.load_state_dict(policy_net.state_dict())
initial_net.load_state_dict(policy_net.state_dict())
target_net.eval()
initial_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #print(type(state))
            #print(state.dtype)
            vec = policy_net(state.float())
            #print("AT vec={}".format(vec))
            action = vec.max(1)[1].view(1, 1)
            return action, True
    else:
        action_random = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
        #print("AR vec={}".format(action_random))
        return action_random, eps_threshold


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return False
    transitions = memory.sample(BATCH_SIZE)
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
    state_action_values = policy_net(state_batch.float()).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = policy_net(next_state_batch.float()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

num_episodes = 5000
episode_len = np.prod(sizes)*6
print("episode_max_len={}".format(episode_len))
weights_initial = list(initial_net.parameters())
for i_episode in range(num_episodes):
    # Initialize the environment and state
    print("episode={}".format(i_episode))

    # initialize
    env.reset()
    #print("init_state = {}".format(env.cur_state))

    for t in range(episode_len):

        if t%100==-1:
            print("t={}".format(t))
            weights = list(policy_net.parameters())
            print("weights={}".format(weights))
            print("del0 weights={}".format(weights[0] - weights_initial[0]))
            print("del1 weights={}".format(weights[1] - weights_initial[1]))
            # Select and perform an action

        # Current state
        cur_state_vec = env.cur_state;
        cur_state_screen = get_screen(cur_state_vec)

        # Action
        action, status_action = select_action(cur_state_screen)
        action_idx = action[0][0]
        action_vec = actions[action_idx]

        # step
        next_state_vec = env.step(action_vec)
        next_state_screen = get_screen(next_state_vec)

        # Reward
        cur_reward = env.get_reward(cur_state_vec)
        reward = torch.tensor([cur_reward], device=device)

        # Store the transition in memory
        memory.push(cur_state_screen, action, next_state_screen, reward)

        # Perform one step of the optimization (on the target network)
        status_optimize = optimize_model()
        # if we are done, we do nothing. Just init the dynamics
        #print("t={}, cur_state={}, cur_reward={}, action_idx={}, action_vec={}, next_state_vec={}, status_action={}, optimize={}".format(t,cur_state_vec, cur_reward, action_idx,  action_vec,  next_state_vec, status_action, status_optimize))
        if env.is_terminal(cur_state_vec):
            print("is_terminal {}".format(t))
            break


    # Update the target network
    #print("t={}".format(t))
    episode_durations.append(t + 1)
    if i_episode % TARGET_UPDATE == -1:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 50 ==0 or i_episode:
        plt.figure(1)
        flt_len = 100
        f = np.array([1] * flt_len)
        f = f/len(f)
        vec = np.convolve(episode_durations,f)
        plt.plot(vec[flt_len:-flt_len])
        plt.pause(0.05)
        #plt.draw()
        #plt.show(block=False)

print('Complete')
print(episode_durations)

#plt.figure(1)
#plt.plot(episode_durations)
#plt.show()


