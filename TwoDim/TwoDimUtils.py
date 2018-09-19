from collections import namedtuple
import random
import torch
import numpy as np
from TwoDim.TwoDimMDP import TwoDimSparseMDPSimulator

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
FilterSample = namedtuple('FilterSample', ('state', 'action', 'value'))
PolicyGradientSample = namedtuple('PolicyGradientSample', ('likelihood', 'reward_func'))


def action_2d_2_letter(action):
    letter = None
    if action==0:
        letter = "U"
    elif action==1:
        letter = "D"
    elif action == 2:
        letter = "L"
    elif action == 3:
        letter = "R"
    else:
        letter = "*"
    return letter


class ReplayMemory(object):

    def __init__(self, capacity, tuple_type=Transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.tuple_type = tuple_type

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.tuple_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_screen(state, environment):
    screen_2d_np = environment.transform_into_screen(state)
    screen_4d_torch = torch.from_numpy(screen_2d_np)
    screen_4d_torch = torch.reshape(screen_4d_torch, (1, 1, screen_2d_np.shape[0], screen_2d_np.shape[1]))
    return screen_4d_torch


def smooth_signal(signal, window_smooth_len):
    f = np.array([1] * window_smooth_len)
    f = f / len(f)
    vec = np.convolve(signal, f)
    vec = vec[window_smooth_len:-window_smooth_len]
    return vec


def get_paper_mdp():
    '''
    shape, noise_prob, rewards, random, start_states, terminal_states
    :return:
    '''
    shape = (6,7)
    noise_prob = 0.1
    rewards = {(shape[0]-1,shape[1]-1):1}
    random = np.random.RandomState(0)
    start_states = [(0,0)]
    terminal_states = [(shape[0]-1,shape[1]-1)]
    mdp = TwoDimSparseMDPSimulator(shape, noise_prob, rewards, random, start_states, terminal_states)
    return mdp


mdp = get_paper_mdp()