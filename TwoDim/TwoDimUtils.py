from collections import namedtuple
import random
import torch
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

    def __init__(self, capacity, tuple_type = Transition):
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


def smooth_signal(signal, filt_len):
    f = np.array([1] * filt_len)
    f = f / len(f)
    vec = np.convolve(signal, f)
    vec = vec[filt_len:-filt_len]
    return vec
