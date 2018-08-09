
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity, random):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.random = random


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=None):
        if batch_size==None:
            return self.sample(self.capacity)
        random_batch = [self.memory[i] for i in self.random.choice(len(self.memory), batch_size)]
        return random_batch

    def get_all(self):
        return self.memory

    def __len__(self):
        return len(self.memory)