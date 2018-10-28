from collections import deque
import numpy as np


class ComputeRisk:

    def __init__(self, gamma, window_size, maximal_num_samples):
        self.gamma = gamma
        self.maximal_num_samples = maximal_num_samples
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)
        self.states = dict()
        self.gamma_vec = [1]
        for i in range(1, self.window_size):
            self.gamma_vec.append(self.gamma_vec[i-1] * gamma)

    def add(self, state, reward):
        self.deque.append((state, reward))
        if len(self.deque) < self.window_size:
            return

        state0 = self.deque[0][0]
        B0 = np.sum([self.deque[i][1] * self.gamma_vec[i] for i in range(self.window_size)])

        if state0 not in self.states:
            self.states[state0] = deque(maxlen=self.maximal_num_samples)
        self.states[state0].append(B0)

    def compute_var(self, state):
        if state not in self.states or len(self.states) < 2:
            return None
        return np.std(self.states[state])