from collections import deque
import numpy as np


class ComputeRiskSimple:

    def __init__(self, gamma, window_size):
        self.gamma = gamma
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)
        self.gamma_vec = [1]
        for i in range(1, self.window_size):
            self.gamma_vec.append(self.gamma_vec[i-1] * gamma)

    def add(self, state, reward, action):
        # (1) Add new state-reward
        self.deque.append((state, reward, action))
        if len(self.deque) < self.window_size:
            return None
        # (2) take the first sample, i.e., the state, and add statistics of it.
        B0 = np.sum([self.deque[i][1] * self.gamma_vec[i] for i in range(self.window_size)])
        return B0


class ComputeRiskGeneral(ComputeRiskSimple):

    def __init__(self, gamma, window_size, maximal_num_samples):
        ComputeRiskSimple.__init__(self, gamma, window_size)
        self.maximal_num_samples = maximal_num_samples
        self.states = dict()
        self.state_actions = dict()

    def add(self, state, reward, action):
        B0 = ComputeRiskSimple.add(self, state, reward, action)
        if B0 is None:
            return None

        state0 = self.deque[0][0]
        action0 = self.deque[0][2]
        if state0 not in self.states:
            self.states[state0] = deque(maxlen=self.maximal_num_samples)
        self.states[state0].append(B0)

        # (2) take the first sample, i.e., the state-action, and add statistics of it.
        state_action = (state0, action0)
        if state_action not in self.state_actions:
            self.state_actions[state_action] = deque(maxlen=self.maximal_num_samples)
        self.state_actions[state_action].append(B0)

        return B0

    def compute_var(self, state, action=None):
        if action is None:
            if state not in self.states or len(self.states[state]) < 2:
                return None
            return np.std(self.states[state])
        else:
            state_action = (state, action)
            if state_action not in self.state_actions or len(self.state_actions[state_action]) < 2:
                return None
            return np.std(self.state_actions[state_action])

