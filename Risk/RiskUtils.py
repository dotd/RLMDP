from collections import deque
import numpy as np


class ComputeRiskSimple:

    def __init__(self,
                 gamma,  # Gamma of the MDP
                 window_size,  # Finite sample approximation window. Used by the deque
                 ):
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

    def __init__(self,
                 gamma,  # Gamma of the MDP
                 window_size,  # Finite sample approximation window. Used by the deque
                 maximal_num_samples,  # how many recent samples to base our estimation on
                 ):
        ComputeRiskSimple.__init__(self, gamma, window_size)
        self.maximal_num_samples = maximal_num_samples
        # We refer here to discrete states
        self.reset()

    def reset(self):
        """
        This method resets the dictionaries
        """
        self.states = dict()
        self.state_actions = dict()

    def free_tail(self):
        """If get into a final state, we have still several risk estimations until the end of the episode."""
        for p in range(1, len(self.deque)):
            B0 = np.sum([self.deque[i][1] * self.gamma_vec[i] for i in range(p, len(self.deque))])
            

    def add(self, state, reward, action):
        B0 = ComputeRiskSimple.add(self, state, reward, action)
        if B0 is None:
            return None

        # (1) take the first sample, i.e., the state, and add statistics of it.
        state0 = self.deque[0][0]
        # Check if the state is in the dictionary. If not, we add it
        if state0 not in self.states:
            self.states[state0] = deque(maxlen=self.maximal_num_samples)
        # Add the new sample
        self.states[state0].append(B0)

        # (2) take the first sample, i.e., the state-action, and add statistics of it.
        action0 = self.deque[0][2]
        # Define state_action for the dictionary
        state_action = (state0, action0)
        # Check if the state_action is in the dictionary. If not, we add it
        if state_action not in self.state_actions:
            self.state_actions[state_action] = deque(maxlen=self.maximal_num_samples)
        # Add the new sample
        self.state_actions[state_action].append(B0)

        return B0

    def compute_var(self, state, action=None, risk_func=None):
        if action is None:
            if state not in self.states or len(self.states[state]) < 2:
                return None
            if risk_func is None:
                return np.std(self.states[state])
            # Compute J
            J = np.mean(self.states[state])
            vec = [risk_func(x-J) for x in self.states[state]]
            return np.mean(vec)
        else:
            state_action = (state, action)
            if state_action not in self.state_actions or len(self.state_actions[state_action]) < 2:
                return None
            if risk_func is None:
                return np.std(self.state_actions[state_action])
            Q = np.mean(self.state_actions[state_action])
            vec = [risk_func(x-Q) for x in self.state_actions[state_action]]
            return np.mean(vec)


    def get_risk_map(self, shape):
        map = np.zeros(shape=(shape[0]*2+1, shape [1]))
        for state in self.states:
            map[state[0]+shape[0], state[1]] = np.var(self.states[state])
        return map
