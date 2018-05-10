import numpy as np
import math
from ReplayMemory import ReplayMemory, Transition
from collections import deque

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

step_factor = 0.75
step_start = 1e2

class Q_Learning():
    def __init__(self, U, random, eps_max_action, **kwargs):
        self.U = U
        if isinstance(self.U, int):
            self.U = list(range(self.U))
        self.envAction2index = {}
        for idx,u in enumerate(self.U):
            self.envAction2index[u] = idx
        self.Q = {}

        self.replay_memory = ReplayMemory(capacity=100, random=random)
        self.steps = 0
        self.random = random
        self.eps_max_action = eps_max_action
        self.sizes = kwargs.get("sizes", None)

    def __str__(self):
        if self.sizes is not None and len(self.sizes)==2:
            lines = []
            lines.append(str(self.U))
            for y in range(self.sizes[1]):
                line = []
                for x in range(self.sizes[0]):
                    if (x,y) in self.Q:
                        s = np.argmax(self.Q[(x,y)])
                        if s==0: s="L"
                        if s==1: s="R"
                        if s==2: s="D"
                        if s==3: s="U"
                    else:
                        s = "-"
                    line.append(str(s))
                lines.append(" ".join(line))
            for x in range(self.sizes[0]):
                for y in range(self.sizes[1]):
                    if (x,y) in self.Q:
                        lines.append(str(x) + "," + str(y) + "->" + str(self.Q[(x,y)]))

            return "\n".join(lines)
        return "NNOONNEE"

    def compute_optimal_action(self, state):
        if state in self.Q:
            entry = self.Q[state]
            maximal_value = np.amax(entry) - self.eps_max_action
            idx_near_optimal = np.where(entry >= maximal_value)[0]
            action_idx = idx_near_optimal[self.random.choice(idx_near_optimal.shape[0])]
            action = self.U[action_idx]
            return action
        action_idx = self.random.choice(len(self.U))
        return self.U[action_idx]


    def choose_action(self, state):
        sample = self.random.uniform()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                  math.exp(-1. * self.steps / EPS_DECAY)
        self.steps += 1
        if sample > self.eps_threshold:
            action = self.compute_optimal_action(state)
            return action
        else:
            action = self.U[self.random.choice(len(self.U))]
            return action

    def update(self):
        if self.steps < len(self.replay_memory):
            return
        samples = self.replay_memory.sample(1)
        for sample in samples:
            if sample.state not in self.Q:
                self.Q[sample.state] = self.random.normal(size=(len(self.U),))*0
            if sample.next_state not in self.Q:
                self.Q[sample.next_state] = self.random.normal(size=(len(self.U),))*0
            lr = step_start/(self.steps**step_factor)
            maximal_next_action = self.compute_optimal_action(sample.next_state)
            maximal_next_action_idx = self.envAction2index[maximal_next_action]
            maximal_action_idx = self.envAction2index[sample.action]
            self.Q[sample.state][maximal_action_idx] = self.Q[sample.state][maximal_action_idx] + lr * (sample.reward + GAMMA*self.Q[sample.next_state][maximal_next_action_idx] - self.Q[sample.state][maximal_action_idx])

    def add_tuple(self, tuple):
        self.replay_memory.push(*tuple)
