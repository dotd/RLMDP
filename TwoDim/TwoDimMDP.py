import numpy as np
from copy import deepcopy

class TwoDimSparseMDPSimulator():
    '''
    Actions up: (0,-1), right(1,0), down (0,1), left (-1,0)
    '''
    def __init__(self, sizes, noise, rewards, random, start_states, terminal_states):
        self.sizes = sizes
        self.dim = len(sizes)
        self.noise = noise
        self.rewards = rewards
        self.random = random
        self.terminal_states = terminal_states
        self.start_states = start_states

    def get_start_state(self):
        return self.start_states[self.random.choice(len(self.start_states))]

    def is_terminal(self, state):
        return state in self.terminal_states


    def get_actions_list(self):
        actions = []
        for dim in range(self.dim):
            for direction in [-1,1]:
                action = [0 for i in range(self.dim)]
                action[dim] = direction
                actions.append(tuple(action))
        return actions

    def compute_next_state(self, pos, action):
        nxt_state = [min(max(pos[d] + action[d],0),self.sizes[d]-1) for d in range(self.dim)]
        return tuple(nxt_state)

    def get_adjucent_squares(self, orig):
        possible_squares = set()
        for d in range(self.dim):
            for m in [-1, 1]:
                pos = list(orig)
                pos[d] = min(max(pos[d] + m,0),self.sizes[d]-1)
                possible_squares.add(tuple(pos))
        return list(possible_squares)

    def get_next_state(self, pos, action):
        ## Check if in terminal state
        if pos in self.terminal_states:
            start_state = self.start_states[self.random.choice(len(self.start_states))]
            reward = self.get_reward_in_state(start_state)
            return start_state, reward

        nxt = self.compute_next_state(pos, action)
        # Add noise. With noise probability we get into uniformly into any of the adjucent squares
        if self.random.uniform(0,1)<self.noise:
            possible_squares = self.get_adjucent_squares(nxt)
            #print(possible_squares)
            nxt = possible_squares[self.random.choice(len(possible_squares))]
        reward = self.get_reward_in_state(nxt)
        return nxt,reward

    def get_reward_in_state(self, state):
        if state in self.rewards.keys():
            return self.rewards[state]
        return 0

