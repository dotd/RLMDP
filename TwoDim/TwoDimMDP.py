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

        # for self managing environment
        self.cur_state = self.get_random_start_state()
        self.actions_list = self.get_actions_list()

    def get_uniform_random_state(self):
        vec = [0] * self.dim
        for i in range(self.dim):
            vec[i] = self.random.randint(self.sizes[i])
        return tuple(vec)


    def get_random_start_state(self):
        if len(self.start_states)==1:
            return self.start_states[0]
        self.cur_state = self.start_states[self.random.choice(len(self.start_states))]
        return self.cur_state[0]

    def transform_into_screen(self, state):
        '''
        Relevant only to 2D state spaces
        Transform the state space into screen, shape of it is 2D where X is the first axis, Y is the second, etc.
        '''

        # Put the agent location on the screen
        screen = np.zeros(shape=self.sizes)
        screen[state[0]][state[1]] = 1

        # Put the rewards on the screen
        #for location, reward in self.rewards.items():
        #    screen[location[0]][location[1]] = reward

        return screen

    def is_terminal(self, state=None):
        if state is None:
            state = self.cur_state
        return state in self.terminal_states

    def get_actions_list(self):
        '''
        Get all the actions sorted by the relevant index
        :returns list of action tuples
        '''
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

    def step(self, action):
        next_state, next_reward = self.get_next_state(action=action)
        self.cur_state = next_state
        return self.cur_state

    def get_next_state(self, pos=None, action=None):
        '''
        action here is vec
        if action is int, we transform it into vec
        '''

        # if action is int, we transform it into vec
        if isinstance(action, int):
            action = self.actions_list[action]

        if pos is None:
            pos = self.cur_state
        ## Check if in terminal state
        if pos in self.terminal_states:
            start_state = self.start_states[self.random.choice(len(self.start_states))]
            reward = self.get_reward_in_state(start_state)
            return start_state, reward

        nxt = self.compute_next_state(pos, action)
        # Add noise. With noise probability we get into uniformly into any of the adjucent squares
        if self.random.uniform(0,1)<self.noise:
            possible_squares = self.get_adjucent_squares(nxt)
            #print(type(possible_squares))
            #print(len(possible_squares))
            nxt = possible_squares[self.random.choice(len(possible_squares))]
        reward = self.get_reward_in_state(nxt)
        return nxt,reward

    def get_reward_in_state(self, state):
        if state in self.rewards.keys():
            return self.rewards[state]
        return 0

    def get_reward(self, state=None):
        if state is None:
            state = self.cur_state
        return self.get_reward_in_state(state)

    def reset(self):
        self.cur_state = self.get_random_start_state()





