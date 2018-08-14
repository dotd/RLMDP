from itertools import product

from gym import Env
import numpy as np


class Minefield(Env):
    """
    Perhaps even inherit from GoalEnv?
    """
    def __init__(self,
                 shape=np.array([64, 53]),
                 mine_penalty=-40,
                 reach_reward=100,
                 rand_action_prob=0.05,
                 num_mines=80,
                 start=np.array([np.array([60, 53])]),
                 end=(60, 2),
                 random_generator=None):

        self.rand_gen = random_generator if random_generator else self.seed(42)
        self.dim = len(shape)
        self.shape = shape
        self.minefield = np.zeros(shape)
        # All possible coordinates+minefield tuples
        self.observation_space = list(product(*[range(d) for d in shape]))
        for coord in self.rand_gen.choice(range(len(self.observation_space)), num_mines, replace=False):
            self.minefield[self.observation_space[coord]] = 1
        self.action_space = list(np.vstack([np.eye(self.dim), -1 * np.eye(self.dim)]))
        self.mine_penalty = mine_penalty
        self.reach_reward = reach_reward
        self.rand_action_prob = rand_action_prob
        self.start_states = start
        self.end_state = np.array(end)
        self.cur_state = self.get_random_start_state()
        self.num_mines = 80
        # Need to generate the mine field here below. Figure it out later.

    def _step(self, action):
        """
        Apply given action, and return a new state and reward
        :param action:
        :return:
        """

    def _reset(self):
        pass

    def _render(self, mode='human'):
        pass

    def _seed(self, seed=42):
        """
        Returns the environment's random number genreator
        :param seed:
        :return:
        """
        return np.random.RandomState(seed)

    def get_random_start_state(self):
        """
        Returns a state selected u.a.r. from the start states
        :return:
        """
        if len(self.start_states) == 1:
            return self.start_states[0]
        return self.start_states[self.rand_gen.choice(len(self.start_states))]
