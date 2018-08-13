from gym import Env
import numpy as np


class Minefield(Env):
    """
    Perhaps even inherit from GoalEnv?
    """
    def __init__(self, shape=[60, 60]):
        self.action_space = None
        self.observation_space = None

    def _step(self, action):
        pass

    def _reset(self):
        pass

    def _render(self, mode='human'):
        pass

    def _seed(self, seed=None):
        pass
