from TwoDim.AgentBase import AgentBase
import numpy as np

class AgentRandom(AgentBase):

    def __init__(self, num_actions, random):
        """
        :param num_actions: the number of the actions in the agent's setup
        :param random: giving the numpy RandomState generator.
        """
        self.num_actions = num_actions
        self.random = random

    def choose_action(self):
        return self.random.randint(self.num_actions)
