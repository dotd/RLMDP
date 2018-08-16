from TwoDim.AgentBase import AgentBase
import numpy as np


class AgentRandom(AgentBase):

    def __init__(self, actions, random = np.random.RandomState(142)):
        """
        :param actions: the actions of the actions in the agent's setup
        :param random: giving the numpy RandomState generator.
        """
        self.actions = actions
        self.random = random
        self.num_actions = len(actions)

    def choose_action(self,state):
        '''
        We choose action randomly
        '''
        return self.actions[self.random.randint(self.num_actions)]

    def update(self, state, action, reward, state_next):
        '''
        For a random agent we do nothing for updating.
        '''
        pass