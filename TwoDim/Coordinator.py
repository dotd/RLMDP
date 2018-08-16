import numpy as np
from TwoDim.Minefield import Minefield
from TwoDim.AgentRandom import AgentRandom

class Coordinator():

    def __init__(self, mdp, agent):
        self.mdp = mdp
        self.agent = agent

    def step(self, num_step=None, debug=None):
        '''
        Step is the following:
        * get the state from environment
        * choose the action from the agent
        * get the next state
        :param num_step:
        :return:
        '''
        if num_step is not None:
            for i in range(num_step):
                self.step(num_step=None, debug = debug)
        state = self.mdp.cur_state
        action = self.agent.choose_action(state)
        next_state, reward, done, info = self.mdp._step(action)
        if debug is not None:
            print("state={}\naction={}\nnext_state={}\nreward={}".format(state, action, next_state, reward))

if __name__ == "__main__":
    mdp = Minefield()
    agent = AgentRandom(actions=mdp.action_space)
    coordinator = Coordinator(mdp, agent)
    coordinator.step(10, True)

