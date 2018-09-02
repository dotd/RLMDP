from TwoDim.AgentBase import AgentBase
import numpy as np
from TwoDim.QLearning.StateActionTable import StateActionTable


class AgentQLearning(AgentBase):

    def __init__(self, states, actions, random, default_q_value=0, eps_greedy = 0.01, gamma = 0.5, lr = 0.01):
        """
        :param actions: the actions of the actions in the agent's setup
        :param random: giving the numpy RandomState generator.
        """
        self.actions = actions
        self.states = states
        self.random = random
        self.num_actions = len(actions)
        self.q_table = StateActionTable(default_value=default_q_value)
        self.visits = StateActionTable(default_value=0)
        self.eps_greedy = eps_greedy
        self.gamma = gamma
        self.lr = lr

    def get_best_action(self, state_cur, eps = 0):
        q_value = [self.q_table.get_value(state_cur, action) for action in self.actions]
        # get the maximum value
        max_value = max(q_value)
        idx = np.where(np.array(q_value) >= max_value - eps)
        if len(idx[0]) == 1:
            return self.actions[int(idx[0])]
        return self.actions[self.random.choice(idx[0])]

    def choose_action(self, state):
        """
        We implement epsilon greedy agent
        """
        if self.random.uniform(0,1) < self.eps_greedy:
            return self.actions[self.random.randint(self.num_actions)]
        return self.get_best_action(state)

    def update(self, state, action, reward, state_next):
        """
        For a random agent we do nothing for updating.
        """
        # Updating the q-value
        action_next = self.get_best_action(state_next)
        q_cur = self.q_table.get_value(state, action)
        q_next = self.q_table.get_value(state_next, action_next)
        td = reward + self.gamma * q_next - q_cur
        updated_value = q_cur + self.lr * td
        self.q_table.set_value(state, action, updated_value)

        # Updating the visits
        self.visits.inc(state, action)


if __name__ == "__main__":
    aql = AgentQLearning(states=list(range(10)), actions = [(0, 1), (0, -1), (1, 0), (-1, 0)])
    state = 1
    aql.q_table.set_value(state, (0, 1), 1)
    print("Get best action 1:")
    for i in range(10):
        print("i={}, best_action={}".format(i, aql.get_best_action(state)))
    aql.q_table.set_value(state, (1, 0), 1)
    print("-----------------")
    for i in range(10):
        print("i={}, best_action={}".format(i, aql.get_best_action(state)))
