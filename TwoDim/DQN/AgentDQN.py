import torch
import torch.optim as optim
import torch.nn.functional as F

from TwoDim.AgentBase import AgentBase
import numpy as np
from TwoDim.DQN.Nets import DQN1Layer

class AgentDQN(AgentBase):

    def __init__(self, dim_states, actions, random, policy_net, eps_greedy=0.1, gamma=0.5, lr=0.01):
        self.states = dim_states
        self.actions = actions
        self.random = random
        self.num_actions = len(actions)
        # network should be already instantiated
        self.policy_net = policy_net
        self.eps_greedy = eps_greedy
        self.gamma = gamma
        self.lr = lr

    def get_best_action(self, state_cur):
        with torch.no_grad():
            q_values = self.policy_net(state_cur.float())
            action_selected = q_values.max(1)[1].view(1, 1)
            q_values_maximal = q_values.gather(1, action_selected)
            return action_selected, q_values_maximal

    def choose_action(self, state_cur):
        """
        We implement epsilon greedy agent
        """
        if self.random.uniform(0, 1) < self.eps_greedy:
            return self.actions[self.random.randint(self.num_actions)]
        action_selected, _ = self.get_best_action(state_cur)
        return action_selected

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

def test_AgentDQN():
    random = np.random.RandomState(142)
    dim_states = 2
    actions = [0, 1]
    policy_net = DQN1Layer(dim_state=dim_states, num_actions=len(actions))

    dqn = AgentDQN(dim_states=dim_states, actions=actions, random=random)

if __name__ == "__main__":
    test_AgentDQN()