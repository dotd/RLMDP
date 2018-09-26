import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import Categorical


"""
Based on the following Medium post.
https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
"""


class AgentPG:

    def __init__(self,
                 dim_states,
                 actions, # The actions come in the order for the indices.
                 random,
                 policy_net_class,
                 policy_net_parameters,
                 gamma,
                 lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = dim_states
        self.actions = actions
        num_actions = len(actions)
        self.random = random
        self.num_actions = len(actions)

        policy_net_parameters["dim_state"] = dim_states
        policy_net_parameters["num_actions"] = num_actions

        # network section
        # network should NOT be already instantiated. Only class pointer
        self.policy_net = policy_net_class(**policy_net_parameters).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)

        self.gamma = gamma

        ################################################################
        # Episode policy and reward history
        self.policy_history = None
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def choose_action(self, state_cur):
        '''
        Compute the best action according to the NN
        :param state_cur: np.ndarray of the state. should be in the dimension on the input
        :return: int, np.ndarray - action index (int) and action vector (np.ndarray)
        '''
        # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state_torch = torch.from_numpy(state_cur).type(torch.FloatTensor)
        state_torch = self.policy_net(Variable(state_torch))
        categorical = Categorical(state_torch)
        try:
            action_idx = categorical.sample()
        except:
            print("Problem!")
            print("state={}".format(state_cur))
            print("state_torch={}".format(state_torch))
            print("")

        if self.policy_history is not None:
            self.policy_history = torch.cat([self.policy_history, categorical.log_prob(action_idx).view(-1)])
        else:
            self.policy_history = categorical.log_prob(action_idx).view(-1)

        action = self.actions[action_idx]
        return action_idx, action, categorical

    """
    def update(self, state, action, reward, state_next, categorical):
        loss = categorical.log_prob(action) * reward
        loss.backward()
    """

    def update_policy(self):
        info = {}
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        if len(self.reward_episode[::-1]) > 1:
            # If there is only a single sample, the std is NaN. We
            info["rewards_before"] = rewards.clone()
            info["rewards_mean"] = rewards.mean()
            info["rewards_std"] = rewards.std()
            info["rewards_std0"] = rewards.std() + torch.FloatTensor([np.finfo(np.float32).eps])
            rewards = (rewards - rewards.mean()) / (rewards.std() + torch.FloatTensor([np.finfo(np.float32).eps]))
        else:
            rewards = (rewards - rewards.mean())
        info["rewards_after"] = rewards.clone()

        # Calculate loss
        loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))
        info["loss"] = loss

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.loss_history.append(loss.data[0])
        self.reward_history.append(np.sum(self.reward_episode))
        # Zeroizing episode vars!
        self.policy_history = None
        self.reward_episode = []
        return info

    def update(self, state_full, action_idx, reward, next_state_full):
        self.reward_episode.append(reward)

    def get_policy_probabilities(self, states):
        state_torch = torch.from_numpy(states).type(torch.FloatTensor)
        state_torch = self.policy_net(Variable(state_torch))
        return state_torch
