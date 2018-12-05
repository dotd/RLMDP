import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import Categorical

from Risk.RiskUtils import ComputeRiskGeneral


class PG1Layer(nn.Module):
    """
    Single Layer Network for the Policy Gradient
    """
    def __init__(self, dim_state, num_actions):
        super(PG1Layer, self).__init__()
        self.dim_state = dim_state.item() if isinstance(dim_state, np.ndarray) else dim_state
        self.W1 = nn.Linear(self.dim_state, out_features=num_actions)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.W1,
            nn.Softmax(dim=-1)
        )
        return model(x)


class RiskEstimatorNet(nn.Module):

    def __init__(self, dim_state, hidden_state):
        super(RiskEstimatorNet, self).__init__()
        self.dim_state = dim_state.item() if isinstance(dim_state, np.ndarray) else dim_state
        self.hidden_state = hidden_state.item() if isinstance(hidden_state, np.ndarray) else hidden_state
        self.W1 = nn.Linear(self.dim_state, out_features=hidden_state)
        self.W2 = nn.Linear(self.hidden_state, out_features=1)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.W1,
            nn.ReLU,
            self.W2,
            nn.Sigmoid,
        )
        return model(x)


class AgentRiskPG:

    def __init__(self,
                 states,
                 actions, # The actions come in the order for the indices.
                 random,
                 gamma,
                 lr,
                 alpha=0,
                 f=lambda x:x*x):
        self.device = torch.device("cpu")
        self.states = states
        self.actions = actions
        self.random = random

        # network section
        self.policy_net = PG1Layer(self.states, self.actions).to(self.device)
        self.optimizer_policy = optim.SGD(self.policy_net.parameters(), lr=lr)

        self.risk_net = RiskEstimatorNet(self.states, self.states).to(self.device)
        self.optimizer_risk = optim.SGD(self.risk_net.parameters(), lr=lr)

        self.gamma = gamma
        self.compute_risk = ComputeRiskGeneral(gamma, window_size=20, maximal_num_samples=20)

        ################################################################
        # Episode policy and reward history
        self.log_policy_history = None
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        # Risk
        self.risk_history = []

    def choose_action(self, state):
        state_torch = torch.from_numpy(state).type(torch.FloatTensor).to(self.device)
        action_vec = self.policy_net(Variable(state_torch))
        categorical = Categorical(action_vec)
        action = categorical.sample()

        if self.log_policy_history is not None:
            self.log_policy_history = torch.cat([self.log_policy_history, categorical.log_prob(action).view(-1)])
        else:
            self.log_policy_history = categorical.log_prob(action).view(-1)

        return action

    def update(self, reward, state=None, risk=None):
        self.reward_episode.append(reward)
        if state is not None:
            self.risk_history.append(risk)

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
        loss = torch.sum(torch.mul(self.log_policy_history, Variable(rewards).to(self.device)).mul(-1), -1)
        info["loss"] = loss

        # Update network weights
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        # Save and intialize episode history counters
        self.loss_history.append(loss.data[0].item())
        self.reward_history.append(np.sum(self.reward_episode))
        # Zeroizing episode vars!
        self.log_policy_history = None
        self.reward_episode = []
        return info


    def get_policy_probabilities(self, states):
        state_torch = torch.from_numpy(states).type(torch.FloatTensor)
        prob_vec = self.policy_net(Variable(state_torch))
        return prob_vec
