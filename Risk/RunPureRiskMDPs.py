import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from collections import deque

from Risk.RiskMDPs import PureRiskMDP
from Risk.RiskAgent import AgentRiskPG
from Risk.RiskUtils import ComputeRiskGeneral


shape = [24, 24]
states = shape[0] * shape[1]
actions = 5
gamma = 0.9
lr = 0.001
random = np.random.RandomState(0)

mdp = PureRiskMDP(random_generator=random, shape=shape, noise_prob=0.01)
state = mdp.reset()
agent = AgentRiskPG(states, actions, random, gamma, lr)

window_len = 100
last_rewards = deque(maxlen=window_len)
num_steps = 100000

def compact2full(state):
    out = np.zeros(shape=shape)
    out[state[0], state[1]] = 1
    out = out.reshape(-1)
    return out


# loop over the steps
episodes = []
episodes_risk = []
episode_reward = 0
episode_risk = 0

compute_risk= ComputeRiskGeneral(gamma, window_size=20, maximal_num_samples=20)
last_risks = deque(maxlen=window_len)

for s in range(num_steps):
    if s%1000==0:
        print("s={}".format(s))
        plt.subplot(2,1,1)
        plt.plot(episodes)
        plt.subplot(2,1,2)
        plt.plot(episodes_risk)
        plt.show(block=False)
        plt.pause(0.01)

    state_in = compact2full(state)
    action = agent.choose_action(state_in)
    next_state, reward, done, info = mdp.step(action)
    episode_reward += reward

    compute_risk.add(tuple(state), reward)
    risk = compute_risk.compute_var(tuple(state))
    if risk is not None:
        episode_risk += risk

    agent.update(reward)
    if done:
        agent.update_policy()
        last_rewards.append(np.sum(episode_reward))
        last_risks.append(np.sum(episode_risk))
        episode_reward = 0
        episode_risk = 0
        if len(last_rewards) > window_len-1:
            episodes.append(np.average(last_rewards))
            episodes_risk.append(np.average(last_risks))

    state = next_state


