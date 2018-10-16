import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from collections import deque

from Risk.PureRiskMDPs import PureRiskMDP
from Risk.RiskAgent import AgentRiskPG


shape = [14, 14]
states = shape[0] * shape[1]
actions = 4
gamma = 0.9
lr = 0.00
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
episode_reward = 0
for s in range(num_steps):
    if s%100==0:
        print("s={}".format(s))
        plt.plot(episodes)
        plt.show(block=False)
        plt.pause(0.1)


    state_in = compact2full(state)
    action = agent.choose_action(state_in)
    next_state, reward, done, info = mdp.step(action)
    episode_reward += reward
    agent.update(reward)
    if done:
        agent.update_policy()
        last_rewards.append(np.sum(episode_reward))
        episode_reward=0
        if len(last_rewards) > window_len-1:
            episodes.append(np.average(last_rewards))
    state = next_state


