import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from Risk.RiskMDPs import NoisyStepsMDP
from Risk.RiskAgent import AgentRiskPG
from Risk.RiskUtils import ComputeRiskGeneral


shape = [5, 5]
maximal_steps = 20
states = shape[0] * shape[1]
actions = 5
gamma = 0.9
lr = 0.001
random = np.random.RandomState(0)

mdp = NoisyStepsMDP(random_generator=random, shape=shape, noise_prob=0.01, maximal_steps=maximal_steps)
state = mdp.reset()
agent = AgentRiskPG(states, actions, random, gamma, lr)

last_rewards_window_len = 100
last_rewards = deque(maxlen=last_rewards_window_len)
num_steps = 40000

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

compute_risk= ComputeRiskGeneral(gamma, window_size=20, maximal_num_samples=200)
last_risks = deque(maxlen=last_rewards_window_len)
hist = np.zeros(shape=[shape[0]*2+1, shape[1]+1])

for s in range(num_steps):
    if s%1000==0 and s!=0:
        print("s={}".format(s))
        plt.clf()
        plt.subplot(2,2,1)
        plt.plot(episodes)
        plt.subplot(2,2,2)
        plt.plot(episodes_risk)
        plt.subplot(2,2,3)
        plt.imshow(hist/np.sum(hist))
        plt.colorbar()
        plt.subplot(2,2,4)
        plt.imshow(compute_risk.get_risk_map(shape))
        plt.colorbar()
        hist = np.zeros(shape=[shape[0] * 2 + 1, shape[1] + 1])
        plt.show(block=False)
        plt.pause(0.01)

    hist[state[0] + shape[0], state[1]] += 1
    state_full = compact2full(state)
    action = agent.choose_action(state_full)
    next_state, reward, done, info = mdp.step(action)
    episode_reward += reward

    compute_risk.add(tuple(state), reward, action)
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

        # Smoothing the last rewards
        if len(last_rewards) > last_rewards_window_len-1:
            # The rewards
            episodes.append(np.average(last_rewards))
            # The risks
            episodes_risk.append(np.average(last_risks))

    state = next_state


plt.show(block=True)