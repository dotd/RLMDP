import numpy as np
from TwoDim.Minefield import Minefield
from TwoDim.Coordinator import Coordinator
from TwoDim.QLearning.AgentQLearning import AgentQLearning
import matplotlib.pyplot as plt
from TwoDim.TwoDimUtils import *

num_episodes = 1000
max_episode_len = 200
episode_durations = []

if __name__ == "__main__":
    mdp = Minefield(
        shape=(6, 7),
        num_mines=1,
        start=np.array([np.array([0, 0], dtype=np.int)]),
        terminal_states=np.array([np.array([5, 6], dtype=np.int)]))
    agent = AgentQLearning(actions=mdp.action_space, states=None)

    for num_episode in range(num_episodes):
        print("num_episode={}".format(num_episode))
        mdp._reset()
        for i in range(max_episode_len):
            state = mdp.cur_state
            action = agent.choose_action(state)
            next_state, reward, done, info = mdp._step(action)
            agent.update(state, action, reward, next_state)
            if done:
                break
        episode_durations.append(i)

    episode_durations_smoothed = smooth_signal(episode_durations, window_smooth_len=5)
    plt.plot(episode_durations_smoothed)
    plt.ylabel('results')
    plt.show(block=True)




