import numpy as np
from TwoDim.Minefield import Minefield
from TwoDim.Coordinator import Coordinator
from TwoDim.QLearning.AgentQLearning import AgentQLearning
import matplotlib.pyplot as plt
from TwoDim.TwoDimUtils import *

num_episodes = 3000
max_episode_len = 200
episode_durations = []

if __name__ == "__main__":
    reach_reward = 100
    shape = (5, 6)
    mdp = Minefield(
        shape=shape,
        num_mines=1,
        reach_reward=reach_reward,
        start=np.array([np.array([0, 0], dtype=np.int)]),
        terminal_states=np.array([np.array([shape[0]-1, shape[1]-1], dtype=np.int)]))
    agent = AgentQLearning(actions=mdp.action_space, states=None)
    shortest_path = sum(shape)-2
    best_average_reward = (reach_reward-shortest_path)/shortest_path

    for num_episode in range(num_episodes):
        print("num_episode={}".format(num_episode))
        mdp._reset()
        reward_vec = []
        for i in range(max_episode_len):
            state = mdp.cur_state
            action = agent.choose_action(state)
            next_state, reward, done, info = mdp._step(action)
            reward_vec.append(reward)
            agent.update(state, action, reward, next_state)
            if done:
                break
        average_reward = np.sum(reward_vec)/(i+1)
        episode_durations.append(average_reward)

    episode_durations_smoothed = smooth_signal(episode_durations, window_smooth_len=50)
    plt.plot(episode_durations_smoothed)
    plt.axhline(y=best_average_reward, xmin=0, xmax=num_episodes-1)
    plt.ylabel('results')
    plt.show(block=True)




