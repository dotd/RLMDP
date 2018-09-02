from TwoDim.Minefield import Minefield
from TwoDim.QLearning.AgentQLearning import AgentQLearning
import matplotlib.pyplot as plt
from TwoDim.TwoDimUtils import *
import numpy as np


def run_main(mdp, agent, num_episodes, max_episode_len):
    episode_durations = []
    shortest_path = sum(mdp.shape)-2
    best_average_reward = (mdp.reach_reward-shortest_path)/shortest_path

    for num_episode in range(num_episodes):
        print("num_episode={}".format(num_episode))
        mdp._reset()
        reward_vec = []
        for i in range(max_episode_len):
            state = tuple(mdp.cur_state)
            action = agent.choose_action(state)
            next_state, reward, done, info = mdp._step(action)
            next_state = tuple(next_state)
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


def run_q_learning(**kwargs):
    # The seed for reproducibility
    random = np.random.RandomState(142)

    # The MDP
    shape = (5, 6)
    mdp = Minefield(
        random_generator=random,
        shape=shape,
        num_mines=2,
        start=np.array([np.array([0, 0], dtype=np.int)]),
        terminal_states=np.array([np.array([shape[0]-1, shape[1]-1], dtype=np.int)])) # Terminal state in the corner

    # The Agent
    agent = AgentQLearning(actions=mdp.action_space, random=random)

    # running it.
    num_episodes = kwargs.get("num_episodes")
    max_episode_len = kwargs.get("max_episode_len")
    run_main(mdp, agent, num_episodes, max_episode_len)

if __name__ == "__main__":
    run_q_learning(num_episodes=550, max_episode_len=200)
