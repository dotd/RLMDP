import matplotlib.pyplot as plt
import numpy as np

from TwoDim.Minefield import Minefield
from TwoDim.Minefield import generate_standard_minefield_parameters
from TwoDim.QLearning.AgentQLearning import AgentQLearning
from TwoDim.TwoDimUtils import smooth_signal


def run_main(mdp, agent, num_episodes, max_episode_len, reach_reward):
    episode_durations = []
    shortest_path = sum(mdp.shape)-2
    best_average_reward = (reach_reward-shortest_path)/shortest_path

    for num_episode in range(num_episodes):
        print("num_episode={}".format(num_episode))
        mdp.reset()
        reward_vec = []
        for i in range(max_episode_len):
            state = mdp.cur_state
            # Action here is a tuple, meaning, the vector action of the environment
            action = agent.choose_action(state)
            next_state, reward, done, info = mdp.step(action)
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
    num_episodes = kwargs.get("num_episodes")
    max_episode_len = kwargs.get("max_episode_len")
    random_seed = kwargs.get("random_seed")
    shape = kwargs.get("shape")
    eps_greedy = kwargs.get("eps_greedy")
    gamma = kwargs.get("gamma")
    lr = kwargs.get("lr")
    default_q_value = kwargs.get("default_q_value")
    agent_class = kwargs.get("agent_class")
    random = np.random.RandomState(random_seed)
    num_mines = kwargs.get("num_mines")
    goal_reward = 100
    mine_reward = -40
    step_reward = -1
    random_action_probability = 0.05

    # The MDP
    start_states, terminal_states, rewards = generate_standard_minefield_parameters(random, num_mines, shape,
                                                                                    goal_reward, mine_reward)
    mdp = Minefield(
        random_generator=random,
        shape=shape,
        step_reward=step_reward,
        rand_action_prob=random_action_probability,
        start_states=start_states,
        terminal_states=terminal_states,
        rewards=rewards)
    # The Agent
    agent = agent_class(actions=mdp.action_space,
                        random=random,
                        eps_greedy=eps_greedy,
                        gamma=gamma,
                        default_q_value=default_q_value,
                        lr=lr)

    # running it.
    run_main(mdp, agent, num_episodes, max_episode_len, goal_reward)


if __name__ == "__main__":
    run_q_learning(num_episodes=2000,
                   max_episode_len=200,
                   random_seed=142,
                   shape=(9, 10),
                   eps_greedy=0.01,
                   gamma=0.9,
                   num_mines=0,
                   lr=0.01,
                   agent_class=AgentQLearning,
                   default_q_value=0
                   )
