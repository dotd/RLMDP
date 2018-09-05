from TwoDim.Minefield import Minefield
from TwoDim.DQN.AgentDQN import AgentDQN
import matplotlib.pyplot as plt
from TwoDim.TwoDimUtils import *
import numpy as np
from TwoDim.DQN.Nets import DQN1Layer
from typing import List


def compact2full(state: np.ndarray, shape: List) -> np.ndarray:

    # The single state case. state is a 1d Numpy array
    if len(state.shape) == 1:
        state_full = np.zeros(shape)
        state_full[state[0]][state[1]] = 1
        state_full = state_full.flatten()
        return state_full

    # The batch of states case. A 2d Numpy array
    outdim = np.prod(shape)
    mat = np.zeros(shape=(state.shape[0], outdim))
    for i in range(state.shape[0]):
        state_full = np.zeros(shape)
        state_full[state[i][0]][state[i][1]] = 1
        mat[i] = state_full.flatten()
    return mat


'''
APIs
The environment is dealing with tuples for states and actions
The agent is dealing *only* with Tensors.
RunDQN is responsible for translating from one to the other. 
'''


def run_main(mdp, agent, num_episodes, max_episode_len):
    """

    :param mdp:
    :param agent:
    :param num_episodes:
    :param max_episode_len:
    :return:
    """
    episode_durations = []
    shortest_path = sum(mdp.shape) - len(mdp.shape)
    best_average_reward = (mdp.reach_reward - shortest_path) / shortest_path

    for num_episode in range(num_episodes):
        print("num_episode={}".format(num_episode))
        mdp._reset()
        reward_vec = []
        for i in range(max_episode_len):
            # Get the state as numpy. Making a full representation from it
            # and transform it to torch (agent is in torch)
            state: np.ndarray = mdp.cur_state
            state_full: np.ndarray = compact2full(state, mdp.shape)

            action_idx, action = agent.choose_action(state_full)
            next_state, reward, done, info = mdp._step(action)

            next_state_full = compact2full(next_state, mdp.shape) if not done else None
            reward_vec.append(reward)

            # It is a numpy array. We should have a Torch array in this implementation
            # TODO: Move PyTorch code into the agent
            state_full_torch = torch.Tensor([state_full])
            action_torch = torch.LongTensor([[action_idx]])
            reward_torch = torch.Tensor([reward])
            next_state_full_torch = torch.Tensor([next_state_full]) if not done else None
            agent.update(state_full_torch, action_torch, reward_torch, next_state_full_torch)
            if done:
                break
        average_reward = np.sum(reward_vec) / (i + 1)
        episode_durations.append(average_reward)

        if num_episode % 20 == 0:
            episode_durations_smoothed = smooth_signal(episode_durations, window_smooth_len=50)
            plt.plot(episode_durations_smoothed)
            plt.axhline(y=best_average_reward, xmin=0, xmax=num_episodes - 1)
            plt.ylabel('results')
            plt.show(block=False)
            plt.pause(0.0001)


def run_dqn(random_seed=142, shape=(9, 10), **kwargs):
    # The seed for reproducibility
    random = np.random.RandomState(random_seed)

    # The MDP
    mdp = Minefield(
        random_generator=random,
        shape=shape,
        num_mines=5,
        start=np.array([np.array([0, 0], dtype=np.int)]),
        terminal_states=np.array(
            [np.array([shape[0] - 1, shape[1] - 1], dtype=np.int)]))  # Terminal state in the corner

    # The Agent
    X = np.prod(shape) # state space size
    A = len(mdp.action_space) # action space size

    dqn_parameters = {"dim_state": X,
                      "num_actions": A,
                      "init_values": "zeros"}


    agent = AgentDQN(dim_states=X,
                     actions=mdp.action_space,
                     random=random,
                     policy_net_class=DQN1Layer,
                     policy_net_parameters=dqn_parameters)

    # running it.
    num_episodes = kwargs.get("num_episodes")
    max_episode_len = kwargs.get("max_episode_len")
    run_main(mdp, agent, num_episodes, max_episode_len)
    plt.figure(2)
    plt.plot(agent.loss_vec)
    plt.show(block=True)


if __name__ == "__main__":
    run_dqn(num_episodes=1500, max_episode_len=200)
