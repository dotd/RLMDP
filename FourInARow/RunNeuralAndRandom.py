from TwoDim.DQN.AgentDQN import AgentDQN
import matplotlib.pyplot as plt
from TwoDim.TwoDimUtils import *
import numpy as np
from TwoDim.DQN.Nets import DQN1Layer
from FourInARow.AgentMCTS import AgentMCTSRandom
from FourInARow.FourInARow import FourInARow


def run_coordinator(mdp, agent, num_episodes, max_episode_len):
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
        mdp.reset()
        reward_vec = []
        for i in range(max_episode_len):
            # Get the state as numpy. Making a full representation from it
            # and transform it to torch (agent is in torch)
            state = mdp.cur_state
            state_full: np.ndarray = state[:]

            action_idx, action = agent.choose_action(state_full)
            next_state, reward, done, info = mdp.step(action)

            next_state_full = next_state[:]
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


def run_four_in_a_raw(**kwargs):
    # The seed for reproducibility
    num_episodes = kwargs.get("num_episodes")
    shape = kwargs.get("shape")
    max_episode_len = kwargs.get("max_episode_len")
    random_seed = kwargs.get("random_seed")
    eps_greedy = kwargs.get("eps_greedy")
    gamma = kwargs.get("gamma")
    lr = kwargs.get("lr")
    replay_memory_capacity = kwargs.get("replay_memory_capacity")
    batch_size = kwargs.get("batch_size")
    agent_class = kwargs.get("agent_class")
    random = np.random.RandomState(random_seed)

    # The MDP
    mdp = FourInARow(
        random_generator=random,
        shape=shape,
        win_len=3)

    # The Agent
    X = np.prod(shape) # state space size
    A = len(mdp.action_space) # action space size

    dqn_parameters = {"dim_state": X,
                      "num_actions": A,
                      "init_values": "zeros"}
    agents = []
    agents.append(agent_class[0](dim_states=X,
                        actions=mdp.action_space,
                        random=random,
                        policy_net_class=DQN1Layer,
                        policy_net_parameters=dqn_parameters,
                        eps_greedy=eps_greedy,
                        gamma=gamma,
                        lr=lr,
                        replay_memory_capacity=replay_memory_capacity,
                        batch_size=batch_size))
    agents.append(agent_class[1](dim_states=X,
                        actions=mdp.action_space,
                        random=random,
                        policy_net_class=DQN1Layer,
                        policy_net_parameters=dqn_parameters,
                        eps_greedy=eps_greedy,
                        gamma=gamma,
                        lr=lr,
                        replay_memory_capacity=replay_memory_capacity,
                        batch_size=batch_size))

    # running it.
    run_coordinator(mdp, agents, num_episodes, max_episode_len)
    plt.figure(2)
    plt.plot(agent.loss_vec)
    plt.show(block=True)


if __name__ == "__main__":
    run_four_in_a_raw(num_episodes=100,
                      max_episode_len=200,
                      random_seed=142,
                      shape=(3, 4),
                      eps_greedy=0,
                      gamma=0.5,
                      lr=0.001,
                      replay_memory_capacity=100,
                      batch_size=40,
                      agent_class=[AgentMCTSRandom, AgentMCTSRandom]
                      )
