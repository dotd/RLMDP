import numpy as np
import matplotlib.pyplot as plt

from TwoDim.Minefield import Minefield
from TwoDim.MineUtils import show_minefield
from TwoDim.PolicyGradient.AgentPolicyGradient import AgentPG
from TwoDim.TwoDimUtils import smooth_signal
from TwoDim.DQN.Nets import PG1Layer
from TwoDim.DQN.RunDQN import compact2full


def run_coordinator_pg(mdp, agent, num_episodes, max_episode_len):
    """
    :param mdp:
    :param agent:
    :param num_episodes:
    :param max_episode_len:
    :return:
    """
    episode_durations = []
    # Sum napkin computation of the shortest path without mines...
    shortest_path = sum(mdp.shape) - len(mdp.shape)
    best_average_reward = (mdp.reach_reward - shortest_path) / shortest_path

    episode_lengths = []
    for num_episode in range(num_episodes):
        if num_episode % 20 == 0:
            print("num_episode={}".format(num_episode))
            print("episode_lengths=<{}>   {}".format("NaN" if len(episode_lengths)==0 else np.mean(episode_lengths), episode_lengths))
            episode_lengths=[]
        mdp.reset()
        reward_vec = []

        # Loop over the episodes
        for i in range(max_episode_len):
            # Get the state as numpy. Making a full representation from it
            # and transform it to torch (agent is in torch)
            state: np.ndarray = mdp.cur_state
            state_full: np.ndarray = compact2full(state, mdp.shape)

            action_idx, action, categorical = agent.choose_action(state_full)
            next_state, reward, done, info = mdp.step(action)

            next_state_full = compact2full(next_state, mdp.shape) if not done else None
            reward_vec.append(reward)

            # It is a numpy array. We should have a Torch array in this implementation
            # TODO: Move PyTorch code into the agent
            agent.update(state_full, action_idx, reward, next_state_full)
            if done:
                break
        episode_lengths.append(i)

        agent.update_policy()
        average_reward = np.sum(reward_vec) / (i + 1)
        episode_durations.append(average_reward)

        if num_episode % 20 == 0:
            episode_durations_smoothed = smooth_signal(episode_durations, window_smooth_len=100)
            plt.plot(episode_durations_smoothed)
            plt.axhline(y=best_average_reward, xmin=0, xmax=num_episodes - 1)
            plt.ylabel('results')
            plt.show(block=False)
            plt.pause(0.0001)
    show_minefield(mdp, agent)


def run_minefield_pg(**kwargs):
    num_episodes = kwargs.get("num_episodes")
    max_episode_len = kwargs.get("max_episode_len")
    random_seed = kwargs.get("random_seed")
    shape = kwargs.get("shape")
    gamma = kwargs.get("gamma")
    lr = kwargs.get("lr")
    agent_class = kwargs.get("agent_class")
    policy_net_class = kwargs.get("policy_net_class")

    # The seed for reproducibility
    random = np.random.RandomState(random_seed)

    # The MDP
    mdp = Minefield(
        random_generator=random,
        shape=shape,
        num_mines=0,
        start=np.array([np.array([0, 0], dtype=np.int)]),
        terminal_states=np.array(
            [np.array([shape[0] - 1, shape[1] - 1], dtype=np.int)]))  # Terminal state in the corner

    # The Agent
    X = np.prod(shape) # state space size
    A = len(mdp.action_space) # action space size

    dqn_parameters = {"dim_state": X,
                      "num_actions": A,
                      "init_values": "zeros"}

    agent = agent_class(dim_states=X,
                        actions=mdp.action_space,
                        random=random,
                        policy_net_class=policy_net_class,
                        policy_net_parameters=dqn_parameters,
                        gamma=gamma,
                        lr=lr)

    # running it.
    run_coordinator_pg(mdp, agent, num_episodes, max_episode_len)


if __name__ == "__main__":
    run_minefield_pg(num_episodes=2000,
                     max_episode_len=200,
                     random_seed=142,
                     shape=(4, 5),
                     gamma=0.9,
                     lr=0.005,
                     batch_size=40,
                     agent_class=AgentPG,
                     policy_net_class=PG1Layer
                    )
