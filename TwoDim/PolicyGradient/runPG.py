import numpy as np
import matplotlib.pyplot as plt

from TwoDim.Minefield import Minefield
from TwoDim.MineUtils import show_minefield
from TwoDim.PolicyGradient.AgentPolicyGradient import AgentPG
from TwoDim.TwoDimUtils import smooth_signal
from TwoDim.DQN.Nets import PG1Layer
from TwoDim.DQN.RunDQN import compact2full
from TwoDim.Minefield import generate_standard_minefield_parameters


def run_coordinator_pg(mdp, agent, num_episodes, max_episode_len, goal_reward):
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
    best_average_reward = (goal_reward - shortest_path) / shortest_path

    episode_lengths = []
    dynamic_figure = None
    start_end_figure = None
    for num_episode in range(num_episodes):
        if num_episode % 20 == 0:
            print("num_episode={}".format(num_episode))
            print("episode_lengths=<{}>   {}".format("NaN" if len(episode_lengths)==0 else np.mean(episode_lengths), episode_lengths))
            episode_lengths=[]
        mdp.reset()
        reward_vec = []
        if start_end_figure is None:
            plt.figure()
            start_end_figure = plt.gcf().number
            plt.subplot(2, 1, 1)
            show_minefield(plt, mdp, agent)
            plt.title("Initial policy")


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

        #print("episode={}".format(num_episode))
        #print("before W={}".format(agent.policy_net.W1.weight))
        info = agent.update_policy()
        #print("after W={}".format(agent.policy_net.W1.weight))
        #print(info)
        average_reward = np.sum(reward_vec) / (i + 1)
        episode_durations.append(average_reward)

        if num_episode % 100 == 0:
            if dynamic_figure is None:
                plt.figure()
                dynamic_figure = plt.gcf().number
            plt.figure(dynamic_figure)
            episode_durations_smoothed = smooth_signal(episode_durations, window_smooth_len=100)
            plt.subplot(2,1,1)
            plt.plot(episode_durations_smoothed)
            plt.axhline(y=best_average_reward, xmin=0, xmax=num_episodes - 1)
            plt.ylabel('results')
            aa = plt.subplot(2, 1, 2)
            aa.cla()
            show_minefield(plt, mdp, agent)
            plt.title("Current policy")
            plt.show(block=False)
            plt.pause(0.1)
            #input(":::")

    # Show last time the dynamic figure
    plt.figure(dynamic_figure)
    plt.show(block=False)
    plt.pause(0.0001)
    print("Showed last time dynamic plot.")
    plt.figure(start_end_figure)
    plt.subplot(2, 1, 2)
    show_minefield(plt, mdp, agent)
    plt.title("Final policy")
    plt.show(block=False)
    plt.pause(0.0001)
    print("Showed second time start-end plot.")


def run_minefield_pg(**kwargs):
    num_episodes = kwargs.get("num_episodes")
    max_episode_len = kwargs.get("max_episode_len")
    random_seed = kwargs.get("random_seed")
    shape = kwargs.get("shape")
    gamma = kwargs.get("gamma")
    lr = kwargs.get("lr")
    agent_class = kwargs.get("agent_class")
    policy_net_class = kwargs.get("policy_net_class")
    num_mines = kwargs.get("num_mines")
    goal_reward = 100
    mine_reward = -40
    step_reward = -1
    randon_action_probability = 0.05

    # The seed for reproducibility
    random = np.random.RandomState(random_seed)

    # The MDP
    start_states, terminal_states, rewards = generate_standard_minefield_parameters(random, num_mines, shape, goal_reward, mine_reward)
    mdp = Minefield(
        random_generator=random,
        shape=shape,
        step_reward=step_reward,
        rand_action_prob=randon_action_probability,
        start_states=start_states,
        terminal_states=terminal_states,
        rewards=rewards)

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
    run_coordinator_pg(mdp, agent, num_episodes, max_episode_len, goal_reward)


if __name__ == "__main__":
    run_minefield_pg(num_episodes=2000,
                     max_episode_len=100,
                     random_seed=139,
                     shape=(5, 4),
                     num_mines=0,
                     gamma=0.8,
                     lr=0.001,
                     batch_size=40,
                     agent_class=AgentPG,
                     policy_net_class=PG1Layer
                    )
    input("Press Enter to continue...")