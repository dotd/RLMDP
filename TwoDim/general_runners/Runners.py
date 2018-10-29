import numpy as np
import matplotlib.pyplot as plt

from TwoDim.Minefield import Minefield
from TwoDim.MineUtils import show_minefield
from TwoDim.PolicyGradient.AgentPolicyGradient import AgentPG
from TwoDim.DQN.AgentDQN import AgentDQN
from TwoDim.TwoDimUtils import smooth_signal
import TwoDim.DQN.Nets as Nets
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

            action_idx, action = agent.choose_action(state_full)
            next_state, reward, done, info = mdp.step(action)

            next_state_full = compact2full(next_state, mdp.shape) if not done else None
            reward_vec.append(reward)

            # It is a numpy array. We should have a Torch array in this implementation
            # TODO: Move PyTorch code into the agent
            agent.update(state_full, action_idx, reward, next_state_full)
            if done:
                break
        episode_lengths.append(i)

        info = agent.update_policy()
        average_reward = np.sum(reward_vec) / (i + 1)
        episode_durations.append(average_reward)

        if num_episode % 100 == 0:
            if dynamic_figure is None:
                plt.figure()
                dynamic_figure = plt.gcf().number
            plt.figure(dynamic_figure)
            episode_durations_smoothed = smooth_signal(episode_durations, window_smooth_len=100)
            aa = plt.subplot(2, 1, 1)
            aa.cla()
            plt.plot(episode_durations_smoothed)
            plt.axhline(y=best_average_reward, xmin=0, xmax=num_episodes - 1)
            plt.ylabel('results')
            aa = plt.subplot(2, 1, 2)
            aa.cla()
            show_minefield(plt, mdp, agent)
            plt.title("Current policy")
            plt.show(block=False)
            plt.pause(0.1)

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


def run_simulation2(simulation_params, environment_params, agent_params):
    # Simulation
    num_episodes = simulation_params["num_episodes"]
    max_episode_len = simulation_params["max_episode_len"]
    random_seed_sim = simulation_params["random_seed_sim"]
    random_sim = np.random.RandomState(random_seed_sim)

    # Environment
    shape = environment_params["shape"]
    goal_reward = environment_params["goal_reward"]
    mine_reward = environment_params["mine_reward"]
    step_reward = environment_params["step_reward"]
    random_action_probability = environment_params["random_action_probability"]
    num_mines = environment_params["num_mines"]

    # Agent
    random_seed_agent = agent_params["random_seed_agent"]
    random_agent = np.random.RandomState(random_seed_agent)
    gamma = agent_params["gamma"]
    lr = agent_params["lr"]
    agent_class = agent_params["agent_class"]
    policy_net_class = agent_params["policy_net_class"]
    policy_net_parameters = agent_params["policy_net_parameters"]

    # The seed for reproducibility

    # The MDP
    start_states, terminal_states, rewards = generate_standard_minefield_parameters(random_sim, num_mines, shape,
                                                                                    goal_reward, mine_reward)
    mdp = Minefield(
        random_generator=random_sim,
        shape=shape,
        step_reward=step_reward,
        rand_action_prob=random_action_probability,
        start_states=start_states,
        terminal_states=terminal_states,
        rewards=rewards)

    # The Agent
    X = np.prod(shape) # state space size
    A = len(mdp.action_space) # action space size

    agent = agent_class(dim_states=X,
                        actions=mdp.action_space,
                        random=random_agent,
                        policy_net_class=policy_net_class,
                        policy_net_parameters=policy_net_parameters,
                        gamma=gamma,
                        lr=lr)

    # running it.
    run_coordinator_pg(mdp, agent, num_episodes, max_episode_len, goal_reward)


def run_simulation(simulation_params, environment_params, agent_params):
    # Simulation
    num_episodes = simulation_params["num_episodes"]
    max_episode_len = simulation_params["max_episode_len"]
    random_seed_sim = simulation_params["random_seed_sim"]
    random_sim = np.random.RandomState(random_seed_sim)

    # Environment
    shape = environment_params["shape"]
    goal_reward = environment_params["goal_reward"]
    mine_reward = environment_params["mine_reward"]
    step_reward = environment_params["step_reward"]
    random_action_probability = environment_params["random_action_probability"]
    num_mines = environment_params["num_mines"]

    # Agent
    """
    random_seed_agent = agent_params["random_seed_agent"]
    random_generator_agent = np.random.RandomState(random_seed_agent)
    gamma = agent_params["gamma"]
    lr = agent_params["lr"]
    agent_class = agent_params["agent_class"]
    policy_net_class = agent_params["policy_net_class"]
    policy_net_parameters = agent_params["policy_net_parameters"]
    """

    # The seed for reproducibility

    # The MDP
    start_states, terminal_states, rewards = generate_standard_minefield_parameters(random_sim, num_mines, shape,
                                                                                    goal_reward, mine_reward)
    mdp = Minefield(
        random_generator=random_sim,
        shape=shape,
        step_reward=step_reward,
        rand_action_prob=random_action_probability,
        start_states=start_states,
        terminal_states=terminal_states,
        rewards=rewards)

    # The Agent
    X = np.prod(shape) # state space size
    #A = len(mdp.action_space) # action space size

    agent_class = agent_params["agent_class"]
    agent_params.pop("agent_class")
    agent_params["dim_states"] = X
    agent_params["actions"] = mdp.action_space
    agent = agent_class(**agent_params)

    # running it.
    run_coordinator_pg(mdp, agent, num_episodes, max_episode_len, goal_reward)


def run_main():
    simulation_params = dict()
    simulation_params["num_episodes"] = 5000
    simulation_params["max_episode_len"] = 100
    simulation_params["random_seed_sim"] = 139

    environment_params = dict()
    environment_params["shape"] = (5, 4)
    environment_params["num_mines"] = 0
    environment_params["goal_reward"] = 100
    environment_params["mine_reward"] = -40
    environment_params["step_reward"] = -1
    environment_params["random_action_probability"] = 0.05

    agent_params = dict()
    agent_params["gamma"] = 0.2
    agent_params["random"] = np.random.RandomState(209)

    agent_type = "dqn"
    if agent_type=="pg":
        print("PG running")
        agent_params["agent_class"] = AgentPG
        agent_params["policy_net_class"] = Nets.PG1Layer
        agent_params["policy_net_parameters"] = {"init_values": "zeros"}
        agent_params["lr"] = 0.01
    else:
        print("DQN running")
        agent_params["agent_class"] = AgentDQN
        agent_params["policy_net_class"] = Nets.DQN2Layers
        agent_params["policy_net_parameters"] = {"intermediate": 10}
        agent_params["eps_greedy"] = 0.1
        agent_params["replay_memory_capacity"] = 1000
        agent_params["batch_size"] = 50
        agent_params["lr"] = 0.00001

    run_simulation(simulation_params, environment_params, agent_params)
    input("Press Enter to continue...")


if __name__ == "__main__":
    run_main()