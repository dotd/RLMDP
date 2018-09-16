import numpy as np
import matplotlib.pyplot as plt

from TwoDim.DQN.RunDQN import run_minefield, run_coordinator
from TwoDim.DQN.Nets import DQN1Layer
from TwoDim.DQNRisk.AgentDQNRisk import AgentDQNRisk
from TwoDim.Minefield import Minefield


def run_minefield_risk(**kwargs):
    # The seed for reproducibility
    num_episodes = kwargs.get("num_episodes")
    max_episode_len = kwargs.get("max_episode_len")
    random_seed = kwargs.get("random_seed")
    shape = kwargs.get("shape")
    eps_greedy = kwargs.get("eps_greedy")
    gamma = kwargs.get("gamma")
    lr = kwargs.get("lr")
    replay_memory_capacity = kwargs.get("replay_memory_capacity")
    batch_size = kwargs.get("batch_size")
    agent_class = kwargs.get("agent_class")
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

    agent = agent_class(dim_states=X,
                        actions=mdp.action_space,
                        random=random,
                        policy_net_class=DQN1Layer,
                        policy_net_parameters=dqn_parameters,
                        eps_greedy=eps_greedy,
                        gamma=gamma,
                        lr=lr,
                        replay_memory_capacity=replay_memory_capacity,
                        batch_size=batch_size)

    # running it.
    run_coordinator(mdp, agent, num_episodes, max_episode_len)
    plt.figure(2)
    plt.plot(agent.loss_vec)
    plt.show(block=True)


if __name__ == "__main__":
    run_minefield(num_episodes=500,
                  max_episode_len=200,
                  random_seed=142,
                  shape=(9, 10),
                  eps_greedy=0,
                  gamma=0.5,
                  lr=0.001,
                  replay_memory_capacity=100,
                  batch_size=40,
                  agent_class=AgentDQNRisk
                  )
