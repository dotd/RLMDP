from TwoDim.DQN.RunDQN import run_minefield
#from TwoDim.DQN.AgentDQN import AgentDQN
from TwoDim.DQNRisk.AgentDQNRisk import AgentDQNRisk


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
