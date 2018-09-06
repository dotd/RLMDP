from TwoDim.DQN.AgentDQN import AgentDQN


class AgentDQNRisk(AgentDQN):

    def __init__(self,
                 dim_states,
                 actions, # The actions come in the order for the indices.
                 random,
                 policy_net_class,
                 policy_net_parameters,
                 eps_greedy,
                 gamma,
                 lr,
                 replay_memory_capacity,
                 batch_size):

        super().__init__(dim_states,
                         actions,
                         random,
                         policy_net_class,
                         policy_net_parameters,
                         eps_greedy,
                         gamma,
                         lr,
                         replay_memory_capacity,
                         batch_size)