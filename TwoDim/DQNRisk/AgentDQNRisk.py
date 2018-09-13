from TwoDim.DQN.AgentDQN import AgentDQN
from Filters.OnlineUtils import OnlineFilter, get_exponential_filter


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
                 batch_size,
                 risk_horizon,
                 risk_function):

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
        self.risk_horizon = risk_horizon
        self.risk_function = risk_function
        filter0 = get_exponential_filter(1.0, self.risk_horizon)
        self.online_filter = OnlineFilter(filter0)

    def update(self, state, action, reward, state_next):
        # We begin with the standard DQN
        super().update(state, action, reward, state_next)
        current_result = self.online_filter.add(reward, state)
        if current_result is not None:
            value = current_result[0]
            current_risk = self.risk_function(value)


