import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from TwoDim.AgentBase import AgentBase
from TwoDim.TwoDimUtils import ReplayMemory
from TwoDim.TwoDimUtils import Transition

'''
Architecture of AgentDQN
'''


class AgentDQN(AgentBase):

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = dim_states
        self.actions = actions
        self.random = random
        self.num_actions = len(actions)

        # network section
        # network should NOT be already instantiated. Only class pointer
        self.policy_net = policy_net_class(**policy_net_parameters).to(self.device)
        self.target_net = policy_net_class(**policy_net_parameters).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)

        self.eps_greedy = eps_greedy
        self.gamma = gamma
        self.lr = lr
        self.replay_memory = ReplayMemory(capacity=replay_memory_capacity)
        self.batch_size = batch_size
        self.map_action_vec_to_action_idx = {}
        self.create_actions_map()
        self.loss_vec = []

    def create_actions_map(self):
        """
        Creates a mapping from action vectors to a unique index.
        For example, the action (0,-1) is mapped to index 0.
        :return: None
        """
        for action_idx, action in enumerate(self.actions):
            self.map_action_vec_to_action_idx[tuple(action)] = action_idx

    def get_best_action(self, state_cur):
        '''
        Compute the best action according to the NN
        :param state_cur: np.ndarray of the state
        :return: int, np.ndarray - action index (int) and action vector (np.ndarray)
        '''
        with torch.no_grad():
            q_values = self.policy_net(torch.Tensor([state_cur]))
            # action_selected_index = q_values.max(1)[1].view(-1, 1)
            action_idx = q_values.max(1)[1].numpy()[0]
            action = self.actions[action_idx]
            return action_idx, action

    def choose_action(self, state_cur):
        """
        Overridden from base class.
        :param state_cur:
        :return: action index, action vector
        """
        if self.random.uniform(0, 1) < self.eps_greedy:
            action_idx = self.random.randint(self.num_actions)
            action = self.actions[action_idx]
            return action_idx, action
        action_idx, action = self.get_best_action(state_cur)
        return action_idx, action

    def update(self, state, action, reward, state_next):
        """
        The update is divided into two parts:
        1) Adding to the replay memory
        2) Updating by the replay memory

        In order for the transition to be comply, we need them to be
        of the following sizes:
        state:
        action:
        reward:
        next_state:
        """

        state_full_torch = torch.Tensor([state])  # state_full
        action_torch = torch.LongTensor([[action]])  # action_idx
        reward_torch = torch.Tensor([reward])  # reward
        next_state_full_torch = torch.Tensor([state_next]) if state_next is not None else None  # next_state_full

        # We push into the replay torch.Tensors. To be comply with the pytorch tutorial on RL
        self.replay_memory.push(state_full_torch, action_torch, next_state_full_torch, reward_torch)
        self.optimize_model()

    def optimize_model(self):
        if len(self.replay_memory) < self.batch_size:
            return
        transitions = self.replay_memory.sample(self.batch_size)
        self.optimize_model_transitions(transitions)

    def optimize_model_transitions(self, transitions):
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device,
                                      dtype=torch.uint8)
        non_final_next_states_batch = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        """
        Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        columns of actions taken
        """
        # This is the
        q_values = self.policy_net(state_batch)
        maximal_q_value = q_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Init to zeros
        next_maximal_q_value = torch.zeros(self.batch_size, device=self.device)
        # Only those that are not final are different than zero!
        next_q_values = self.target_net(non_final_next_states_batch)
        # Getting the maximum and detach (same memory, different ref)
        next_maximal_q_value[non_final_mask] = next_q_values.max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_maximal_q_value * self.gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        # Compute Huber loss
        # lhs_loss is moving all the arguments of the loss to the lhs.
        # In order to maintain the loss the same, on the rhs we put zeros in the same size as lhs.
        lhs_loss = maximal_q_value - expected_state_action_values
        loss = F.smooth_l1_loss(lhs_loss, torch.zeros_like(lhs_loss))
        self.loss_vec.append(loss.detach().numpy())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def get_policy_probabilities(self, states):
        state_torch = torch.from_numpy(states).type(torch.FloatTensor)
        state_torch = self.policy_net(Variable(state_torch))
        return state_torch


if __name__ == "__main__":
    pass