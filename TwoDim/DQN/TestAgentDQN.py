import unittest
import numpy as np
from TwoDim.Minefield import Minefield
from TwoDim.DQN.AgentDQN import AgentDQN
from TwoDim.DQN.Nets import DQN1Layer
from TwoDim.DQN.RunDQN import compact2full
import torch


class TestAgentDQN(unittest.TestCase):

    def test_compact2full(self):
        shape = (2, 3)
        mat_1d = np.array([0, 0])
        mat_2d = np.array([[0, 0], [shape[0]-1, shape[1]-1]])
        mat_1d_full = compact2full(mat_1d, shape)
        mat_2d_full = compact2full(mat_2d, shape)
        print(mat_1d_full)
        print(mat_2d_full)

    def test_get_best_action(self, random_seed=142, shape=(2,3)):
        # The seed for reproducibility
        random = np.random.RandomState(random_seed)

        # The MDP
        mdp = Minefield(
            random_generator=random,
            shape=shape,
            num_mines=2,
            start=np.array([np.array([0, 0], dtype=np.int)]),
            terminal_states=np.array(
                [np.array([shape[0] - 1, shape[1] - 1], dtype=np.int)]))  # Terminal state in the corner

        # The Agent
        dim_state = np.prod(shape)
        num_actions = len(mdp.action_space)
        dqn_parameters = {"dim_state": dim_state,
                          "num_actions": num_actions,
                          "init_values":{"weight": np.eye(num_actions, dim_state), "bias": [0]*num_actions}}

        agent = AgentDQN(dim_states=np.prod(shape),
                         actions=mdp.action_space,
                         random=random,
                         policy_net_class=DQN1Layer,
                         policy_net_parameters=dqn_parameters)
        fake_state = np.array([1,2])
        fake_state_full = compact2full(fake_state, mdp.shape)

        # Tests of get_best_action
        fake_action_index, fake_action = agent.get_best_action(fake_state_full)
        self.assertIsInstance(fake_action, np.ndarray,
                              "AgentDQN::get_best_action is not returning 0th return value right type: {}"
                              .format(type(fake_action_index)))
        self.assertIsInstance(fake_action_index, np.int64,
                              "AgentDQN::get_best_action is not returning 1st return value right type: {}"
                              .format(type(fake_action_index)))

        # Tests of choose_action
        agent.eps_greedy = 1.001 # making sure the randomization is working
        fake_action_index, fake_action = agent.choose_action(fake_state_full)
        self.assertIsInstance(fake_action, np.ndarray,
                              "AgentDQN::choose_action is not returning 0th return value right type: {}"
                              .format(type(fake_action)))
        self.assertIsInstance(fake_action_index, int,
                              "AgentDQN::choose_action is not returning 1st return value right type: {}"
                              .format(type(fake_action_index)))

        print("network parameters:\n{}".format(agent.policy_net.W1.weight))

        print("Checking the q_value correctness")
        state_vec = np.array([[0, 0], [shape[0]-1, shape[1]-1]])
        state_vec_full = compact2full(state_vec, shape)
        with torch.no_grad():
            # compute the q-values
            q_values = agent.policy_net(torch.Tensor(state_vec_full))
            print("q_values={}".format(q_values))
            # choose the actions
            #action_selected, q_values_maximal = agent.get_best_action(state_vec)
            #print("action_selected=\n{}\nq_values_maximal=\n{}".format(action_selected, q_values_maximal))

        # Checking the gradient pipeline
        # The following lines tell what the dimensions of the tensors should be (and their types too!)
        # The state is [1, 2] vector
        # the action is 1
        # the reward is 1
        # the next state is [-1, -2]
        '''
        state_t0 = torch.Tensor([[1, 2]])
        action_t0 = torch.LongTensor([[1]])
        reward_t0 = torch.Tensor([1])
        state_t1 = torch.Tensor([[-1, -2]])
        agent.update(state_t0, action_t0, reward_t0, state_t1)
        print("network parameters2:\n{}".format(agent.policy_net.W1.weight))
        '''