import unittest
import torch

from TwoDim.DQN.Nets import DQN1Layer


class TestNets(unittest.TestCase):

    def test_DQN1Layer(self):
        dqn = DQN1Layer(dim_state=2, num_actions=2, init_values=True)
        print("Weights are:\n{}".format(dqn.W1.weight))
        print("Bias is:\n{}".format(dqn.W1.bias))

        dqn = DQN1Layer(dim_state=2, num_actions=2, init_values={"weight": [[0, 1], [2, 3]], "bias": [-4, -5]})
        print("Weights are:\n{}".format(dqn.W1.weight))
        print("Bias is:\n{}".format(dqn.W1.bias))

        input_vec = torch.Tensor([[0.5, 1], [1, 2]])
        output_vec = dqn.forward(input_vec)
        print("if input is:\n{}\nthen, output is:\n{}".format(input_vec, output_vec))
