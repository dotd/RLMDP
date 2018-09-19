import unittest
import torch

from TwoDim.DQN.Nets import DQN1Layer
from TwoDim.DQN.Nets import PG1Layer


class TestNets(unittest.TestCase):

    def test_DQN1Layer(self):
        print("test_DQN1Layer")
        print("Network 1 - Random")
        print("==================")
        dqn = DQN1Layer(dim_state=3, num_actions=2, init_values=None)
        print("Weights are:\n{}".format(dqn.W1.weight))
        print("Bias is:\n{}".format(dqn.W1.bias))

        print("\n\n\n")

        print(">>> Network 2 - Non-Random with specific numbers")
        print(">>> ============================================")
        dqn = DQN1Layer(dim_state=3,
                        num_actions=2,
                        init_values={"weight": [[0, 1, 2], [3, 4, 5]],
                                     "bias": [-5, -4]},
                        output_activation=None)
        print(">>> Note!: There is no activation")
        print("Weights are:\n{}".format(dqn.W1.weight))
        print("Bias is:\n{}".format(dqn.W1.bias))

        input_vec = torch.Tensor([[0, 1, 2], [1, 1, 1]])
        print(">>> Version: with torch.no_grad()")
        with torch.no_grad():
            output_vec = dqn.forward(input_vec)
            print("if input is:\n{}\nthen, output is:\n{}".format(input_vec, output_vec))
        print(">>> Version: withOUT torch.no_grad()")
        output_vec = dqn.forward(input_vec)
        print("if input is:\n{}\nthen, output is:\n{}".format(input_vec, output_vec))

        print("Network 3 - Non-Random with all zeros")
        print("============================================")
        dqn = DQN1Layer(dim_state=3, num_actions=2, init_values="zeros")
        print("Weights are:\n{}".format(dqn.W1.weight))
        print("Bias is:\n{}".format(dqn.W1.bias))

        output_vec = dqn.forward(input_vec)
        print("if input is:\n{}\nthen, output is:\n{}".format(input_vec, output_vec))


