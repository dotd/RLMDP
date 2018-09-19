import unittest
from numpy import log

import torch

from TwoDim.DQN.Nets import PG1Layer


class TestAgentDQN(unittest.TestCase):

    def test_PG1Layer(self):
        init_values = {"weight": [[log(1), log(2), log(1)], [log(1), log(1), log(1)]],"bias": [0, 0]}
        pg = PG1Layer(dim_state=3, num_actions=2, init_values=init_values)
        print("Weights are:\n{}".format(pg.W1.weight))
        print("Bias is:\n{}".format(pg.W1.bias))

        input_vec = torch.Tensor([[2, 2, 2]])
        print(">>> Version: with torch.no_grad()")
        with torch.no_grad():
            output_vec = pg.forward(input_vec)
            print("if input is:\n{}\nthen, output is:\n{}".format(input_vec, output_vec))
        probs = pg.forward(input_vec)
        print("probs=\n{}".format(probs))



