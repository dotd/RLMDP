import unittest
import numpy as np
from TwoDim.Minefield import generate_standard_minefield_parameters


class TestGenerateStandardMinefield(unittest.TestCase):

    def test_generate(self):
        rand_gen = np.random.RandomState(0)
        num_mines = 2
        shape = (5, 4)
        goal_reward = 100
        mine_reward = -40

        start_states, terminal_states, rewards = \
            generate_standard_minefield_parameters(rand_gen, num_mines, shape, goal_reward, mine_reward)
        print("start_states=\n{}\nterminal_states=\n{}\nreward=\n{}".format(start_states, terminal_states, rewards))