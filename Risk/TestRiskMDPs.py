import unittest
import numpy as np

from Risk.RiskMDPs import NoisyStepsMDP


class TestNoisyStepsMDP(unittest.TestCase):

    def test_show_trajectory(self):
        random = np.random.RandomState(0)
        T = 60
        mdp = NoisyStepsMDP(random_generator=random, shape=[4,3], noise_prob=0.1, maximal_steps=30)
        for t in range(T):
            mdp.step(mdp.action_space.sample())
            mdp.render()


