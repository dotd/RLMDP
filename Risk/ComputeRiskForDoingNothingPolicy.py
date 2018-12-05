import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from Risk.RiskMDPs import NoisyStepsMDP
from Risk.RiskUtils import ComputeRiskGeneral


shape = [5, 5]
maximal_steps = 20
states = shape[0] * shape[1]
actions = 5
gamma = 0.9
lr = 0.001
random = np.random.RandomState(0)

mdp = NoisyStepsMDP(random_generator=random, shape=shape, noise_prob=0, maximal_steps=maximal_steps)
state = mdp.reset()
