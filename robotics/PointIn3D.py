from gym import Env
import numpy as np


class PointIn3D(Env):

    def __init__(self,
                 random_generator,
                 shape=np.array([[0, 1], [0, 1], [0, 1]]),
                 mine_penalty=-40,
                 reach_reward=100,
                 step_cost=-1,
                 rand_action_prob=0.05,
                 num_mines=80,
                 start=np.array([np.array([60, 52], dtype=np.int)]),
                 terminal_states=np.array([np.array([60, 2], dtype=np.int)]),
                 ):
        pass