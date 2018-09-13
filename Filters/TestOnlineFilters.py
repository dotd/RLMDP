import unittest
import numpy as np
from Filters.OnlineUtils import OnlineFilter


class TestOnlineFilters(unittest.TestCase):

    def test_online_filter_correctness_integer(self):
        filter_vec = [1, 0.5, 0]
        state = [0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3]
        reward = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        action = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
        sampler_len = None
        self.online_filter_correctness(filter_vec, state, reward, action, sampler_len=sampler_len)

    def test_online_filter_correctness(self):
        filter_vec = [1, 0.5, 0]
        state = [np.array([0, 0]),
                 np.array([1, 1]),
                 np.array([2, 2]),
                 np.array([3, 3]),
                 np.array([2, 2]),
                 np.array([1, 1]),
                 np.array([0, 0]),
                 np.array([0, 0]),
                 np.array([1, 1]),
                 np.array([2, 2]),
                 np.array([3, 3]),]

        action = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]

        reward = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        sampler_len = None
        self.online_filter_correctness(filter_vec, state, reward, action, sampler_len=sampler_len)

    def online_filter_correctness(self, filter_vec, state, reward, action, sampler_len):
        '''
        The filter works in the following way (explanation by example):
        Suppose we have a filter [1, 0.5, 0]
        Next, we see the rewards given by the mdp and the states:

        time:   t=0     t=1     t=2
        state:  0       1       2
        reward: 10      20      30
        filter: 1       0.5     0

        For state 0 we have value function of 10*1 + 20*0.5 + 0*30 = 20

        Convention: low indices correspond to the past

        :return:
        '''

        true_result = [20, ]
        # state = [(s, s) for s in state]
        online_filter = OnlineFilter(filter_vec, sampler_len=sampler_len)
        print("Before: deque_reward={}".format(online_filter.deque_reward))
        print("Before: deque_state ={}".format(online_filter.deque_state))
        print("Before: deque_action ={}".format(online_filter.deque_action))
        print("Before: deque_result={}".format(online_filter.deque_result))
        result_vec = []
        print()
        for i in range(len(reward)):
            valid = online_filter.add(reward[i], state[i], action[i])
            print("Add reward value {} at time {}".format(reward[i], i))
            print("valid (from add method[])={}".format(valid))
            print("t={}: filter      ={}".format(i, online_filter.filter))
            print("t={}: deque_time  ={}".format(i, list(online_filter.deque_time)))
            print("t={}: deque_state ={}".format(i, list(online_filter.deque_state)))
            print("t={}: deque_reward={}".format(i, list(online_filter.deque_reward)))
            print("t={}: deque_action={}".format(i, list(online_filter.deque_action)))
            print("t={}: deque_result_time={}".format(i, list(online_filter.deque_result_time)))
            print("t={}: deque_result={}".format(i, list(online_filter.deque_result)))
            valid = online_filter.get()
            print("valid (by get)={}".format(valid))
            print("\n")

        if online_filter.sampler is not None:
            print("\n".join([str(key) + "\t" + str(value) for key, value in online_filter.sampler.items()]))
