
from collections import deque
import numpy as np
from math import sqrt

class OnlineFilter():
    def __init__(self, filter):
        self.filter = filter[::-1]
        self.len = self.filter.shape[0] if type(self.filter) is np.ndarray else len(self.filter)
        self.deque = deque(maxlen=self.len)
        for i in range(self.len):
            self.deque.append(0)
        self.samples_counter = 0

    def add(self, sample):
        self.deque.append(sample)
        return self.get()

    def get(self):
        res = np.sum([x*y for x,y in zip(self.deque,self.filter)])
        return res

class Stats:
    def __init__(self, moments_funcs, id):
        self.id = id
        self.times = 0
        self.mean = None
        self.moments_funcs = moments_funcs
        self.values = [0 for i in range(len(moments_funcs))]

    def add(self, new_value):
        if self.times==0:
            self.mean = new_value
            for idx, func in enumerate(self.moments_funcs):
                self.values[idx] = func(new_value - self.mean)
            self.times += 1
        else:
            self.times +=1
            alpha = 1/self.times
            #alpha = 0.1
            self.mean = (1-alpha) * self.mean + alpha * new_value
            for idx, func in enumerate(self.moments_funcs):
                self.values[idx] = (1-alpha) * self.values[idx] + alpha * func(new_value - self.mean)

    def get(self):
        return [self.mean, *self.values]

class ComputeBasicStats():
    def __init__(self, X, filter, moments_funcs):
        self.X = X
        self.filter = filter[::-1]
        self.onlineFilter = OnlineFilter(filter)
        self.moments_funcs = moments_funcs
        self.stats = [Stats(moments_funcs,x) for x in range(self.X)]
        self.acum_j = []

    def add(self, x, r):
        J = self.onlineFilter.add(r)
        self.acum_j.append(J)
        self.stats[x].add(J)

    def add_vecs(self, x, r):
        for idx in range(len(x)):
            self.add(x[idx],r[idx])

    def get(self,x=None):
        if x is not None:
            return self.stats[x].get()
        else:
            return [self.stats[x].get() for x in range(self.X)]


def OnlineFilterTest():
    filter = [1,0.5,0]
    reward = [0,1,2,3,4,5,6]
    state = [0,1,0,1,0,1,0]
    of = OnlineFilter(filter)
    for i in range(len(reward)):
        res = of.add(reward[i])
        print("i={}, sample={}, filtered={}".format(i, reward[i], res))

    cbs = ComputeBasicStats(X=2, filter=filter, moments_funcs= [lambda x: x * x, lambda x: abs(x)])
    for idx in range(len(state)):
        cbs.add(state[idx], reward[idx])
        print("i={}, x={}, r={}, filtered=\n{}".format(idx, state[idx], reward[idx], cbs.get()))




#OnlineFilterTest()

#####################################################################
#### VERSION 2


'''
convention:
low indices correspond to the past
'''

class OnlineFilter2():
    def __init__(self, filter, sampler_len = 1):
        # First we flip the filter
        self.filter = filter

        # Get filter len
        self.len = self.filter.shape[0] if type(self.filter) is np.ndarray else len(self.filter)

        # implementation as deque
        self.deque_reward = deque(maxlen=self.len)
        self.deque_state = deque(maxlen=self.len)
        self.deque_result = deque(maxlen=self.len)
        self.deque_time = deque(maxlen=self.len)

        # fill-up with zeros
        for i in range(self.len):
            self.deque_reward.append(0)
            self.deque_state.append(None)
            self.deque_result.append(None)
            self.deque_time.append(None)

        self.samples_counter = 0

        # Sampler
        self.sampler = {}
        self.sampler_len = sampler_len

    def add(self, reward, state):
        self.deque_reward.append(reward)
        self.deque_state.append(state)
        res = np.sum([x * y for x, y in zip(self.deque_reward, self.filter)])
        self.deque_result.append(res)
        self.deque_time.append(self.samples_counter)
        self.samples_counter +=1
        current_result = self.get()
        if current_result is not None:
            state = current_result[2]
            sample = current_result[0]
            if state not in self.sampler:
                self.sampler[state] = deque(maxlen=self.sampler_len)
            self.sampler[state].append(sample)
        return current_result

    def get(self):
        if not self.is_valid():
            return None
        return (self.deque_result[-1], self.deque_time[0], self.deque_state[0])

    def is_valid(self):
        return self.samples_counter>=self.len


def OnlineFilterTest2():

    filter = [1,0.5,0]
    reward = [10,20,30,40,50,60,70,80,90,100,110]
    state = [0,1,2,3,2,1,0,0,1,2,3]
    state = [(s,s) for s in state]
    of = OnlineFilter2(filter, sampler_len=4)
    print("Before: deque_reward={}".format(of.deque_reward))
    print("Before: deque_state ={}".format(of.deque_state))
    print("Before: deque_result={}".format(of.deque_result))
    for i in range(len(reward)):
        valid = of.add(reward[i], state[i])
        print("{}".format(valid))
        print("{}: deque_reward={}".format(i, list(of.deque_reward)))
        print("{}: filter      ={}".format(i, of.filter))
        print("{}: deque_state ={}".format(i, list(of.deque_state)))
        print("{}: deque_time  ={}".format(i, list(of.deque_time)))
        print("{}: deque_result={}".format(i, list(of.deque_result)))
        print("\n")

    print("\n".join([str(key) + "\t" + str(value) for key,value in of.sampler.items()]))



#OnlineFilterTest2()

