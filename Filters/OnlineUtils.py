
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
            self.mean = (1-alpha) * self.mean + alpha * new_value
            for idx, func in enumerate(self.moments_funcs):
                self.values[idx] = (1-alpha) * self.values[idx] + alpha * func(new_value - self.mean)

    def get(self):
        return [self.mean, *self.values]

class ComputeBasicStats():
    def __init__(self, X, filter, moments_funcs):
        self.X = X
        self.filter = filter
        self.onlineFilter = OnlineFilter(filter)
        self.moments_funcs = moments_funcs
        self.stats = [Stats(moments_funcs,x) for x in range(self.X)]

    def add(self, x, r):
        J = self.onlineFilter.add(r)
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
    signal = [0,1,2,3,4,5, 6]
    of = OnlineFilter(filter)
    for i in range(len(signal)):
        res = of.add(signal[i])
        print("i={}, sample={}, filtered={}".format(i, signal[i], res))

    x = [ x % 3 for x in range(0,20)]
    r = [ (x % 5)*0.1 for x in range(0,20)]
    cbs = ComputeBasicStats(X=3, filter=filter, moments_funcs= [lambda x: x * x, lambda x: abs(x)])
    for idx in range(len(x)):
        cbs.add(x[idx], r[idx])
        print("i={}, x={}, r={}, filtered=\n{}".format(idx, x[idx], r[idx], cbs.get()))




#OnlineFilterTest()

