from collections import namedtuple, deque

class BasicSampler():
    def __init__(self, capacity=None, variable_names=[], rate = 1):
        self.capacity = capacity
        self.variable_names = variable_names
        self.rate = rate
        self.Transition = namedtuple('Transition', variable_names)
        if capacity is not None:
            self.deque = deque(maxlen=self.capacity)
        else:
            # limitless
            self.deque = deque()
        self.counter = 0

    def add(self, *args):
        if self.counter % self.rate==0:
            self.deque.append(self.Transition(*args))

    def get(self, debug=False):
        transitions = list(self.deque)
        mat = list(zip(*transitions))
        mat = [list(x) for x in mat]
        if debug:
            print(mat)
        return self.Transition(*mat)






