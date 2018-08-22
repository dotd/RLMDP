

class StateActionTable:

    def __init__(self, default_value):
        self.dict = {}
        self.default_value = default_value

    def get_value(self, state, action):
        z = (tuple(state), tuple(action))
        if z in self.dict:
            return self.dict[z]
        return self.default_value

    def set_value(self, state, action, value):
        z = (tuple(state), tuple(action))
        self.dict[z] = value

    def inc(self, state, action, value=1):
        self.set_value(state, action, self.get_value(state, action)+1)

if __name__ == "__main__":
    sat = StateActionTable(1.1)
    print(sat.get_value(state=(1, 1), action=(0, 0)))
    sat.set_value(state=(1, 1), action=(0, 0), value=2.3)
    print(sat.get_value(state=(1, 1), action=(0, 0)))
    sat.inc(state=(1, 1), action=(0, 0))
    print(sat.get_value(state=(1, 1), action=(0, 0)))
