

class StateActionTable:

    def __init__(self, default_value):
        self.dict = {}
        self.default_value = default_value

    def get_value(self, state, action):
        z = (state, action)
        if z in self.dict:
            return self.dict[z]
        return self.default_value

    def set_value(self, state, action, value):
        z = (state, action)
        self.dict[z] = value


if __name__ == "__main__":
    sat = StateActionTable(1.1)
    print(sat.get_value(state=(1, 1), action=(0, 0)))
    sat.set_value(state=(1, 1), action=(0, 0), value=2.3)
    print(sat.get_value(state=(1, 1), action=(0, 0)))
