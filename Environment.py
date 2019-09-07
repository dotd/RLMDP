

class Environment():
    def __init__(self, mdp, agent):
        self.mdp = mdp
        self.agent = agent

    def init(self):
        # We start with environment
        self.mdp.init()

    def play_round(self, num_rounds=None):
        if num_rounds==None:
            state = self.mdp.get_state()
            action = self.agent.select_action(state)
            next_state, reward, _, _ = self.mdp.step(action)
            self.agent.update(state, action, reward, next_state)
            return state, action, next_state, reward

        res = []
        for i in range(num_rounds):
            res.append(self.play_round())
        return res


class Sim2Real_simple_model():
    def __init__(self, mdp_real, mdp_sim, agent):
        self.mdp_real = mdp_real
        self.mdp_sim = mdp_sim
        self.agent = agent

    def init(self):
        # We start with environment
        self.mdp_real.init()
        self.mdp_sim.init()

    def play_round(self, num_rounds=None):
        if num_rounds==None:
            state = self.mdp.get_state()
            action = self.agent.select_action(state)
            next_state, reward, _, _ = self.mdp.step(action)
            self.agent.update(state, action, reward, next_state)
            return state, action, next_state, reward

        res = []
        for i in range(num_rounds):
            res.append(self.play_round())
        return res


