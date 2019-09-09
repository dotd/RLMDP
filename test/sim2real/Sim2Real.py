from MDP import MDP
import SpecificMDPs
from QAgents import SimpleQAgent
from Utils import ReplayBuffer

class AgentSim2Real(SimpleQAgent):

    def __init__(self, X, U, gamma, random, **kwargs):
        """
        U[0] is the num of MDPs
        U[1] the action space of each MDP
        """
        SimpleQAgent.__init__(self, X, U, gamma, random, **kwargs)
        er_capacity = kwargs.get("er_capacity", 100)
        self.er = ReplayBuffer(er_capacity, random)


class MDPSim2Real:
    def __init__(self, P, R, R_std, random, noise=0.1, num_mdps=2):
        self.random = random
        self.noise = noise
        self.num_mdps = num_mdps
        self.mdps = list()
        for i in range(num_mdps):
            if i==0:
                self.mdps.append(MDP(P, R, R_std))
            else:
                self.mdps.append(self.mdps[0].pertubate_P(noise))

    def reset(self, idx=None):
        if idx is None:
            for i in range(self.num_mdps):
                self.mdps[i].reset()
        else:
            self.mdps[idx].reset()

    def step(self, u):
        return self.mdps[u[0]].step(u[1])

    def show(self):
        lines = []
        for idx, mdp in enumerate(self.mdps):
            lines.append("MDP no. {}".format(idx))
            lines += mdp.show()


def generate_random_Sim2Real(X, U, B, std, random_state, noise):
    P, R, R_std = SpecificMDPs.generate_random_MDP_params(X, U, B, std, random_state)
    mdp = MDPSim2Real(P, R, R_std, random_state, noise=noise, num_mdps=2)
    return mdp

