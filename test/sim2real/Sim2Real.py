from MDP import MDP
import SpecificMDPs

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

    def step(self, idx, u):
        return self.mdps[idx].step(u)


def generate_random_Sim2Real(X, U, B, std, random_state, noise):
    P, R, R_std = SpecificMDPs.generate_random_MDP_params(X, U, B, std, random_state)
    mdp = MDPSim2Real(P, R, R_std, random_state, noise=noise, num_mdps=2)
    return MDP
