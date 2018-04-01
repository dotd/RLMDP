import numpy as np
import math
import Utils


class MDPSim:
    '''
    P, R, R_var are variance classes of sizes U*X*X
    '''
    def __init__(self, P, R, R_std = 0, random = np.random.RandomState(0), basis = None, info = {}):
        self.P = P
        self.R = R
        self.R_std = R_std
        self.U = self.P.shape[0]
        self.X = self.P.shape[1]
        self.rand = random
        self.cur_state = None
        self.basis = basis
        if self.basis is not None:
            self.D = self.basis.shape[1]
        self.reset()
        self.info = info

    def reset(self):
        self.cur_state = self.rand.randint(self.X)
        return self.get_cur_state()

    def simulate(self, x, policy, num_samples):
        '''
        policy of size X*U

        '''
        mu = policy[x]
        trajectory = []
        for n in range(num_samples):
            u = self.rand.choice(range(self.U), p=policy[x])
            y = self.rand.choice(range(self.X), p=self.P[u,x])
            r = self.R[u, x, y] + self.rand.normal()*self.R_std[u,x,y]
            trajectory.append([x,u,r])
            x = y
        return trajectory

    def step(self,u):
        x = self.cur_state
        y = self.rand.choice(range(self.X), p=self.P[u, x])
        r = self.R[u, x, y] + self.rand.normal()*self.R_std[u,x,y]
        self.cur_state = y
        if self.basis is None:
            # Gym format
            return y,r, None, None
        else:
            return self.basis[y],r, None, None

    def show(self):
        lines = []
        lines.append("P=")
        lines.append(Utils.show_3dMat(self.P))
        lines.append("R=")
        lines.append(Utils.show_3dMat(self.R))
        lines.append("R_std=")
        lines.append(Utils.show_3dMat(self.R_std))
        return "\n".join(lines)

    def get_cur_state(self):
        if self.basis is None:
            return self.cur_state
        else:
            return self.basis[self.cur_state]

def get_R_M2(P, R, R_std, gamma, J):
    R_M2 = R*R + R_std * R_std + 2* gamma * R * (np.dot(P,J))
    return R_M2

def get_R_V(P, R, R_std, gamma, J):
    Jy = np.dot(P,J)
    R_V = gamma*gamma * ( np.dot(P,J*J) - Jy * Jy) + R_std*R_std
    return R_V




