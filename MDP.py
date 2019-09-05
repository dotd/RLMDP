import numpy as np
import copy
import Utils


class MDP:
    '''
    P, R, R_var are variance classes of sizes U*X*X
    '''
    def __init__(self, P, R, R_std , random = np.random.RandomState(0), basis = None, info = None, init_method="random"):
        self.P = P
        self.R = R
        self.R_std = R_std
        self.U = self.P.shape[0]
        self.X = self.P.shape[1]
        self.random = random
        self.cur_state = None
        self.basis = basis
        if self.basis is not None:
            self.D = self.basis.shape[1]
        self.reset()
        self.info = info
        self.init_method = init_method

    def reset(self):
        self.cur_state = self.random.randint(self.X)
        return self.get_cur_state()

    def simulate(self, x, policy, num_samples):
        '''
        policy of size X*U

        '''
        mu = policy[x]
        trajectory = []
        for n in range(num_samples):
            u = self.random.choice(range(self.U), p=policy[x])
            y = self.random.choice(range(self.X), p=self.P[u,x])
            r = self.R[u, x, y] + self.random.normal()*self.R_std[u,x,y]
            trajectory.append([x,u,r,y])
            x = y
        return trajectory

    def init(self):
        state = self.random.choice(self.X)
        return state

    def get_state(self):
        return self.cur_state

    def step(self,u):
        x = self.cur_state
        vec = np.squeeze(self.P[u, x])
        y = self.random.choice(range(self.X), p=vec)
        r = self.R[u, x, y] + self.random.normal()*self.R_std[u,x,y]
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
        if (not np.any(self.R_std)):
            lines.append("All zero")
        else:
            lines.append(Utils.show_3dMat(self.R_std))

        lines.append(self.show_info())
        return "\n".join(lines)

    def show_info(self):
        if self.info==None:
            return "info:empty"
        lines = []
        for x in range(self.X):
            for y in range(self.X):
                for u in range(self.U):
                    if self.P[u,x,y] != 0.0:
                        x_str = self.info["states2coords"][x]
                        y_str = self.info["states2coords"][y]
                        u_str = self.info["actions"][u]
                        r_str = self.R[u, x, y]
                        lines.append("x,u,y={},{},{};\t{},{},{};\t{}".format(x,u,y,x_str,u_str,y_str,r_str))

        return "\n".join(lines)

    def show_policy(self, policy):
        '''Given a full policy, show the policy on the 2d-maze'''

        lines = []
        for x in range(self.X):
            dist = policy[x]
            u = np.argmax(dist)
            lines.append("state={},\tx,y={},\tu={} ({})".format(x,self.info["states2coords"][x], u, self.info["actions"][u]))
        return "\n".join(lines)

    def get_cur_state(self):
        if self.basis is None:
            return self.cur_state
        else:
            return self.basis[self.cur_state]

    def get_copy(self):
        mdp_new = MDP(copy.deepcopy(self.P),
                      copy.deepcopy(self.R),
                      copy.deepcopy(self.R_std),
                      random=self.random,
                      basis=copy.deepcopy(self.basis),
                      info=copy.deepcopy(self.info),
                      init_method=copy.deepcopy(self.init_method))
        return mdp_new

    def pertubate_P(self, noise):
        mdp_new = self.get_copy()
        mdp_new.P += noise*self.random.uniform(low=0, high=1.0, size=self.P.shape)
        for u in range(self.P.shape[0]):
            for x in range(self.P.shape[1]):
                if np.sum(mdp_new.P[u, x]) > 0:
                    mdp_new.P[u, x] = mdp_new.P[u, x] / np.sum(mdp_new.P[u, x])
        return mdp_new



def get_R_M2(P, R, R_std, gamma, J):
    R_M2 = R*R + R_std * R_std + 2* gamma * R * (np.dot(P,J))
    return R_M2


def get_R_V(P, R, R_std, gamma, J, moment_func):
    #Jy =  np.dot(P,J)
    #R_V = np.dot(P,moment_func(J - Jy ))  #+ moment_func(R_std)
    R_V = np.dot(P,moment_func(R + np.dot(P,J))) - moment_func(J)
    return R_V


def get_R_as_V_minus(P, R, gamma, J, moment_func = lambda x:x*x):
    MT1 = np.dot(P, moment_func(J))
    MT2 = moment_func(np.dot(P,J))
    return moment_func(gamma) * (MT1-MT2)


def get_R_as_V_def(P, R, gamma, J, moment_func = lambda x:x*x):
    return moment_func(gamma) * np.dot(P, moment_func(J - np.dot(P, J)))


def get_R_as_V_detailed(P, R, gamma, J, moment_func = lambda x:x*x):
    T1 = np.dot(P, moment_func(J))
    T2 = -2*np.dot(P,  J ) * np.dot(P, J)
    T3 = np.dot(P, np.dot(P, J)*np.dot(P, J))
    T3b = np.dot(P, J)*np.dot(P, J)
    return moment_func(gamma) * np.dot(P, moment_func(J - np.dot(P, J)))


def compute_L1_R(P, R, gamma, J):
    PJ =np.dot(P,J)
    return abs(PJ) + np.sign(PJ) * (J-PJ)


