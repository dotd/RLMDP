__author__ = 'dot'
import numpy as np
from VarMDPs import SimpleVarMDP

class MDP:

    def __init__(self, **kwargs):
        self.debug_flag = kwargs.get('debug_flag', True)
        random_state = kwargs.get('random_state', None)
        self.init_random(random_state)
        self.gamma = kwargs.get('gamma', 0.9)

        if "P" in kwargs:
            self.P = kwargs.get('P')
            self.r = kwargs.get('R')
            self.X = self.P.shape[1]
            self.U = self.P.shape[0]
        else:
            self.X = kwargs.get('X', None)
            self.U = kwargs.get('U', None)
            self.mode = kwargs.get('mode')
            self.init_P_and_R()

        self.init_policy()
        self.debug_P_and_R()

    def init_random(self, random_state):
        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(0)

    def init_P_and_R(self):
        if self.mode=="random":
            self.init_P_and_R_random()
        elif self.mode=="1D":
            self.init_P_and_R_1D()
        elif self.mode == "test_maze":
            self.test_maze()
        elif self.mode == "SimpleVarMDP":
            self.init_P_and_R_SimpleVarMDP()
        else:
            self.init_P_and_R_random()

    def init_P_and_R_SimpleVarMDP(self):
        m = SimpleVarMDP()
        self.P = m.P
        self.r = m.r
        self.X = 4
        self.U = 2


    def init_P_and_R_random(self):
        self.P = self.random_state.uniform(low=0, high=1.0, size=(self.U, self.X, self.X))
        for u in range(self.U):
            for x in range(self.X):
                sum = np.sum(self.P[u][x])
                self.P[u][x] = self.P[u][x] / sum
        self.r = np.random.normal(size=(self.U, self.X, self.X))

    def init_P_and_R_1D(self):
        '''
        x=0, most left state
        x=X-1, most right state
        u=0, going left
        u=1, going right
        '''
        self.U = 2
        self.P = np.zeros(shape=(self.U, self.X, self.X))
        for x in range(self.X):
            for u in range(self.U):
                if u==0:
                    nxt_state = max(0,x-1)
                else:
                    nxt_state = min(x+1,self.X - 1)
                self.P[u][x][nxt_state] = 1
        self.r = np.zeros(shape=(self.U, self.X, self.X))
        self.r[:,self.X-1,:]=1

    def debug(self, str):
        if self.debug_flag:
            print(str)

    def debug_P_and_R(self):
        self.debug("P=")
        self.debug(str(self.P))
        self.debug("r=")
        self.debug(str(self.r))
        self.debug("policy_ml=")
        self.debug(str(self.policy_ml))
        self.debug("policy_dist=")
        self.debug(str(self.policy_dist))

    def init_policy(self):
        self.policy_ml = self.random_state.randint(0,self.U,size=(self.X))
        self.policy_dist = np.zeros(shape=(self.X, self.U))
        for x in range(self.X):
            self.policy_dist[x][self.policy_ml[x]]=1

    def get_MRP(self, tmp_policy=None):
        '''
        Returns the relevant MRP.
        :param tmp_policy:
        :return:
        '''

        policy = self.get_PI_PE_defaults(tmp_policy)

        # the transition matrix
        tmp_P = np.zeros(shape=(self.X,self.X))
        tmp_r = np.zeros(shape=(self.X,))
        for x in range(self.X):
            for iu, u in enumerate(tmp_policy[x]):
                # the transition matrix
                tmp_P[x] = tmp_P[x] + self.P[iu][x] * u
                # the effective reward.
                for y in range(self.X):
                    tmp_r[x] += self.P[iu][x][y]*tmp_policy[x][iu] * self.r[iu][x][y]

        return (tmp_P, tmp_r)

    def belmman_update(self, mu, P, R, curr_V, curr_state):
        V_s = 0
        for a in range(self.U):
            # no need to do the sum for policy which is 0
            if mu[curr_state][a]==0:
                continue;
            for nxt_state in range(self.X):
                V_s += P[a][curr_state][nxt_state] * \
                      (R[a][curr_state][nxt_state] + self.gamma* curr_V[nxt_state])
        return V_s

    def algebraic_PE(self, policy=None):
        policy = self.get_PI_PE_defaults(policy)

        (tmp_P,tmp_r) = self.get_MRP(policy)
        V = np.dot(np.linalg.inv(np.identity(self.X) - self.gamma * tmp_P) , tmp_r)
        return  V

    def series_PE(self, policy=None, times=10):
        policy = self.get_PI_PE_defaults(policy)

        (tmp_P, tmp_r) = self.get_MRP(policy)
        sum = np.identity(self.X)
        for t in range(times-1):
            sum = self.gamma * np.dot(sum, tmp_P) + np.identity(self.X)
        V = np.dot(sum, tmp_r)
        return V

    def iterative_PE(self, policy=None, eps_stop = 0.001, iterations_max = 1000):
        '''
        Sutton and Barto implementation
        :param tmp_policy:
        :param eps_stop:
        :param iterations_max:
        :return: the V
        '''
        policy = self.get_PI_PE_defaults(policy)

        V = self.random_state.normal(size=(self.X))
        for iter in range(iterations_max):
            delta = 0
            for s in range(self.X):
                v = V[s]
                V[s] = self.belmman_update(policy, self.P, self.r, V, s)
                delta = max(delta,abs(V[s]-v))
            if delta<eps_stop:
                break;
        return V

    def get_PI_PE_defaults(self, policy):
        if policy is None:
            policy=self.policy_dist

        return policy

    def get_policy_ml(self, policy_dist=None):
        policy_dist = self.get_PI_PE_defaults(policy_dist)
        policy_ml = np.zeros(shape=(self.X,), dtype=int)
        for x in range(self.X):
            policy_ml[x] = np.argmax(policy_dist[x])
        return policy_ml

    def get_max_action(self, x, V):
        action_value = np.zeros(shape=(self.U,))
        for a in range(self.U):
            for y in range(self.X):
                action_value[a] +=self.P[a][x][y] * (self.r[a][x][y] + self.gamma*V[y])
        max_value = action_value.max()
        # look for the max value
        maximal_indices = action_value==max_value
        indices = np.arange(self.U)
        indices = indices[maximal_indices]
        chosen_action = self.random_state.choice(indices)
        return chosen_action

    def get_policy_dist(self, policy_ml):
        policy_dist = np.zeros(shape=(self.X,self.U))
        for x in range(self.X):
            policy_dist[x][policy_ml[x]] = 1.0
        return policy_dist

    def PI_step(self, V, policy_dist=None):
        policy_dist = self.get_PI_PE_defaults(policy_dist)
        self.get_policy_ml(policy_dist)
        policy_stable = True
        policy_ml = self.get_policy_ml()
        for x in range(self.X):
            old_action = policy_ml[x]
            policy_ml[x] = self.get_max_action(x, V)
            if old_action != policy_ml[x]:
                policy_stable = False
        policy_dist = self.get_policy_dist(policy_ml)
        return policy_ml, policy_dist, policy_stable

    def compute_rho(self, policy=None, V=None):
        policy = self.get_PI_PE_defaults(policy)

        if V is None:
            V = self.iterative_PE(policy=policy)
        (tmp_P, tmp_r) = self.get_MRP(policy)
        rho = np.zeros_like(V)

        for x in range(self.X):
            second_sum = 0
            first_sum = 0
            for y in range(self.X):
                second_sum += tmp_P[x][y] * V[y]
                first_sum = tmp_P[x][y] * V[y]*V[y]
            rho[x] = first_sum - second_sum*second_sum
        return rho

    def PI(self):
        # 1) Initializtion
        # nothing to do

        policy_stable = False
        while not policy_stable:
            # Step 2 - policy evaluation
            V = self.algebraic_PE(self.policy_dist)

            # Step 3 - policy improvement
            self.policy_ml, self.policy_dist, policy_stable = self.PI_step(V, self.policy_dist)

        return


