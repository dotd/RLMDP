import numpy as np
import MDPSimulator
import Utils

def get_MRP(mdp, mu):
    P = np.zeros((mdp.X,mdp.X))
    for x in range(mdp.X):
        for u in range(mdp.U):
            P[x] += mu[x,u] * mdp.P[u,x]
    R = Utils.ThreeDVec2OneDVec(mdp.R,mdp.P, mu)
    R_std = Utils.ThreeDVec2OneDVec(mdp.R_std, mdp.P, mu)
    return P, R, R_std

def get_J(P, R, gamma):
    J = np.dot(np.linalg.inv(np.identity(P.shape[0]) - gamma * P), R)
    return J


def get_J_iteratively(P, R, gamma, maximal_num_iters=10, maximal_eps=1e-3, norm = lambda x: np.linalg.norm(x)):
    X = P.shape[0]
    J = np.zeros(shape=(X,))
    J_prev = J - 10*maximal_eps*X
    iters_counter = 0
    while norm(J-J_prev) > maximal_eps and iters_counter<maximal_num_iters:
        J_prev = J
        J = R + gamma * np.dot(P,J)
        iters_counter +=1
    return J, norm(J-J_prev), iters_counter


def compute_reward_to_go(trajectory, idx_start, gamma):
    R = 0
    d = 1
    for idx in (range(idx_start,len(trajectory))):
        R += trajectory[idx][2] * d
        d *= gamma
    return R


def get_J_as_MC_raw(trajectory, gamma, X = None, func = lambda x: x):
    if X is None:
        X = max([x[0] for x in trajectory])+1
    times = np.zeros((X,))
    values = np.zeros((X,))
    for idx in range(len(trajectory)):
        x = trajectory[idx][0]
        reward2go = compute_reward_to_go(trajectory, idx, gamma)
        values[x] += func(reward2go)
        times[x] +=1
    J = values/times
    return J

def get_discount_factor_as_filter(gamma, filt_len):
    filt = np.zeros((filt_len,))
    filt[0] = 1
    for k in range(1,filt_len):
        filt[k] = filt[k-1]*gamma
    filt = np.flip(filt,0)
    return filt

def get_J_as_MC_filter(trajectory, gamma, X=None, filt_len = 40, func = lambda x: x):
    # The trajectory is x,u,r
    X = np.array([vec[0] for vec in trajectory]).max()+1 if X is None else X
    # get the reward
    r = np.array([vec[2] for vec in trajectory])
    # make the filter
    filt = get_discount_factor_as_filter(gamma, filt_len)
    res = np.convolve(r,filt)
    start_idx = filt_len-1 # 39 is the first index, meaning we removed 39 indices
    end_idx = start_idx + r.shape[0]
    res = res[start_idx:end_idx]

    # Do the stats
    times = np.zeros((X,))
    values = np.zeros((X,))
    for k in range(len(trajectory)):
        x = trajectory[k][0]
        times[x] +=1
        values[x] += func(res[k])
    J = values/times
    return J

def get_J_as_TD(trajectory, gamma, X, alpha):
    '''
    trajectory is x,u,r
    '''
    J1 = np.zeros(X)

    for k in range(len(trajectory)-1):
        x = trajectory[k][0]
        r = trajectory[k][2]
        y = trajectory[k+1][0]

        J1[x] = J1[x] + alpha/(k+1) * (r + gamma * J1[y] - J1[x])
    return J1

def get_exact_J_LSTD(phi, P, gamma, r):
    A = np.linalg.multi_dot([phi.T, np.identity(P.shape[0]) - gamma*P ,phi])
    b = np.dot(phi.T,r)
    w = np.linalg.solve(A, b)
    return w

def get_V_by_J_M(J,M):
    return M - J*J

def get_simulation_J_LSTD(phi, trajectory, gamma):
    '''
    phi should be an instance of a class where is has a method get(x) which returns vector
    '''
    A = 0
    b = 0
    for k in range(len(trajectory)-1):
        x = trajectory[k][0]
        r = trajectory[k][2]
        y = trajectory[k+1][0]
        phi_x = phi.get(x)
        A += np.outer(phi_x,phi_x - gamma * phi.get(y))
        b += phi_x * r
    w = np.linalg.solve(A, b)
    return w

def get_Q(mdp, gamma, J):
    Q = np.zeros(shape=(mdp.X, mdp.U))
    for x in range(mdp.X):
        for u in range(mdp.U):
            for y in range(mdp.X):
                Q[x,u] += mdp.P[u,x,y] * (mdp.R[u,x,y] + gamma * J[y])
    return Q

def get_J_from_maximal_Q(Q):
    J = np.amax(Q, axis=1)
    return J

def get_policy_from_Q(Q):
    mu = np.zeros_like(Q)
    idx_max = np.argmax(Q, axis=1)
    for idx_x, idx_maximal in enumerate(idx_max):
        mu[idx_x, idx_maximal] = 1
    return mu

def PI(mdp, gamma):
    mu = MDPSimulator.generate_deterministic_policy(mdp.X, mdp.U)
    mu_prev = np.zeros_like(mu)
    iter_counter = 0
    J_collector = []
    while np.array_equal(mu_prev, mu)==False:
        P, R, R_std = get_MRP(mdp, mu)
        J = get_J(P, R, gamma)
        J_collector.append(J)
        Q = get_Q(mdp, gamma, J)
        mu_prev = mu
        mu = get_policy_from_Q(Q)
        iter_counter +=1
    return mu, J_collector, Q, iter_counter

def check_J_collector_monotone(J_collector, debug_print = False, limit_Js = 10, limit_dim = 10  ):
    for i in range(len(J_collector)-1):
        J0 = J_collector[i]
        J1 = J_collector[i+1]
        delta = (J1-J0)
        deltaBoolean = delta<0
        if debug_print:
            print("J0:",J0[0:limit_dim])
            print("J1:",J1[0:limit_dim])
            print(delta[0:limit_dim])
        if deltaBoolean.any():
            return False
    return True

def VI(mdp, gamma, limit_loops=100, theta=1e-3):
    J = np.zeros(mdp.X)
    iter = 0
    delta = 0
    for iter in range(limit_loops):
        delta = 0
        for x in range(mdp.X):
            v = J[x]
            vec_u = np.zeros(mdp.U)
            for u in range(mdp.U):
                for y in range(mdp.X):
                    vec_u[u] += mdp.P[u,x,y] * (mdp.R[u,x,y] + gamma * J[y])
            # get the maximum
            J[x] = np.amax(vec_u)
            delta = max(delta, np.abs(J[x]-v))
        if (delta<theta):
            break
    Q = get_Q(mdp, gamma, J)
    mu = get_policy_from_Q(Q)
    return J, mu, iter, delta

