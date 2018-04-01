import numpy as np
from Utils import *
import math
import Policies
import MDPSimulator

def generate_investment_sim(p_noise = 0, **kwargs):
    P = np.array([[[1 - p_noise, p_noise], [1 - p_noise, p_noise]], [[p_noise, 1 - p_noise], [p_noise, 1 - p_noise]]])
    R1 = kwargs.get("R1", 2)
    R1_std = kwargs.get("R1_std", math.sqrt(2))
    R1D = np.array([1, R1])
    R1D_std = np.array([0, R1_std])
    R = OneDVec2ThreeDVec(R1D,U=2)
    R_std = OneDVec2ThreeDVec(R1D_std,U=2)

    sparse_flag = kwargs.get("sparse_flag", False)
    if sparse_flag:
        pass
    else:
        mdp = MDPSimulator.MDPSim(P = P, R = R, R_std=R_std)
    return mdp

def func1():
    mdp = generate_investment_sim()
    policy = Policies.generate_uniform_policy(mdp.X,mdp.U)
    print("the mdp:")
    print(mdp.show())
    print("the policy:")
    print(show_2dMat(policy))
    trajectory = mdp.simulate(x=0,policy=policy,num_samples=10)
    print(trajectory)

def generate_random_MDP(X, U, B, R_sparse, std = 0, random_state = np.random.RandomState(0), basis = None):
    P = np.zeros(shape=(U,X,X))
    R = np.zeros(shape=(U,X,X))
    R_std = std*np.ones(shape=(U,X,X))

    for u in range(U):
        for x in range(X):
            P[u, x] = get_random_sparse_vector(X, B, True, "uniform", random_state)
            R[u, x] = get_random_sparse_vector(X, R_sparse, False, "gaussian", random_state)
            R[u,x,:]  = R[u,x,0]
        if u>=1:
            R[u] = R[0]

    if basis is not None:
        basis = random_state.normal(size=(X, basis))
    mdp = MDPSimulator.MDPSim(P = P, R = R, R_std=R_std, basis = basis)
    return mdp

def func2():
    mdp = generate_random_MDP(X=5,U=3,B=2,R_sparse=1)
    print(mdp.show())
