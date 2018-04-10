import numpy as np
from Utils import *
import math
import Policies
import MDPSimulator
import collections

def generate_investment_sim(p_noise = 0, **kwargs):
    P = np.array([[[1 - p_noise, p_noise], [1 - p_noise, p_noise]], [[p_noise, 1 - p_noise], [p_noise, 1 - p_noise]]])
    R_state_1 = kwargs.get("R_state_1", 2)
    R1_std = kwargs.get("R1_std", math.sqrt(2))
    R1D = np.array([1, R_state_1])
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

def check_generate_random_MDP():
    mdp = generate_random_MDP(X=5,U=3,B=2,R_sparse=1)
    print(mdp.show())

def generate_clean_2d_maze(x_size=4, y_size = 3, reward_coord = (3,2), start_method="random"):
    actions = {0:"increase_y", 1:"increase_x", 2:"decrease_y", 3:"decrease_x"}
    coords2states = {}
    states2coords = {}
    map = dict()
    states_cnt = 0
    for x in range(x_size):
        for y in range(y_size):
            coords2states[(x,y)] = states_cnt
            states2coords[states_cnt] = (x,y)
            states_cnt+=1

    X = len(coords2states)
    U = len(actions)

    P = np.zeros(shape=(U,X,X))
    r = np.zeros(shape=(U,X,X))
    # This loops based on the state and action tells what are the next_state
    for x_origin in range(x_size):
        for y_origin in range(y_size):
            for u,u_str in actions.items():
                x_target = x_origin
                y_target = y_origin
                # doing the dynamics
                if u_str=="increase_y" and y_origin<y_size-1:
                    y_target = y_origin+1
                if u_str=="increase_x" and x_origin<x_size-1:
                    x_target = x_origin+1
                if u_str=="decrease_y" and y_origin>0:
                    y_target = y_origin-1
                if u_str=="decrease_x" and x_origin>0:
                    x_target = x_origin-1

                state_origin = coords2states[(x_origin,y_origin)]
                state_target = coords2states[(x_target,y_target)]

                P[u,state_origin,state_target] = 1.0
                if (x_origin,y_origin)==reward_coord:
                    r[u,state_origin,state_target] = 1.0
    # If we reached the reward, we randomly go to other places in the grid.
    for u in range(U):
        for y in range(X):
            P[u,coords2states[reward_coord],y] = 1/X
    r_std = np.zeros_like(r)
    mdp = MDPSimulator.MDPSim(P,r,r_std,info={"coords2states":coords2states, "states2coords":states2coords, "actions":actions})
    return mdp

def check_generate_clean_2d_maze():
    mdp = generate_clean_2d_maze()
    print(mdp.show())

#check_generate_clean_2d_maze()

