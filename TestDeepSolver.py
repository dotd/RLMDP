import MDPSimulator
import MDPSolver
import SpecificMDPs
import Policies
import numpy as np
import time
from DeepSolver import DeepSolver
import torch.nn as nn


# The MDP itself
mdp = SpecificMDPs.generate_investment_sim()
# The policy
mu = Policies.generate_uniform_policy(mdp.X, mdp.U)
# discount factor
gamma = 0.5
# for the simulator, how many steps.
num_samples = 10

# Get the MRP (Markov Reward Process, i.e., MDP, under a specific policy)
P, R, R_std = MDPSolver.get_MRP(mdp, mu)

# get the trajectory
trajectory = mdp.simulate(0, mu, num_samples=num_samples)

start = time.time()
J_exact = MDPSolver.get_J(P, R, gamma)
print("J_exact={}".format(J_exact))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

num_batches = 10
start = time.time()
ds = DeepSolver(X=2,gamma=0.5,layer_sizes=(2,2,1), layer_activations=(nn.ReLU,None),input_mode="one_hot")
ds.add_trajectory(trajectory)
J_pred = ds.infer()
print("J_pred={}".format(np.array(J_pred.data).T))
for i in range(num_batches):
    ds.step()
    J_pred = ds.infer()
    print("{} J_pred={}".format(i,np.array(J_pred.data).T))

