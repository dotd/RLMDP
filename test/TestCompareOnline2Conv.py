import MDP
import MDPSolver
import SpecificMDPs
import Policies
import numpy as np
import time
from Filters.OnlineUtils import ComputeBasicStats
import Utils
filt_len = 2

# The MDP itself

#mdp = SpecificMDPs.generate_investment_sim(R1_std=0)
mdp = SpecificMDPs.generate_random_MDP(X=7, U=2, B=6, std = 0, random_state = np.random.RandomState(3))
# The policy
mu = Policies.generate_uniform_policy(mdp.X, mdp.U)
# discount factor
gamma = 0.6
# for the simulator, how many steps.
num_samples = 10

# simulate a trajectory
trajectory = mdp.simulate(0, mu, num_samples=num_samples)
x = [s[0] for s in trajectory]
r = [s[2] for s in trajectory]

J0, res0, values0, times0 = MDPSolver.get_J_as_MC_filter(trajectory, gamma, filt_len=filt_len, X=mdp.X, info=True)
filter = MDPSolver.get_discount_factor_as_filter(gamma, filt_len = filt_len)
cbs = ComputeBasicStats(X=mdp.X, filter=filter, moments_funcs=[lambda x: x*x])
cbs.add_vecs(x,r)

print("finish")
print("\n".join(["{},{}".format(x,y) for x,y in zip(res0, cbs.acum_j)]))