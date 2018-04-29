import MDP
import MDPSolver
import SpecificMDPs
import Policies
import numpy as np
import time
from Filters.OnlineUtils import ComputeBasicStats
import Utils

# The MDP itself

#mdp = SpecificMDPs.generate_investment_sim(R1_std=0)
mdp = SpecificMDPs.generate_random_MDP(X=4, U=2, B=3, std = 0, random_state = np.random.RandomState(0))
# The policy
mu = Policies.generate_uniform_policy(mdp.X, mdp.U)
# discount factor
gamma = 0.5
# for the simulator, how many steps.
num_samples = 100000
# Print the DMP to the console
print("The dynamics of the MDP:")
print(mdp.show())

# Get the MRP (Markov Reward Process, i.e., MDP, under a specific policy)
P, R, R_std = MDPSolver.get_MRP(mdp, mu)

start = time.time()
J_exact = MDPSolver.get_J(P, R, gamma)
print("J_exact={}".format(Utils.show_numpy_vector_nicely(J_exact)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

R_M2 = MDP.get_R_M2(P, R, R_std, gamma, J_exact)
start = time.time()
M2_exact = MDPSolver.get_J(P, R_M2, gamma**2)
print("M2_exact={}".format(Utils.show_numpy_vector_nicely(M2_exact)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

# simulate a trajectory
trajectory = mdp.simulate(0, mu, num_samples=num_samples)
x = [s[0] for s in trajectory]
r = [s[2] for s in trajectory]

start = time.time()
J_td = MDPSolver.get_J_as_TD(trajectory=trajectory, gamma=gamma, X=mdp.X, alpha=10)
print("J_td={}".format(Utils.show_numpy_vector_nicely(J_td)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

if num_samples <= 1001:
    start = time.time()
    J_MC = MDPSolver.get_J_as_MC_raw(trajectory, gamma, X=mdp.X)
    print("J_MC={}".format(Utils.show_numpy_vector_nicely(J_MC)))
    print(time.time() - start)

    start = time.time()
    M2_MC = MDPSolver.get_J_as_MC_raw(trajectory, gamma, X=mdp.X, func=lambda x: x * x)
    print("M2_MC={}".format(Utils.show_numpy_vector_nicely(M2_MC)))
    print("Elapsed since last time: {}".format(time.time() - start))
else:
    print("Skip straight forward MonteCarlo simulation")


start = time.time()
J_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X)
print("J_MC_filt={}".format(Utils.show_numpy_vector_nicely(J_MC_filt)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

start = time.time()
M2_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X, func=lambda x:x*x)
print("M2_MC_filt={}".format(Utils.show_numpy_vector_nicely(M2_MC_filt)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

start = time.time()
ABS_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X, func=lambda x:abs(x))
print("ABS_MC_filt={}".format(Utils.show_numpy_vector_nicely(ABS_MC_filt)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

phi = np.random.normal(size=(mdp.X,mdp.X))
class PhiClass:
    def __init__(self, phi):
        self.phi = phi
    def get(self, x):
        return self.phi[x]

phi_class = PhiClass(phi)

start = time.time()
w_exact_LSTD = MDPSolver.get_exact_J_LSTD(phi=phi, P=P, gamma=gamma, r=R)
J_exact_LSTD = np.dot(phi, w_exact_LSTD)
print("J_exact_LSTD={}".format(Utils.show_numpy_vector_nicely(J_exact_LSTD)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

start = time.time()
wm_exact_LSTD = MDPSolver.get_exact_J_LSTD(phi=phi, P=P, gamma=gamma**2, r=R_M2)
M_exact_LSTD = np.dot(phi, wm_exact_LSTD)
print("M_exact_LSTD={}".format(Utils.show_numpy_vector_nicely(M_exact_LSTD)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

V_exact_by_M_J = MDPSolver.get_V_by_J_M(J_exact, M2_exact)

R_V_exact = MDP.get_R_V(P, R, R_std, gamma, J_exact, moment_func=lambda x: x*x)
V_exact_direct = MDPSolver.get_J(P, R_V_exact, gamma**2)

filter = MDPSolver.get_discount_factor_as_filter(gamma, filt_len = 40)
V_sample = MDPSolver.get_B_moments_by_filter(mdp.X, x, r, filter, moment_func = lambda x: x*x, reward_func = lambda x: x)
special_func = lambda x: abs(x) ** 3
cbs = ComputeBasicStats(X=mdp.X, filter=filter, moments_funcs=[lambda x: x*x])
cbs.add_vecs(x,r)

print("V_exact_by_M_J={}".format(Utils.show_numpy_vector_nicely(V_exact_by_M_J)))
print("V_exact_direct={}".format(Utils.show_numpy_vector_nicely(V_exact_direct)))
print("V_sample={}".format(Utils.show_numpy_vector_nicely(V_sample)))
result_cbs = cbs.get()
print("V_online={}".format(Utils.show_numpy_vector_nicely([result_cbs[x][1] for x in range(mdp.X)])))
print("\n")
print("Comparing the J for the online:")
print("J_exact={}".format(Utils.show_numpy_vector_nicely(J_exact)))
print("J_online={}".format(Utils.show_numpy_vector_nicely([result_cbs[x][0] for x in range(mdp.X)])))
print("\n")
print("doing experiment with L1 moment")
R_S1_exact = MDP.get_R_V(P, R, R_std, gamma, J_exact, moment_func=special_func)
S1_exact_direct = MDPSolver.get_J(P, R_S1_exact, special_func(gamma))
print("S1_exact_direct={}".format(Utils.show_numpy_vector_nicely(S1_exact_direct)))
# getting the result from cbs which is simulator direct
print("S_online={}".format(Utils.show_numpy_vector_nicely([result_cbs[x][1] for x in range(mdp.X)])))

print("\n")

start = time.time()
w_sim = MDPSolver.get_simulation_J_LSTD(phi_class, trajectory, gamma)
J_sim_LSTD = np.dot(phi_class.phi, w_sim)
print("J_sim_LSTD={}".format(Utils.show_numpy_vector_nicely(J_sim_LSTD)))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

start = time.time()
mu_PI_opt, J_collector_PI, Q_PI, iter_counter = MDPSolver.PI(mdp, gamma)
print("mu_PI_opt=\n{}".format(mu_PI_opt))
print("J_collector_PI=\n{}".format(J_collector_PI))
print("Q_PI={}".format(Q_PI))
print("iter_counter={}".format(iter_counter))
print("Elapsed since last time: {}".format(time.time() - start))
print("\n")

