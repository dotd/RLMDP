import MDPSimulator
import MDPSolver
import SpecificMDPs
import Policies
import numpy as np
import time

mdp = SpecificMDPs.generate_investment_sim()
mu = Policies.generate_uniform_policy(mdp.X, mdp.U)
gamma = 0.5
num_samples = 10000
print(mdp.show())

P, R, R_std = MDPSolver.get_MRP(mdp, mu)

start = time.time()
J_exact = MDPSolver.get_J(P, R, gamma)
print("J_exact={}".format(J_exact))
print("Elapsed since last time: {}".format(time.time() - start))

R_M2 = MDPSimulator.get_R_M2(P, R, R_std, gamma, J_exact)
start = time.time()
M2_exact = MDPSolver.get_J(P, R_M2, gamma**2)
print("M2_exact={}".format(M2_exact))
print("Elapsed since last time: {}".format(time.time() - start))

trajectory = mdp.simulate(0, mu, num_samples=num_samples)
start = time.time()
J_td = MDPSolver.get_J_as_TD(trajectory=trajectory, gamma=gamma, X=mdp.X, alpha=10)
print("J_td={}".format(J_td))
print("Elapsed since last time: {}".format(time.time() - start))

if num_samples<=1001:
    start = time.time()
    J_MC = MDPSolver.get_J_as_MC_raw(trajectory, gamma, X=mdp.X)
    print("J_MC={}".format(J_MC))
    print(time.time() - start)

    start = time.time()
    M2_MC = MDPSolver.get_J_as_MC_raw(trajectory, gamma, X=mdp.X, func=lambda x: x * x)
    print("M2_MC={}".format(M2_MC))
    print("Elapsed since last time: {}".format(time.time() - start))
else:
    print("Skip straight forward MonteCarlo simulation")

start = time.time()
J_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X)
print("J_MC_filt={}".format(J_MC_filt))
print("Elapsed since last time: {}".format(time.time() - start))

start = time.time()
M2_MC_filt = MDPSolver.get_J_as_MC_filter(trajectory, gamma, X=mdp.X, func=lambda x:x*x)
print("M2_MC_filt={}".format(M2_MC_filt))
print("Elapsed since last time: {}".format(time.time() - start))


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
print("J_exact_LSTD={}".format(J_exact_LSTD))
print("Elapsed since last time: {}".format(time.time() - start))

start = time.time()
wm_exact_LSTD = MDPSolver.get_exact_J_LSTD(phi=phi, P=P, gamma=gamma**2, r=R_M2)
M_exact_LSTD = np.dot(phi, wm_exact_LSTD)
print("M_exact_LSTD={}".format(M_exact_LSTD))
print("Elapsed since last time: {}".format(time.time() - start))

V_exact_by_M_J = MDPSolver.get_V_by_J_M(J_exact, M2_exact)
R_V_exact = MDPSimulator.get_R_V(P, R, R_std, gamma, J_exact)
V_exact_direct = MDPSolver.get_J(P, R_V_exact, gamma**2)
print("V_exact_by_M_J={}".format(V_exact_by_M_J))
print("V_exact_direct={}".format(V_exact_direct))


start = time.time()
w_sim = MDPSolver.get_simulation_J_LSTD(phi_class, trajectory, gamma)
J_sim_LSTD = np.dot(phi_class.phi, w_sim)
print("J_sim_LSTD={}".format(J_sim_LSTD))
print("Elapsed since last time: {}".format(time.time() - start))
