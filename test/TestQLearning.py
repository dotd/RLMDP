import SpecificMDPs
import numpy as np
import MDPSolver
import graphs.Graphs as g
import QAgents as q
import Environment as e
from samplers.Sampler import BasicSampler
import matplotlib.pyplot as plt

X = 10
U = 4
B = 3
std = 0
random_state = np.random.RandomState(0)
gamma = 0.5

mdp = SpecificMDPs.generate_random_MDP(X,U,B,std = 1, random_state=random_state)
mu_PI, J_collector, Q_PI, iter_counter = MDPSolver.PI(mdp, gamma, mu=None, max_iters=10)
print("mu=\n{}".format(mu_PI))
print("J_collector=\n{}".format(J_collector))
g.show_J_collector(J_collector)

q_agent = q.SimpleQAgent(X, U, gamma, epsil=0.1, alpha=1e-1, random=random_state)

env = e.Environment(mdp, q_agent)

bs = BasicSampler(variable_names=["Q","epsilon","step","delQ","del_mu"])

for p in range(100):
    delQ = np.linalg.norm(q_agent.Q - Q_PI,"fro")
    del_mu = np.linalg.norm(q_agent.get_policy() - mu_PI,"fro")
    bs.add(q_agent.Q, q_agent.epsilon, q_agent.lr, delQ, del_mu)
    env.play_round(100)

plt.figure(2)   
plt.subplot(211)
plt.plot(bs.get().delQ)
plt.subplot(212)
plt.plot(bs.get().del_mu)
plt.show()


