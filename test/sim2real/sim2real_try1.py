import SpecificMDPs
import numpy as np
import MDPSolver
import graphs.Graphs as g
import QAgents as q
import Environment as e
from samplers.Sampler import BasicSampler
import matplotlib.pyplot as plt

X = 5
U = 3
B = X
std = 0
random_state = np.random.RandomState()
gamma = 0.5

mdp = SpecificMDPs.generate_random_MDP(X,U,B,std=std, random_state=random_state)
#mdp = SpecificMDPs.generate_1D_MDP(X, noise=0.1, random_state=random_state)
U = mdp.U
print(mdp.show())

mu_PI, J_collector, Q_PI, iter_counter = MDPSolver.PI(mdp, gamma, mu=None, max_iters=10)
print("iter_counter={}".format(iter_counter))
print("mu=\n{}".format(mu_PI))
print("J_collector=\n{}".format(J_collector))
g.show_J_collector(J_collector)

q_agent = q.SimpleQAgent(X, U, gamma, epsil=0.1, alpha=1e-1, random=random_state)

env = e.Environment(mdp, q_agent)

bs = BasicSampler(variable_names=["Q","epsilon","step","delQ","del_mu"])

print("epsilon={}".format(env.agent.epsilon))
num_rounds = 1000
for p in range(num_rounds):
    if p % (num_rounds//10) == 0 or p==num_rounds-1:
        print("p={}".format(p))
    delQ = np.linalg.norm(q_agent.Q - Q_PI,"fro")
    del_mu = np.linalg.norm(q_agent.get_policy() - mu_PI,"fro")
    bs.add(q_agent.Q, q_agent.epsilon, q_agent.lr, delQ, del_mu)
    env.play_round(300)

print("mu_PI=\n{}".format(mu_PI))
print("q_agent.get_policy()=\n{}".format(q_agent.get_policy()))



plt.figure(2)
plt.subplot(411)
plt.semilogy(bs.get().delQ)
plt.title("Delta Q from optimal")
plt.subplot(412)
plt.plot(bs.get().del_mu)
plt.title("Delta mu from optimal")
plt.subplot(413)
plt.semilogy(bs.get().step)
plt.title("step")
plt.subplot(414)
plt.semilogy(bs.get().epsilon)
plt.title("epsilon")
plt.show()
