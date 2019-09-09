import numpy as np
import MDPSolver
from samplers.Sampler import BasicSampler
import matplotlib.pyplot as plt
import test.sim2real.Sim2Real as s2r
import time as tim


def runner(mdp_s2r, random_state, sim2real_policy_vec, total_rounds):
    gamma = 0.5

    print(mdp_s2r.show())

    mu_PI, J_collector, Q_PI, iter_counter = MDPSolver.PI(mdp_s2r.mdps[0], gamma, mu=None, max_iters=10)
    #print("iter_counter={}".format(iter_counter))
    #print("mu=\n{}".format(mu_PI))
    #print("J_collector=\n{}".format(J_collector))  # g.show_J_collector(J_collector)

    X = mdp_s2r.mdps[0].X
    U = mdp_s2r.mdps[0].U
    q_agent = s2r.AgentSim2Real(X, U, gamma, epsil=0.1, alpha=1e-1, random=random_state)

    bs = BasicSampler(variable_names=["Q", "epsilon", "step", "delQ", "del_mu", "time"])
    sampler_period = 10

    print("epsilon={}".format(q_agent.epsilon.epsilon))
    c_vec = [0, 0, 0]

    for t in range(total_rounds):
        if t % (total_rounds//10) == 0 or t==total_rounds-1:
            print("t={}".format(t))

        if t % sampler_period==0 or t == total_rounds-1:
            delQ = np.sum(np.abs(q_agent.Q - Q_PI))/2
            del_mu = np.sum(np.abs(q_agent.get_policy() - mu_PI))/2
            bs.add(q_agent.Q, q_agent.epsilon.epsilon, q_agent.lr.lr, delQ, del_mu, t)

        c = random_state.choice(3, p=sim2real_policy_vec/np.sum(sim2real_policy_vec))
        c_vec[c] += 1
        # Decide to get from reality (online), simulation (online), reality replay buffer (offline)

        if c == 0:
            state = mdp_s2r.mdps[0].get_state()
            action = q_agent.select_action(state)
            next_state, reward, _, _ = mdp_s2r.mdps[0].step(action)
            q_agent.update(state, action, reward, next_state)
            q_agent.er.push([state, action, reward, next_state])

        if c == 1:
            samples = q_agent.er.sample(20)
            if samples is not None:
                for sample in samples:
                    q_agent.update(*sample)

        if c == 2:
            state = mdp_s2r.mdps[1].get_state()
            action = q_agent.select_action(state)
            next_state, reward, _, _ = mdp_s2r.mdps[1].step(action)
            q_agent.update(state, action, reward, next_state)

    print("q_agent.get_policy() - mu_PI=\n{}".format(q_agent.get_policy() - mu_PI))
    print("c_vec={}".format(c_vec))
    return bs


"""
The actions are the following:
0 - from reality into rb and do an online roll-out
1 - from reality replay buffer
2 - simulate
"""


mdp_s2r = s2r.generate_random_Sim2Real(X=10, U=3, B=6, std=0, random_state=np.random.RandomState(0), noise=0.5)
min_mu = []
random_state = np.random.RandomState(0)
vec_p = [0.01, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.99]
repeats = 3
total_exps = repeats * len(vec_p)

res = np.zeros(shape=(repeats, len(vec_p)))
cnt = 0
for idx, p in enumerate(vec_p):
    sim2real_policy_vec = [p, (1-p)/2, (1-p)/2]
    for r in range(repeats):
        print("cnt={}/{}, idx={}, repeat={}, value={}".format(cnt, total_exps, idx, r, p))
        bs = runner(mdp_s2r, random_state, sim2real_policy_vec, total_rounds=15000)
        res[r, idx] = np.mean(bs.get().del_mu)
        cnt += 1


    """
    plt.figure()
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
    plt.draw()
    plt.pause(0.001)
    plt.show(block=False)
    """

print(np.mean(res, axis=0))
input("Press Enter to continue...")