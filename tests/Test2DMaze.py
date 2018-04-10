
from SpecificMDPs import *
from QAgents import SimpleQAgent
from MDPSolver import PI
from Environment import Environment

mdp = generate_clean_2d_maze(x_size=4, y_size = 3, reward_coord = (3,2))
print(mdp.show())

gamma = 0.9;
mu = Policies.generate_deterministic_policy(mdp.X, mdp.U)
print("Before leanring")
print("mu=\n{}".format(mu))
mu, J_collector, Q, iter_counter = PI(mdp,gamma, mu=mu)
print("After leanring")
print("mu=\n{}".format(mu))
print("Q=\n{}".format(Q))
print(mdp.show_policy(mu))

q_agent = SimpleQAgent(mdp.X,mdp.U,gamma)
environment = Environment(mdp=mdp, agent=q_agent)
environment.mdp.init()

steps = 100000
for step in range(steps):
    environment.play_round()
    if step%((steps-1)//2)==0:
        print("====={}={}=========".format("step",step))
        print(mdp.show_policy(q_agent.get_policy()))
        print(q_agent.Q)

print("Q=\n{}".format(Q))
print("compare policies:")
print(q_agent.get_policy()-mu)