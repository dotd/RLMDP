
from SpecificMDPs import *
from QAgents import SimpleQAgent
from DeepAgents.DeepQAgent import DQN
from MDPSolver import PI
from Environment import Environment
import torch.nn as nn

x_size = 4
y_size = 3

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


if False:
    q_agent = SimpleQAgent(mdp.X, mdp.U, gamma)
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


if True:
    replay_capacity = 1000
    layer_sizes = [mdp.X, 10, mdp.U]
    layer_activations = [nn.ReLU, None]
    batch_size = 2
    dqn = DQN( mdp.X, mdp.U, gamma, replay_capacity,layer_sizes, layer_activations, batch_size)

    environment = Environment(mdp=mdp, agent=dqn)
    environment.mdp.init()
    steps = 100000
    for step in range(steps):
        environment.play_round()
        if step%((steps-1)//20)==0:
            print("====={}={}=========".format("step",step))
            print(mdp.show_policy(dqn.get_policy()))
            print(dqn.get_Q())
