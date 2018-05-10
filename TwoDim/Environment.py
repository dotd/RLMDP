from TwoDim.TwoDimMDP import  TwoDimSparseMDPSimulator
from TwoDim.Q_learning import Q_Learning, Transition
import numpy as np
random = np.random.RandomState(0)
import itertools

sizes = (7,7)
rewards = {(6,6):1}
terminal_states = [(6,6)]
start_states = list(itertools.product(range(sizes[0]),range(sizes[1])))
start_states = [(0,0)]
X = TwoDimSparseMDPSimulator( sizes=sizes, noise=0.1, rewards={(6,6):1}, random = random, start_states=start_states, terminal_states=terminal_states)
actions = X.get_actions_list()

Q = Q_Learning(U=actions, random = random, eps_max_action = 1e-3, sizes = sizes)
state = X.get_start_state()
last_time = 0
deltas = []
num_steps = 200000
debug_show_times = 2
period_of_show = num_steps//debug_show_times - 1
for i in range(num_steps):
    if X.is_terminal(state):
        delta = i-last_time
        deltas.append(delta)
        last_time = i
        print("d={}, e={}".format(delta, Q.eps_threshold))

    action = Q.choose_action(state)
    next_state, reward = X.get_next_state(state, action)
    Q.add_tuple(Transition(state, action, next_state, reward))
    Q.update()
    if i % period_of_show==0:
        print(Q)
    state = next_state


