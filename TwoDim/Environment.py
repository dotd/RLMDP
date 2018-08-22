from TwoDim.TwoDimMDP import  TwoDimSparseMDPSimulator
from TwoDim.QLearning import Q_Learning, Transition
import numpy as np
random = np.random.RandomState(0)
import itertools

sizes = (10,10)
rewards = {(9,9):1}
terminal_states = [(9,9)]
#start_states = list(itertools.product(range(sizes[0]),range(sizes[1])))
start_states = [(0,0)]
X = TwoDimSparseMDPSimulator(shape=sizes, noise=0.1, rewards=rewards, random = random, start_states=start_states, terminal_states=terminal_states)
actions = X.get_actions_list()

Q = Q_Learning(U=actions, random = random, eps_max_action = 1e-3, sizes = sizes)
state = X.get_random_start_state()
last_time = 0
deltas = []
num_steps = 20000
debug_show_times = 2
period_of_show = num_steps//debug_show_times - 1
limit_episode = 50
episode_counter = 0
for i in range(num_steps):
    if X.is_terminal(state) or episode_counter>=limit_episode:
        delta = i-last_time
        deltas.append(delta)
        last_time = i
        episode_counter = 0
        print("d={}, e={}, lr={}, max_delta_Q={}".format(delta, Q.eps_threshold, Q.lr, Q.max_abs_delta_Q))
        Q.max_abs_delta_Q = 0
        state = X.get_uniform_random_state()
        continue

    action = Q.choose_action(state)
    next_state, reward = X.get_next_state(state, action)
    Q.add_tuple(Transition(state, action, next_state, reward))
    Q.update()
    #if i % period_of_show==0:
    #    print(Q)
    state = next_state
    episode_counter += 1


