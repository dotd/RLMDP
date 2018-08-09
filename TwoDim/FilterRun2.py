from TwoDim.AgentFilter import AgentFilter
from TwoDim.TwoDimMDP import TwoDimSparseMDPSimulator
import matplotlib.pyplot as plt
from TwoDim.TwoDimUtils import *

np_random = np.random.RandomState(0)
print("AAAAAAA")
lr = 0.00001
num_episodes = 5000
noise = 0.2

sizes = (10, 9)
rewards = {(sizes[0]-1, sizes[1]-1): 100}
start_states = [(0, 0)]
terminal_states = [key for key,value in rewards.items()]

# Environment Definition
env = TwoDimSparseMDPSimulator(sizes=sizes, noise=noise, rewards=rewards, random=np_random, start_states=start_states,
                             terminal_states=terminal_states)
actions = env.get_actions_list()
num_actions = len(actions)

# Agent Definition
agent = AgentFilter(grid_sizes=sizes, num_actions=num_actions, lr=lr, GAMMA=0.5, filter_in=[1]*5)
episode_durations = []

episode_len = np.sum(sizes) * 10
flt_len=10
#optimization_method = "regular"
optimization_method = "filter"
for i_episode in range(num_episodes):
    # Initialize the environment and state
    print("episode={}".format(i_episode))

    env.reset()

    # print("init_state = {}".format(env.cur_state))

    for t in range(episode_len):
        # Current state
        cur_state_vec = env.cur_state
        cur_state_screen = get_screen(cur_state_vec, env)

        # Action
        action, status_action = agent.select_action(cur_state_screen)
        action_idx = action[0][0]
        action_vec = actions[action_idx]

        # step
        next_state_vec = env.step(action_vec)
        next_state_screen = get_screen(next_state_vec, env)

        # Reward
        cur_reward = env.get_reward(cur_state_vec)
        reward = torch.tensor([cur_reward], device=agent.device)

        # Store the transition in memory
        if optimization_method=="regular":
            agent.memory.push(cur_state_screen, action, next_state_screen, reward)
            status_optimize = agent.optimize_model()
        elif optimization_method=="filter":
            # Perform one step of the optimization (on the target network)
            agent.update_filters(cur_state_screen, action, reward=cur_reward)
            status_optimize = agent.optimize_filter()
        else:
            print("optimization_method invalid")


        # if we are done, we do nothing. Just init the dynamics
        if cur_state_vec in env.rewards:
            print("t={}, cur_state={}, cur_reward={}, action_idx={}, action_vec={}, next_state_vec={}, status_action={}, optimize={}".format(t,cur_state_vec, cur_reward, action_idx,  action_vec,  next_state_vec, status_action, status_optimize))
        if env.is_terminal(cur_state_vec) and env.rewards[cur_state_vec] > 0:
            print("is_terminal {}".format(t+1))
            break

    # Update the target network
    episode_durations.append(t + 1)

    if i_episode % 15 ==0:
        plt.figure(1)
        vec = smooth_signal(episode_durations, flt_len)
        plt.plot(vec)
        plt.pause(0.05)

print('Complete')
print(episode_durations)

plt.figure(1)
vec = smooth_signal(episode_durations, flt_len)
plt.plot(vec)
plt.show(block=True)

