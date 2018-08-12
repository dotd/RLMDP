########### IMPORTS #################
import math
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn.functional as F
from TwoDim.TwoDimMDP import TwoDimSparseMDPSimulator
from TwoDim.TwoDimUtils import *
from TwoDim.Nets import DQN2

########### ENVIRONMENT #################

# Working well
# sizes = (5, 3); lr = 0.0001; num_episodes = 5000; noise = 0.0; BATCH_SIZE = 10; GAMMA = 0.6; # Adam
# sizes = (20, 19); lr = 0.005; num_episodes = 5000; noise = 0.0; BATCH_SIZE = 30; GAMMA = 0.6; # Adam
# sizes = (20, 19); lr = 0.004; num_episodes = 5000; noise = 0.0; BATCH_SIZE = 30; GAMMA = 0.6; # Adam

# Stable and big enough...
# sizes = (20, 19); lr = 0.003; num_episodes = 5000; noise = 0.0; BATCH_SIZE = 30; GAMMA = 0.6;

# Trying
# sizes = (32, 25); lr = 0.0005; num_episodes = 10000; noise = 0.0; BATCH_SIZE = 10; GAMMA = 0.5;

# Working OK
# optimizer = optim.SGD(policy_net.parameters(), lr = lr)
# sizes = (20, 19); lr = 0.001; num_episodes = 5000; noise = 0.0; BATCH_SIZE = 30; GAMMA = 0.6;

# sizes = (20, 12); lr = 0.0005; num_episodes = 1000; noise = 0.0; BATCH_SIZE = 200; GAMMA = 0.1;
sizes = (10, 9); lr = 0.00003; num_episodes = 5000; noise = 0.2; BATCH_SIZE = 100; GAMMA = 0.5 # ADAM

# Large rewards
# rewards = {(sizes[0]-1, sizes[1]-1): 100, (sizes[0]-2, sizes[1]-1): 100,(sizes[0]-1, sizes[1]-2): 100, (sizes[0]-2, sizes[1]-2): 100}
rewards = {(sizes[0]-1, sizes[1]-1): 100}
when_to_start_from_zeros = 0
episode_len = np.sum(sizes) * 10

nrr = np.random.RandomState(1234) # nrr - negative reward random
nrr_value = -10
number_of_negative_points = 0
for c in range(number_of_negative_points):
    rewards[(nrr.choice(sizes[0]),nrr.choice(sizes[0]))]=nrr_value

terminal_states = [key for key,value in rewards.items()]
# start_states = list(itertools.product(range(sizes[0]),range(sizes[1])))
start_states = [(0, 0)]
np_random = np.random.RandomState(0)
env = TwoDimSparseMDPSimulator(shape=sizes, noise=noise, rewards=rewards, random=np_random, start_states=start_states,
                               terminal_states=terminal_states)
actions = env.get_actions_list()
num_actions = len(actions)
device = torch.device("cpu")


def show_map_2d_as_text():
    lines = []
    for x in range(sizes[0]):
        line =[]
        for y in range(sizes[1]):
            with torch.no_grad():
                pos = (x, y)
                screen = get_screen(pos, env)
                state = policy_net(screen.float())
                action = state.max(1)[1].view(1, 1)
                letter = action_2d_2_letter(action)
                line.append(letter)
        lines.append("".join(line))
    return "\n".join(lines)


EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10
TARGET_UPDATE = 1

policy_net = DQN2(sizes=sizes).to(device)
# optimizer = optim.RMSprop(policy_net.parameters(),  lr=lr)
optimizer = optim.Adam(policy_net.parameters(), lr = lr)
memory = ReplayMemory(1000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            vec = policy_net(state.float())
            action_selected = vec.max(1)[1].view(1, 1)
            return action_selected, True
    else:
        action_random = torch.tensor([[random.randrange(num_actions)]], device=device, dtype=torch.long)
        return action_random, eps_threshold
episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return False
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    '''
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    '''
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch.float()).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values = policy_net(next_state_batch.float()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss


print("episode_max_len={}".format(episode_len))
print(show_map_2d_as_text())

flt_len = 10
for i_episode in range(num_episodes):
    # Initialize the environment and state
    print("episode={}".format(i_episode))

    # initialize
    if i_episode <= when_to_start_from_zeros:
        env.cur_state = env.get_uniform_random_state()
    else:
        env.reset()

    # print("init_state = {}".format(env.cur_state))

    for t in range(episode_len):
        # Current state
        cur_state_vec = env.cur_state
        cur_state_screen = get_screen(cur_state_vec, env)

        # Action
        action, status_action = select_action(cur_state_screen)
        action_idx = action[0][0]
        action_vec = actions[action_idx]

        # step
        next_state_vec = env.step(action_vec)
        next_state_screen = get_screen(next_state_vec, env)

        # Reward
        cur_reward = env.get_reward(cur_state_vec)
        reward = torch.tensor([cur_reward], device=device)

        # Store the transition in memory
        memory.push(cur_state_screen, action, next_state_screen, reward)

        # Perform one step of the optimization (on the target network)
        status_optimize = optimize_model()
        # if we are done, we do nothing. Just init the dynamics
        if cur_state_vec in env.rewards:
            print("t={}, cur_state={}, cur_reward={}, action_idx={}, action_vec={}, next_state_vec={}, status_action={}, optimize={}".format(t,cur_state_vec, cur_reward, action_idx,  action_vec,  next_state_vec, status_action, status_optimize))
        if env.is_terminal(cur_state_vec) and env.rewards[cur_state_vec] > 0:
            print("is_terminal {}".format(t+1))
            if t%10 == 0:
                print(show_map_2d_as_text())
            break

    # Update the target network
    episode_durations.append(t + 1)

    if i_episode % 15 ==0:
        plt.figure(1)
        vec = smooth_signal(episode_durations, flt_len)
        plt.plot(vec[flt_len:-flt_len])
        plt.pause(0.05)

print('Complete')
print(episode_durations)

plt.figure(1)
vec = smooth_signal(episode_durations, flt_len)
plt.plot(vec[flt_len:-flt_len])
plt.show(block=True)
