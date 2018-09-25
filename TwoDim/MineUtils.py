import numpy as np
import matplotlib.pyplot as plt

from TwoDim.DQN.RunDQN import compact2full

"""
Each entry in policy has the following fields:
x
y
up
down
left
right
"""


def show_policy(plt_param, policy, start_states, terminal_states, x_scale=1, y_scale=1):
    """
    :param plt_param:
    :param policy:
    :param x_scale: If we want to scale on the X axis
    :param y_scale: If we want to scale on the Y axis
    :param start_states:
    :param terminal_states
    :return: None = plot on the given plt_param
    """
    for entry in policy:
        center_x = entry["x"] * x_scale
        center_y = entry["y"] * y_scale
        up_arrow = center_y + y_scale / 2 * entry["up"]
        down_arrow = center_y - y_scale / 2 * entry["down"]
        left_arrow = center_x + x_scale / 2 * entry["right"]
        right_arrow = center_x - x_scale / 2 * entry["left"]
        plt_param.plot(center_x, center_y, 'rp', markersize=5)
        plt_param.plot([center_x, left_arrow], [center_y, center_y], color="blue")
        plt_param.plot([center_x, right_arrow], [center_y, center_y], color="blue")
        plt_param.plot([center_x, center_x], [center_y, up_arrow], color="blue")
        plt_param.plot([center_x, center_x], [center_y, down_arrow], color="blue")
        plt_param.plot([center_x - x_scale / 2, center_x + x_scale / 2], [center_y - y_scale / 2, center_y - y_scale / 2], color="black")
        plt_param.plot([center_x - x_scale / 2, center_x + x_scale / 2], [center_y + y_scale / 2, center_y + y_scale / 2], color="black")
        plt_param.plot([center_x - x_scale / 2, center_x - x_scale / 2], [center_y - y_scale / 2, center_y + y_scale / 2], color="black")
        plt_param.plot([center_x + x_scale / 2, center_x + x_scale / 2], [center_y - y_scale / 2, center_y + y_scale / 2], color="black")
    for start in start_states:
        center_x = start[0] * x_scale
        center_y = start[1] * y_scale
        plt_param.text(center_x, center_y, 'S', horizontalalignment='center', verticalalignment = 'center')
    for terminal_state in terminal_states:
        center_x = terminal_state[0] * x_scale
        center_y = terminal_state[1] * y_scale
        plt_param.text(center_x, center_y, 'T', horizontalalignment='center', verticalalignment = 'center')


def show_minefield(plt_param, mdp, agent):
    states = mdp.get_all_states()
    policy_state = []
    for state in states:
        policy = {}
        state_full = compact2full(np.array(state), mdp.shape)
        probabilities = agent.get_policy_probabilities(state_full)
        policy["x"] = state[0]
        policy["y"] = state[1]
        policy["right"] = probabilities[0].item()
        policy["up"] = probabilities[1].item()
        policy["left"] = probabilities[2].item()
        policy["down"] = probabilities[3].item()
        policy_state.append(policy)
    show_policy(plt_param, policy_state, mdp.start_states, mdp.terminal_states)


if __name__ == "__main__":
    policy_syn = [{"x": 0.0, "y": 0.0, "up": 0.5, "down": 0.2, "right": 0.2, "left": 0.1}]
    policy_syn += [{"x": 0.0, "y": 1.0, "up": 0.9, "down": 0.1, "right": 0, "left": 0}]
    policy_syn += [{"x": 1.0, "y": 0.0, "up": 0, "down": 1, "right": 0, "left": 0}]
    policy_syn += [{"x": 1.0, "y": 1.0, "up": 0.5, "down": 0.2, "right": 0.2, "left": 0.1}]
    policy_syn += [{"x": 1.0, "y": 2.0, "up": 0.5, "down": 0.2, "right": 0.2, "left": 0.1}]
    show_policy(10, 10, policy_syn)
