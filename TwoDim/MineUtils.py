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


def show_policy(policy, x_scale=1, y_scale=1, x=None, y=None):
    plt.figure()

    for entry in policy:
        center_x = entry["x"] * x_scale
        center_y = entry["y"] * y_scale
        up_arrow = center_y + y_scale / 2 * entry["up"]
        down_arrow = center_y - y_scale / 2 * entry["down"]
        left_arrow = center_x + x_scale / 2 * entry["left"]
        right_arrow = center_x - x_scale / 2 * entry["right"]
        plt.plot(center_x, center_y, 'rp', markersize=5)
        plt.plot([center_x, left_arrow], [center_y, center_y], color="blue")
        plt.plot([center_x, right_arrow], [center_y, center_y], color="blue")
        plt.plot([center_x, center_x], [center_y, up_arrow], color="blue")
        plt.plot([center_x, center_x], [center_y, down_arrow], color="blue")
        plt.plot([center_x - x_scale / 2, center_x + x_scale / 2], [center_y - y_scale / 2, center_y - y_scale / 2], color="black")
        plt.plot([center_x - x_scale / 2, center_x + x_scale / 2], [center_y + y_scale / 2, center_y + y_scale / 2], color="black")
        plt.plot([center_x - x_scale / 2, center_x - x_scale / 2], [center_y - y_scale / 2, center_y + y_scale / 2], color="black")
        plt.plot([center_x + x_scale / 2, center_x + x_scale / 2], [center_y - y_scale / 2, center_y + y_scale / 2], color="black")
    plt.show()


def show_minefield(mdp, agent):
    states = mdp.get_all_states()
    policy_state = []
    for state in states:
        policy = {}
        state_full = compact2full(np.array(state), mdp.shape)
        probabilities = agent.get_policy_probabilities(state_full)
        policy["x"] = state[0]
        policy["y"] = state[1]
        policy["up"] = probabilities[0].item()
        policy["down"] = probabilities[1].item()
        policy["right"] = probabilities[2].item()
        policy["left"] = probabilities[3].item()
        policy_state.append(policy)
    show_policy(policy_state)


if __name__ == "__main__":
    policy = [{"x": 0.0, "y": 0.0, "up": 0.5, "down": 0.2, "right": 0.2, "left": 0.1}]
    policy += [{"x": 0.0, "y": 1.0, "up": 0.9, "down": 0.1, "right": 0, "left": 0}]
    policy += [{"x": 1.0, "y": 0.0, "up": 0, "down": 1, "right": 0, "left": 0}]
    policy += [{"x": 1.0, "y": 1.0, "up": 0.5, "down": 0.2, "right": 0.2, "left": 0.1}]
    policy += [{"x": 1.0, "y": 2.0, "up": 0.5, "down": 0.2, "right": 0.2, "left": 0.1}]
    show_policy(10, 10, policy)
