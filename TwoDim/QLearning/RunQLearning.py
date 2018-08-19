import numpy as np
from TwoDim.Minefield import Minefield
from TwoDim.Coordinator import Coordinator
from TwoDim.QLearning.AgentQLearning import AgentQLearning


if __name__ == "__main__":
    mdp = Minefield(
        shape=(6, 7),
        num_mines=1,
        start=np.array([np.array([0, 0], dtype=np.int)]),
        terminal_states=np.array([np.array([5, 6], dtype=np.int)]))
    agent = AgentQLearning(actions=mdp.action_space, states=None)
    coordinator = Coordinator(mdp, agent)
    coordinator.step(2, True)

