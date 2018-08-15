from itertools import product

from gym import Env
import numpy as np


class Minefield(Env):
    """
    Perhaps even inherit from GoalEnv?
    """
    def __init__(self,
                 shape=np.array([64, 53]),
                 mine_penalty=-40,
                 reach_reward=100,
                 step_cost=1,
                 rand_action_prob=0.05,
                 num_mines=80,
                 start=np.array([np.array([60, 52])]),
                 terminal_states=np.array([np.array([60, 2])]),
                 random_generator=None):

        self.rand_gen = random_generator if random_generator else self.seed(42)
        self.dim = len(shape)
        self.shape = shape
        self.num_mines = 80
        self.num_mines = num_mines
        # All possible coordinates+minefield tuples
        self.observation_space = [np.array(coord) for coord in product(*[range(d) for d in shape])]

        self.action_space = list(np.vstack([np.eye(self.dim), -1 * np.eye(self.dim)]))
        self.mine_penalty = mine_penalty
        self.reach_reward = reach_reward
        self.step_cost = step_cost
        self.rand_action_prob = rand_action_prob
        self.start_states = start
        self.terminal_states = terminal_states
        self.cur_state = self.get_random_start_state()
        self.minefield = self.gen_field(num_mines=num_mines)
        # Need to generate the mine field here below. Figure it out later.

    def _step(self, input_action, debug=False):
        """
        Apply given action, and return a new state and reward
        :param action:
        :return:
        """

        # Take a random action with probability self.rand_action_prob
        action = input_action if self.rand_gen.uniform(0, 1) > self.rand_action_prob else self.get_random_action()
        if debug and action !=input_action:
            print("Randomly selected a different action")

        self.cur_state = self.compute_next_state(self.cur_state, action)
        done = self.cur_state in self.terminal_states or self.minefield[self.cur_state] == 1

        reward = (self.reach_reward if self.cur_state in self.terminal_states
                  else (self.mine_penalty if self.minefield[self.cur_state] == 1
                        else self.step_cost))
        # Change this if necessary
        info = None
        return self.cur_state, reward, done, info

    def _reset(self):
        self.cur_state = self.get_random_start_state()

    def _render(self, mode='human', close=False):
        return self.minefield

    def _seed(self, seed=42):
        """
        Returns the environment's random number genreator
        :param seed:
        :return:
        """
        return np.random.RandomState(seed)

    def get_random_start_state(self):
        """
        Returns a state selected u.a.r. from the start states
        :return:
        """
        if len(self.start_states) == 1:
            return self.start_states[0]
        return self.start_states[self.rand_gen.choice(len(self.start_states))]

    def get_random_action(self):
        return self.action_space[self.rand_gen.choice(len(self.action_space))]

    def is_terminal(self, state=None):
        if state is None:
            state = self.cur_state
        return state in self.terminal_states

    def gen_field(self, num_mines):
        """
        Generates a random Minefield
        :return:
        """
        self.minefield = np.zeros(self.shape)
        # Mines cannot be located in a start state or a terminal state
        candidate_locations = [coord for coord in self.observation_space
                               if coord not in self.terminal_states + self.start_states]
        mine_indices = self.rand_gen.choice(range(len(candidate_locations)), num_mines, replace=False)
        for coord in mine_indices:
            self.minefield[tuple(candidate_locations[coord])] = 1
        return self.minefield

    def compute_next_state(self, action, input_state=None):
        """
        Return the next step, after applying the action. Corrects for boundary limitations
        :param state: Numpy array of coordinates
        :param action:
        :return: a new state, respecting the borders of the minefield
        """
        state = self.cur_state if input_state is None else input_state
        return np.minimum(np.maximum(state + action,
                          np.zeros(len(self.shape))),
                          self.shape - 1)

    def get_adjacent_squares(self, state):
        """
        Get all states reachable by a single action
        :param state:
        :return:
        """
        return np.unique([self.compute_next_state(state, action) for action in self.action_space], axis=0)

if __name__ == "__main__":
    m = Minefield()
    actions = m.action_space
    assert(all(m.compute_next_state(action=np.array([1, 0]), input_state=np.array([63, 52])) == np.array([63, 52])))
    assert (all(m.compute_next_state(action=np.array([0, 1]), input_state=np.array([63, 52])) == np.array([63, 52])))
    assert (all(m.compute_next_state(action=np.array([-1, 0]), input_state=np.array([0, 0])) == np.array([0, 0])))
    action_index = 0
    done = False
    while not done:
        new_state, reward, done, info = m._step(m.action_space, debug=True)
        action_index = (action_index + 1) % len(m.action_space)

    print("Done! last reward: ", reward, " , last state: ", new_state)