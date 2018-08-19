from itertools import product

from gym import Env
import numpy as np


class Minefield(Env):
    """
    Perhaps even inherit from GoalEnv?
    """
    def __init__(self,
                 shape=np.array([64, 53], dtype=np.int),
                 mine_penalty=-40,
                 reach_reward=100,
                 step_cost=-1,
                 rand_action_prob=0.05,
                 num_mines=80,
                 start=np.array([np.array([60, 52], dtype=np.int)]),
                 terminal_states=np.array([np.array([60, 2], dtype=np.int)]),
                 random_generator=None):

        self.rand_gen = random_generator if random_generator else self.seed(142)
        self.dim = len(shape)
        self.shape = shape
        self.num_mines = 80
        self.num_mines = num_mines
        # All possible coordinates+minefield tuples
        self.observation_space = [np.array(coord, dtype=np.int) for coord in product(*[range(d) for d in shape])]

        self.action_space = list(np.vstack([np.eye(self.dim, dtype=np.int), -1 * np.eye(self.dim, dtype=np.int)]))
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
        :param input_action:
        :return:
        """

        # Take a random action with probability self.rand_action_prob
        if debug:
            print("Taking a step: ", input_action)
        action = input_action if self.rand_gen.uniform(0, 1) > self.rand_action_prob else self.get_random_action()
        if debug and not np.array_equiv(action, input_action):
            print("Randomly selected a different action")

        self.cur_state = self.compute_next_state(self.cur_state, action)
        hit_mine = self.minefield[self.cur_state[0], self.cur_state[1]] == 1
        reached_terminal = self.is_terminal(self.cur_state)
        if debug and hit_mine:
            print("Hit a mine at location: ", self.cur_state)

        done = reached_terminal or hit_mine
        reward = (self.reach_reward if reached_terminal
                  else (self.mine_penalty if hit_mine else self.step_cost))
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
        return any([np.array_equiv(state, terminal) for terminal in self.terminal_states])

    def gen_field(self, num_mines):
        """
        Generates a random Minefield
        :return:
        """

        self.minefield = np.zeros(self.shape, dtype=np.int)
        # Choose num_mines according to a linearly decaying distribution, without replacement
        # Mines cannot be located in a start state or a terminal state
        while len(self.minefield.nonzero()[0]) < num_mines:
            x = int(self.rand_gen.triangular(0, 0, self.shape[1] - 1))
            y = int(self.rand_gen.triangular(0, 0, self.shape[0] - 1))
            if np.array([y, x], dtype=np.int) not in (self.terminal_states + self.start_states) and self.minefield[y, x] == 0:
                self.minefield[y, x] = 1

        return self.minefield

    def compute_next_state(self, action, input_state=None):
        """
        Return the next step, after applying the action. Corrects for boundary limitations
        :param input_state: Numpy array of coordinates
        :param action:
        :return: a new state, respecting the borders of the minefield
        """
        input_state = self.cur_state if input_state is None else input_state
        return np.minimum(np.maximum(input_state + action,
                          np.zeros(len(self.shape), dtype=np.int)),
                          self.shape - 1)

    def get_adjacent_squares(self, input_state):
        """
        Get all states reachable by a single action
        :param input_state:
        :return:
        """
        return np.unique([self.compute_next_state(input_state, action) for action in self.action_space], axis=0)

if __name__ == "__main__":
    m = Minefield()
    actions = m.action_space
    assert(all(m.compute_next_state(action=np.array([1, 0]), input_state=np.array([63, 52])) == np.array([63, 52])))
    assert (all(m.compute_next_state(action=np.array([0, 1]), input_state=np.array([63, 52])) == np.array([63, 52])))
    assert (all(m.compute_next_state(action=np.array([-1, 0]), input_state=np.array([0, 0])) == np.array([0, 0])))
    action_index = 0
    done = False
    total_reward = 0
    state = m.cur_state
    num_steps = 0
    print("Starting")
    while not done:
        num_steps += 1
        state, reward, done, info = m._step(m.action_space[action_index], debug=True)
        action_index = (action_index + 1) % len(m.action_space)
        # note: this should be probably be a decaying series
        total_reward += reward

    print("Done! last reward: ", total_reward, " , last state: ", state, ", number of steps: ", num_steps)
    # Sanity check, compute total scores based on all incurred costs:
    assert(total_reward == m.step_cost * (num_steps - 1) + (m.reach_reward if m.is_terminal(state) else m.mine_penalty))
