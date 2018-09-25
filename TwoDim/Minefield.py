from itertools import product

from gym import Env
import numpy as np


def generate_standard_minefield_parameters(rand_gen, num_mines, shape, goal_reward, mine_reward, counter_limit=3000):
    start_state = (0,0)
    start_states = set([start_state])
    terminal_state = (shape[0] - 1, shape[1] - 1)
    terminal_states = set([terminal_state])
    rewards = {terminal_state: goal_reward}
    mines = {}
    counter = 0
    while len(mines) < num_mines and counter < counter_limit:
        counter += 1
        x = int(rand_gen.triangular(0, (shape[1] - 1)//2, shape[1] - 1))
        y = int(rand_gen.triangular(0, (shape[0] - 1)//2, shape[0] - 1))
        mine_coord = (y, x)
        if mine_coord not in mines and mine_coord not in terminal_states and mine_coord not in start_states:
            mines[mine_coord] = mine_reward

    if counter >= counter_limit:
        print("did not generated all required mines")

    for coord, reward in mines.items():
        terminal_states.add(coord)
        rewards[coord] = reward

    return start_states, terminal_states, rewards


class Minefield(Env):
    """
    Perhaps even inherit from GoalEnv?

    Everything should be numpy arrays?
    """
    def __init__(self,
                 random_generator,
                 shape,
                 step_reward,
                 rand_action_prob,
                 start_states,
                 terminal_states,
                 rewards
                 ):

        self.rand_gen = random_generator
        self.dim = len(shape)
        self.shape = shape
        # All possible coordinates+minefield tuples
        # self.observation_space = [np.array(coord, dtype=np.int) for coord in product(*[range(d) for d in shape])]

        self.action_space = list(np.vstack([np.eye(self.dim, dtype=np.int), -1 * np.eye(self.dim, dtype=np.int)]))
        self.step_reward = step_reward
        self.rand_action_prob = rand_action_prob

        self.start_states = list(start_states)
        self.terminal_states = terminal_states
        self.rewards = rewards

        self.cur_state = self.get_random_start_state()
        # Need to generate the mine field here below. Figure it out later.

    def step(self, input_action: np.ndarray, debug=False):
        """
        Apply given action, and return a new state and reward
        :param input_action: ndarray - input vec
        :param debug: - debug flag
        :return: next state
        """

        # Take a random action with probability self.rand_action_prob
        if debug:
            print("Taking a step: ", input_action)
        random_step_flag = self.rand_gen.uniform(0, 1) < self.rand_action_prob
        action = self.get_random_action() if random_step_flag else input_action
        if debug and random_step_flag:
            print("Randomly selected a different action")

        prev_state = self.cur_state
        self.cur_state = self.compute_next_state(self.cur_state, action)
        changed_state = np.not_equal(prev_state, self.cur_state)
        hit_terminal = self.is_terminal(self.cur_state)
        if debug and hit_terminal:
            print("Hit a terminal location: ", self.cur_state)

        done = hit_terminal
        cur_state_tuple = tuple(self.cur_state)
        reward = self.rewards[cur_state_tuple] if cur_state_tuple  in self.rewards else self.step_reward
        # Change this if necessary
        info = {"random_step": random_step_flag, "changed_state": changed_state}
        return self.cur_state, reward, done, info

    def reset(self):
        self.cur_state = self.get_random_start_state()

    def render(self, mode='human', close=False):
        return self.minefield

    def seed(self, seed=42):
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
        state = self.cur_state if state is None else state
        is_terminal_flag = tuple(state) in self.terminal_states
        return is_terminal_flag


    """    
    def gen_field(self, rand_gen, num_mines):

        self.minefield = np.zeros(self.shape, dtype=np.int)
        # Choose num_mines according to a linearly decaying distribution, without replacement
        # Mines cannot be located in a start state or a terminal state
        while len(self.minefield.nonzero()[0]) < num_mines:
            x = int(self.rand_gen.triangular(0, 0, self.shape[1] - 1))
            y = int(self.rand_gen.triangular(0, 0, self.shape[0] - 1))
            if any([np.array_equiv(np.array([y, x], dtype=np.int), terminal) for terminal in self.terminal_states]) and \
                    any([np.array_equiv(np.array([y, x], dtype=np.int), start) for start in self.start_states]) and \
                self.minefield[y, x] == 0:
                self.minefield[y, x] = 1
                self.mines.add((y,x))
    """

    def compute_next_state(self, input_state=None, action=None):
        """
        Return the next step, after applying the action. Corrects for boundary limitations
        :param input_state: Numpy array of coordinates
        :param action:
        :return: a new state, respecting the borders of the minefield
        """
        input_state = self.cur_state if input_state is None else input_state
        return np.minimum(np.maximum(input_state + action,
                          np.zeros(len(self.shape), dtype=np.int)),
                          np.array(self.shape) - 1)

    def get_adjacent_squares(self, input_state):
        """
        Get all states reachable by a single action
        :param input_state:
        :return:
        """
        return np.unique([self.compute_next_state(input_state, action) for action in self.action_space], axis=0)

    def close(self):
        pass

    def __str__(self):
        lines = list()
        lines.append("dim={}".format(self.dim))
        lines.append("shape={}".format(self.shape))

    def get_all_states(self):
        """
        :return: a list of all coordinates on the board
        """
        return list(product(*[range(dim) for dim in self.shape]))


