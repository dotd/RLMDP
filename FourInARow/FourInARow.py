from gym import Env
import numpy as np

'''
There are two players {1,2}
Player i is indicated with index i
'''


class FourInARow(Env):

    def __init__(self,
                 random_generator,
                 shape,  # (rows, columns)
                 win_len
                 ):

        self.random_generator = random_generator
        self.dim = len(shape)
        self.shape = shape
        self.win_len = win_len
        self.num_rows = self.shape[0]
        self.num_columns = self.shape[1]
        self.action_space = list(range(self.num_columns))
        self.board = None
        self.current_state = None
        self.current_player = None
        self.reset()

    def step(self, action_column):
        next_free_spot_row = self.get_next_free_spot_raw(action_column)
        self.board[next_free_spot_row, action_column] = self.current_player
        is_winner = self.check_winner(next_free_spot_row, action_column)
        self.current_player = -self.current_player
        # TODO next_state, reward, done, info
        return is_winner

    def get_next_free_spot_raw(self, column_idx):
        for row_idx in range(self.num_rows):
            if self.board[row_idx, column_idx] == 0:
                return row_idx
        return None

    def reset(self):
        self.board = np.zeros(shape=self.shape)
        self.current_player = self.random_generator.choice([-1, 1])

    def render(self, mode='human', close=False):
        str = []
        # str.append("It is turn of player={}".format(self.current_player))
        for row_idx in range(self.num_rows-1, -1, -1):
            line = []
            for column_idx in range(self.num_columns):
                if self.board[row_idx, column_idx] == 0:
                    line.append(".")
                elif self.board[row_idx, column_idx] > 0:
                    line.append("+")
                else:
                    line.append("-")
            str.append("".join(line))
        return "\n".join(str)

    def seed(self, seed=42):
        """
        Returns the environment's random number generator
        :param seed:
        :return:
        """
        return self.random_generator

    def close(self):
        pass

    def check_winner(self, last_move_row, last_move_column):
        # We need to check in 5 directions
        condition = self.are_all_the_same(last_move_row, last_move_column, -1, 0) or \
                    self.are_all_the_same(last_move_row, last_move_column, -1, -1) or \
                    self.are_all_the_same(last_move_row, last_move_column, -1, 1) or \
                    self.are_all_the_same(last_move_row, last_move_column, 0, 1) or \
                    self.are_all_the_same(last_move_row, last_move_column, 0, -1)
        return condition * self.current_player

    def are_all_the_same(self, last_move_row, last_move_column, direction_row, direction_column):
        if last_move_row + (self.win_len - 1) * direction_row < 0:
            return False
        if last_move_column + (self.win_len - 1) * direction_column < 0 or \
                last_move_column + (self.win_len - 1) * direction_column >= self.num_columns:
            return False
        player = self.board[last_move_row, last_move_column]
        for i in range(1, self.win_len):
            if self.board[last_move_row + i * direction_row, last_move_column + i * direction_column] != player:
                return False
        return True
