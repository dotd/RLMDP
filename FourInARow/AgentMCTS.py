from TwoDim.AgentBase import AgentBase


class AgentMCTSRandom(AgentBase):

    def __init__(self,
                 random_generator,
                 shape,
                 player_num):
        self.random_generator = random_generator
        self.dim = len(shape)
        self.shape = shape
        self.num_rows = self.shape[0]
        self.num_columns = self.shape[1]
        self.player_num = player_num

    def choose_action(self, board):
        possible_columns = self.get_next_free_spot_raw(board)
        if len(possible_columns)==0:
            return None
        random_idx = self.random_generator.randint(len(possible_columns))
        return possible_columns[random_idx]

    def get_next_free_spot_raw(self, board):
        possible_columns = []
        for column_idx in range(self.num_columns):
            for row_idx in range(self.num_rows):
                if board[row_idx, column_idx] == 0:
                    possible_columns.append(column_idx)
                    break
        return possible_columns

    def update(self, state):
        pass
