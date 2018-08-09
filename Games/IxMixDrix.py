import numpy as np


class IxMixDrix:
    def __init__(self, x,y , num_players, size_victory, random):
        self.random = random
        self.x = x
        self.y = y
        self.size_victory = size_victory
        self.num_players = num_players
        self.board = np.zeros(shape=(y,x))
        self.player_turn = random.choice(num_players)+1

    def advance_player(self):
        self.player_turn += 1
        if self.player_turn == self.num_players +1:
            self.player_turn = 1

    def play(self, player, y, x):
        if self.board[y,x]==0:
            self.board[y,x] = player
            return
        return None

    def get_possible_moves(self):
        pass

    def check_victory(self):
        for x in range(self.x-self.size_victory+1):
            for y in range(self.y-self.size_victory+1):
                # this square is captured by a player
                if self.board[y,x]>0:
                    # horizontal
                    if self.board[y,x:(x+self.size_victory)]==self.board[y,x] or
                        self.board[y:(y+self.size_victory), x] == self.board[y, x] or
                        self.board
                        return self.board[y,x]
                    # vertical




