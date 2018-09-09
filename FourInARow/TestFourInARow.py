import unittest
import numpy as np
from FourInARow.FourInARow import FourInARow
from FourInARow.AgentMCTS import AgentMCTSRandom


class TestAgentDQN(unittest.TestCase):

    def test_run_general(self, random_seed=142, shape=(7, 10), win_len=4):
        random = np.random.RandomState(random_seed)
        four_in_row = FourInARow(random_generator=random, shape=shape, win_len=win_len)

        players = dict([(1, AgentMCTSRandom(random_generator=random, shape=shape, player_num=1)),
                   (-1, AgentMCTSRandom(random_generator=random, shape=shape, player_num=-1))])

        for i in range(np.prod(shape)+1):
            print("-----\nBegin of round {}.".format(i))
            print("Board is:\n{}".format(four_in_row.render()))
            print("Now player num {} is playing.".format(four_in_row.current_player))
            player = players[four_in_row.current_player]
            column_action = player.choose_action(four_in_row.board)
            if column_action is None:
                # The board is full. No action to select
                break;
            is_winner = four_in_row.step(column_action)
            if is_winner != 0:
                print("We have a winner! Player number={}".format(is_winner))
                print("The final board is:\n{}".format(four_in_row.render()))
                break

    def test_specific_boards_for_winning(self, random_seed=142, shape=(6, 7), win_len=4):
        random = np.random.RandomState(random_seed)
        four_in_row = FourInARow(random_generator=random, shape=shape, win_len=win_len)
        four_in_row.board[0,0:2] = 1
