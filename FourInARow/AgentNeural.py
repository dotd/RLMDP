from FourInARow.AgentMCTS import AgentMCTSRandom


class AgentNeural(AgentMCTSRandom):

    def __init__(self,
                 random_generator,
                 shape,
                 player_num):
        super().__init__(random_generator, shape, player_num)