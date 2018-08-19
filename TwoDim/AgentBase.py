from abc import ABC, ABCMeta, abstractmethod


class AgentBase(ABC):
    """
    This class is an abstract class for all agents
    """

    @abstractmethod
    def choose_action(self, state):
        """
        This method is only for a given state to choose the next action
        :param state:
        :return:
        """
        pass

    @abstractmethod
    def update(self, state, action, reward, state_next):
        """
        Once the state, action, reward, and state_next are given, we can update our agent
        :param state:
        :param action:
        :param reward:
        :param state_next:
        :return:
        """
        pass


