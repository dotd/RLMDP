from abc import ABC, ABCMeta, abstractmethod

class AgentBase(ABC):

    @abstractmethod
    def choose_action(self):
        pass


