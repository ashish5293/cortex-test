from abc import ABC, abstractmethod


class BaseUpdateService(ABC):

    @abstractmethod
    def update(self):
        pass
