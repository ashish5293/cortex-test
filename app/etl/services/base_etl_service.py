from abc import ABC, abstractmethod


class BaseEtlService(ABC):

    @abstractmethod
    def extract(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass
