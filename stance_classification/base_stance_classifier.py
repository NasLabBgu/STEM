from abc import ABC, abstractmethod


class BaseStanceClassifier(ABC):

    @abstractmethod
    def classify_stance(self, op: str):
        return NotImplemented

    @abstractmethod
    def clear(self):
        return NotImplemented

    @abstractmethod
    def draw(self):
        return NotImplemented


