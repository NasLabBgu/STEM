from abc import ABC, abstractmethod
from typing import Set, Tuple


class BaseStanceClassifier(ABC):

    @abstractmethod
    def classify_stance(self, op: str):
        return NotImplemented

    @abstractmethod
    def set_input(*args, **kwargs):
        return NotImplemented

    @abstractmethod
    def clear(self):
        return NotImplemented

    @abstractmethod
    def draw(self):
        return NotImplemented

    @abstractmethod
    def get_supporters(self) -> Set[str]:
        return NotImplemented

    @abstractmethod
    def get_complement(self) -> Set[str]:
        return NotImplemented

    @abstractmethod
    def get_cut(self) -> Set[Tuple[str, str]]:
        return NotImplemented
