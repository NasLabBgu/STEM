from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Set, Tuple, Any, Dict, Sequence

import networkx as nx


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
