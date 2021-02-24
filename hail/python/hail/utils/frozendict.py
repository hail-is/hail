from typing import TypeVar, Dict
from collections.abc import Mapping

T = TypeVar("T")
U = TypeVar("U")

class frozendict(Mapping):
    def __init__(self, d: Dict[T, U]):
        self.d = d.copy()

    def __getitem__(self, k: T) -> U:
        return self.d[k]

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))

    def __len__(self) -> int:
        return len(self.d)

    def __iter__(self):
        return iter(self.d)
