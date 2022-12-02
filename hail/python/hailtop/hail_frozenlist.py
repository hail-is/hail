from typing import TypeVar, Sequence, List
from frozenlist import FrozenList as _FrozenList


T = TypeVar('T')


class frozenlist(_FrozenList, Sequence[T]):
    def __init__(self, items: List[T]):
        super().__init__(items)
        self.freeze()
