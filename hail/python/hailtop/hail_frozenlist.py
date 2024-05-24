from typing import List, Sequence, TypeVar

from frozenlist import FrozenList as _FrozenList

T = TypeVar('T')


class frozenlist(_FrozenList, Sequence[T]):
    def __init__(self, items: List[T]):
        super().__init__(items)
        self.freeze()

    def __repr__(self) -> str:
        return f'frozenlist({list(self)})'
