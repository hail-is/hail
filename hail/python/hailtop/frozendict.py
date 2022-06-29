from typing import TypeVar, Dict, Generic
from collections.abc import Mapping

T = TypeVar("T")
U = TypeVar("U")


class frozendict(Mapping, Generic[T, U]):
    """
    An object representing an immutable dictionary.

    >>> my_frozen_dict = hl.utils.frozendict({1:2, 7:5})

    To get a normal python dictionary with the same elements from a `frozendict`:

    >>> dict(frozendict({'a': 1, 'b': 2}))

    Note
    ----
    This object refers to the Python value returned by taking or collecting
    Hail expressions, e.g. ``mt.my_dict.take(5)``. This is rare; it is much
    more common to manipulate the :class:`.DictExpression` object, which is
    constructed using :func:`.dict`. This class is necessary because hail
    supports using dicts as keys to other dicts or as elements in sets, while
    python does not.

    """
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

    def __repr__(self):
        return f'frozendict({self.d!r})'
