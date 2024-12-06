import logging
from typing import Any, Callable, Generator, List, Sequence, TypeVar, Union

A = TypeVar('A')


def chunk(size: int, seq: Sequence[A]) -> Generator[Sequence[A], A, Any]:
    for pos in range(0, len(seq), size):
        yield seq[pos : pos + size]


B = TypeVar('B')


def maybe(f: Callable[[A], B], ma: Union[A, None], default: Union[B, None] = None) -> Union[B, None]:
    return f(ma) if ma is not None else default


def prune(kvs: dict) -> dict:
    return {k: v for k, v in kvs.items() if v is not None}


def select(keys: List[str], **kwargs) -> List[Union[Any, None]]:
    return [kwargs.get(k) for k in keys]


def init_logging(file=None):
    logging.basicConfig(format="%(asctime)-15s: %(levelname)s: %(message)s", level=logging.INFO, filename=file)
