from typing import TypeVar, Callable, cast, Protocol
from typing_extensions import ParamSpec
from decorator import decorator as _decorator

P = ParamSpec('P')
T = TypeVar('T')


class Wrapper(Protocol[P, T]):
    def __call__(self, fun: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        ...


def decorator(fun: Wrapper[P, T]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    return cast(Callable[[Callable[P, T]], Callable[P, T]], _decorator(fun))
