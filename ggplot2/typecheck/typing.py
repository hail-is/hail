from dataclasses import dataclass
from typing import Callable, ForwardRef, TypeVar
from types import ModuleType


ReturnType = TypeVar("ReturnType")
TypecheckedFunc = Callable[..., ReturnType]
WrappedDecorator = Callable[[ReturnType], ReturnType]


class TypecheckedForwardRef(ForwardRef, _root=True):
    def __init__(self: "TypecheckedForwardRef", typename: str, module: ModuleType) -> None:
        super().__init__(typename)
        self.module = module
