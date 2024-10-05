from dataclasses import dataclass, fields
from functools import wraps
from typing import Any, Callable, TypeVar, Union

from typeguard import check_type

ReturnType = TypeVar("ReturnType")
WrappedDecorator = Callable[[ReturnType], ReturnType]


def typeguard_dataclass(cls: ReturnType = None, /, **kwargs: Any) -> Union[ReturnType, WrappedDecorator]:
    """
    Creates a `dataclass` that is `frozen` by default and has runtime typechecking for its fields.
    """

    @wraps(dataclass)
    def wrapper(cls: ReturnType) -> ReturnType:
        def __setattr__(obj: ReturnType, name: str, value: Any) -> None:
            if len(types := [_field.type for _field in fields(obj) if _field.name == name]) == 0:
                raise TypeError(f"'{getattr(cls, '__name__', str(cls))}' has no field '{name}'.")
            super().__setattr__(name, check_type(value, types[0]))

        def __post_init__(obj: ReturnType) -> None:
            for _field in fields(obj):
                check_type(getattr(obj, _field.name), _field.type)

        setattr(
            cls,
            *(
                ["__post_init__", __post_init__]
                if (frozen := kwargs.get("frozen", True))
                else ["__setattr__", __setattr__]
            ),
        )
        dataclass(cls, frozen=frozen, **{k: v for k, v in kwargs.items() if k != "frozen"})
        return cls

    return wrapper if cls is None else wrapper(cls)
