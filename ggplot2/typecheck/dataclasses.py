from dataclasses import dataclass, fields
from functools import wraps

from typing import Any, Union

from .typecheck import typecheck_val, typename
from .typing import ReturnType, WrappedDecorator


def typechecked_dataclass(cls: ReturnType = None, /, **kwargs: Any) -> Union[ReturnType, WrappedDecorator]:
    """
    Creates a `dataclass` that is `frozen` by default and has runtime typechecking for its fields.
    """

    @wraps(dataclass)
    def wrapper(cls: ReturnType) -> ReturnType:
        def __typechecked_setattr__(obj: type(cls), name: str, value: Any) -> None:
            if len(types := [field.type for field in fields(obj) if field.name == name]) == 0:
                raise TypeError(f"'{typename(cls)}' has no field '{name}'.")
            typecheck_val(obj, name, value, types[0])
            super().__setattr__(name, value)

        def __typechecked_post_init__(obj: type(cls)) -> None:
            for field in fields(obj):
                typecheck_val(obj, field.name, getattr(obj, field.name), field.type)

        setattr(
            cls,
            *(
                ["__post_init__", __typechecked_post_init__]
                if (frozen := kwargs.get("frozen", True))
                else ["__setattr__", __typechecked_setattr__]
            ),
        )
        dataclass(cls, frozen=frozen, **{k: v for k, v in kwargs.items() if k != "frozen"})
        return cls

    return wrapper if cls is None else wrapper(cls)


def pformat(obj: Any, indent: int = 1) -> str:
    indent_width = "    "
    deindent_str = indent_width * (indent - 1)
    indent_str = indent_width * indent
    if getattr(obj, "__dataclass_fields__", None) is not None:
        inner = f",\n{indent_str}".join(
            [
                f"{field.name} = {pformat(getattr(obj, field.name), indent + 1)}"
                for field in fields(obj)
                if not field.name.startswith("_")
            ]
        )
        return f"{obj.__class__.__name__}(\n{indent_str}{inner}\n{deindent_str})"
    if isinstance(obj, list) and len(obj) > 0:
        inner = f",\n{indent_str}".join([pformat(inner_obj, indent + 1) for inner_obj in obj])
        return f"[\n{indent_str}{inner}\n{deindent_str}]"
    return str(obj)


def pprint(obj: Any) -> None:
    print(pformat(obj))
