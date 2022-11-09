from functools import wraps
from inspect import getmodule, signature, stack
from typing import _SpecialForm, Any, ForwardRef, get_args, get_origin, get_type_hints, List, Type, Union

from .typing import ReturnType, TypecheckedFunc, TypecheckedForwardRef


def typecheck(func: TypecheckedFunc) -> TypecheckedFunc:
    """
    Provides runtime typechecking for functions using their type annotations. Adapted from
    https://stackoverflow.com/questions/50563546/validating-detailed-types-in-python-dataclasses.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> ReturnType:
        for arg, val in {**dict(zip(signature(func).parameters, args)), **kwargs}.items():
            typecheck_val(func, arg, val, get_type_hints(func).get(arg, Any))
        return func(*args, **kwargs)

    return wrapper


def typecheck_val(obj: Any, name: str, val: Any, typ: Type) -> None:
    if len(types := reduce_type(typ)) > 0 and not isinstance(val, tuple(types)):
        actual_type, *expected_types = [f"'{typename(typ)}'" for typ in [type(val), *types]]
        raise TypeError(
            f"{getattr(obj, '__name__', obj)}: Argument '{name}' with value '{val}' is expected to be of type "
            f"{' or '.join(expected_types)}, but is of type {actual_type}."
        )


def typename(typ: Type):
    return getattr(typ, '__name__', str(typ))


def reduce_type(typ: Type) -> List[Type]:
    if isinstance(typ, _SpecialForm):
        return []
    origin = get_origin(typ)
    if origin is None:
        if isinstance(typ, TypecheckedForwardRef):
            return [getattr(typ.module, typ.__forward_arg__)]
        if isinstance(typ, ForwardRef):
            raise TypeError(
                f"{typ} is a 'ForwardRef', but only 'TypecheckForwardRef's are supported by the typecheck decorator."
            )
        return [typ]
    if not isinstance(origin, _SpecialForm):
        return [origin]
    types = []
    for arg in get_args(typ):

        """
        See https://docs.python.org/3/library/typing.html#typing.get_type_hints. In Python versions lower than 3.11,
        `get_type_hints` assumes that if a function argument has a default value of `None` and a type hint of type `A`,
        the type hint should be transformed to `Optional[A]`. This is true even if `A = Any`, leading to an edge case
        where the nonsensical type hint `Optional[Any]` is occasionally encountered.
        """
        # FIXME: if hail no longer supports any python version below 3.11, the next two lines can be safely removed.
        if origin is Union and arg is Any:
            return []

        types.extend(reduce_type(arg))
    return types


def typechecked_forward_ref(typename: str) -> TypecheckedForwardRef:
    return TypecheckedForwardRef(typename, getmodule(stack()[1][0]))
