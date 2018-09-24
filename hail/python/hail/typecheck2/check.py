import sys
import inspect
from typing import *
import collections

implicit_conversions: Dict[type, Set[type]] = {}

warned_for: Set[type] = set()


def qualified_name(t):
    return qualified_type(type(t))


def qualified_type(t):
    module = t.__module__
    if module == 'builtins':
        return t.__name__
    elif module == 'typing':
        return str(t).replace('typing.', '')  # HACKY HACK
    else:
        return t.__qualname__


class TypeCheckFailure(Exception):
    __slots__ = ['msg']

    def __init__(self, msg: str):
        self.msg = msg


def get_signature(f) -> inspect.Signature:
    if hasattr(f, '__memo'):
        return f.__memo
    else:
        signature = inspect.signature(f)
        f.__memo = signature
        return signature


def typecheck(f: Callable):
    """Typecheck a function's arguments against its type annotations.

    Examples
    --------
    >>> def f(i: int, s: str) -> str:
    ...     typecheck(f)
    ...     return i * s

    Notes
    -----
    Functions should call this function with themselves as an argument. This
    is required to look up the function signature without hacking around in
    the garbage collector and frames too deeply.

    Not all types in :mod:`typing` are supported at this time.

    Parameters
    ----------
    f
        Function to check.

    Raises
    ------
    TypeError
    """
    try:
        frame = inspect.currentframe().f_back
        args = frame.f_locals
        spec = get_signature(f)
        for arg_name, param in spec.parameters.items():
            assert isinstance(param, inspect.Parameter)
            t = param.annotation
            if t:
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                    check_t(args[arg_name], t, f"argument {repr(arg_name)} of {f.__qualname__}")
                elif param.kind == param.VAR_POSITIONAL:
                    varargs = args[arg_name]
                    assert isinstance(varargs, tuple)
                    for i, value in enumerate(varargs):
                        check_t(value, t, f"element {i} of argument {repr(arg_name)} of {f.__qualname__}")
                else:
                    assert param.kind == param.VAR_KEYWORD
                    varkw = args[arg_name]
                    assert isinstance(varkw, collections.Mapping)
                    for name, value in varkw.items():
                        check_t(value, t, f"keyword argument {repr(arg_name)} of {f.__qualname__}")
    finally:
        del frame
        del args

def register_conversion(from_type: type, to_type: type) -> None:
    """Register a recognized conversion from type to type.

    Examples
    --------
    >>> register_conversion(int, float)

    Notes
    -----
    The only self-initialized conversion is from int to float.

    Parameters
    ----------
    from_type : type
    to_type : type
    """
    if to_type not in implicit_conversions:
        implicit_conversions[to_type] = {from_type}
    else:
        implicit_conversions[to_type].add(from_type)


def check_union(arg, t, context: str) -> None:
    cast(t, Union)
    union_params = t.__args__

    for t_ in union_params:
        try:
            _check_t(arg, t_, context)
            return
        except TypeCheckFailure:
            pass

    options = ', '.join(qualified_type(t_) for t_ in union_params)
    raise TypeCheckFailure(f"expected ({options}); found '{qualified_name(arg)}' at {context}: {repr(arg)}")


def _check_collection(arg: collections.Collection, element_type, context: str, indexed: bool) -> None:
    if indexed:
        for i, v in enumerate(arg):
            _check_t(v, element_type, f'element {i} of {context}')
    else:
        for v in arg:
            _check_t(v, element_type, f'element of {context}')


def check_list(arg, t, context: str) -> None:
    cast(t, List)
    if not isinstance(arg, list):
        raise TypeCheckFailure(f"expected 'list', found '{qualified_name(arg)}' at {context}: {repr(arg)}")

    if t is List:
        return
    element_type = getattr(t, '__args__', t.__parameters__)[0]
    _check_collection(arg, element_type, context, indexed=True)


def check_sequence(arg, t, context: str) -> None:
    cast(t, Sequence)
    if not isinstance(arg, collections.Sequence):
        raise TypeCheckFailure(f"expected 'Sequence', found '{qualified_name(arg)}' at {context}: {repr(arg)}")

    if t is Sequence:
        return
    element_type = getattr(t, '__args__', t.__parameters__)[0]
    _check_collection(arg, element_type, context, indexed=True)


def check_set(arg, t, context: str) -> None:
    cast(t, Set)
    if not isinstance(arg, set):
        raise TypeCheckFailure(f"expected 'set', found '{qualified_name(arg)}' at {context}: {repr(arg)}")

    if t is Set:
        return
    element_type = getattr(t, '__args__', t.__parameters__)[0]
    _check_collection(arg, element_type, context, indexed=False)


def check_frozenset(arg, t, context: str) -> None:
    cast(t, FrozenSet)
    if not isinstance(arg, frozenset):
        raise TypeCheckFailure(f"expected 'frozenset', found '{qualified_name(arg)}' at {context}: {repr(arg)}")

    if t is FrozenSet:
        return
    element_type = getattr(t, '__args__', t.__parameters__)[0]
    _check_collection(arg, element_type, context, indexed=False)


def _check_mapping(arg: collections.Mapping, key_type, value_type, context: str) -> None:
    for k, v in arg.items():
        _check_t(k, key_type, f'key of {context}')
        _check_t(v, value_type, f'value for key {repr(k)} of {context}')


def check_dict(arg, t, context: str) -> None:
    cast(t, Dict)
    if not isinstance(arg, dict):
        raise TypeCheckFailure(f"expected 'dict', found '{qualified_name(arg)}' at {context}: {repr(arg)}")

    if t is Dict:
        return
    key_type, value_type = getattr(t, '__args__', t.__parameters__)
    _check_mapping(arg, key_type, value_type, context)


def check_mapping(arg, t, context: str) -> None:
    cast(t, Mapping)
    if not isinstance(arg, collections.Mapping):
        raise TypeCheckFailure(f"expected 'Mapping', found '{qualified_name(arg)}' at {context}: {repr(arg)}")

    if t is Mapping:
        return
    key_type, value_type = getattr(t, '__args__', t.__parameters__)
    _check_mapping(arg, key_type, value_type, context)


def check_tuple(arg, t, context: str) -> None:
    cast(t, Tuple)
    if not isinstance(arg, tuple):
        raise TypeCheckFailure(f"expected 'tuple', found '{qualified_name(arg)}' at {context}: {repr(arg)}")

    if t is Tuple:
        return
    types = t.__args__
    if types[-1] is Ellipsis:
        # variable-length tuple
        element_type = types[0]
        _check_collection(arg, element_type, context, indexed=True)
    else:
        if len(arg) != len(types):
            types_str = ', '.join(map(qualified_type, types))
            raise TypeCheckFailure(f"expected tuple with {len(types)} elements of types {types_str}, "
                                   f"found {len(arg)} elements at {context}")
        for i, element in enumerate(arg):
            _check_t(element, types[i], f"element{i} of {context}")


def check_callable(arg, t, context: str) -> None:
    cast(t, Callable)
    if not callable(arg):
        raise TypeCheckFailure(f'expected callable {qualified_type(t)}, found not callable at {context}')
    if t is Callable:
        return

    if isinstance(t.__args__, tuple):
        argument_types = t.__args__[:-1]
        check_args = argument_types != (Ellipsis,)

        if check_args:
            signature = inspect.signature(arg)

            parameter_types = list(p.kind for p in signature.parameters.values())
            if any(kind == inspect.Parameter.KEYWORD_ONLY for kind in parameter_types):
                raise TypeCheckFailure(f"found keyword-only arguments on callable signature at {context}")
            elif any(kind == inspect.Parameter.VAR_POSITIONAL for kind in parameter_types):
                raise TypeCheckFailure(f"found variable-length args on callable signature at {context}")
            elif any(kind == inspect.Parameter.VAR_KEYWORD for kind in parameter_types):
                raise TypeCheckFailure(f"found variable keyword arguments on callable signature at {context}")
            elif len(parameter_types) != len(argument_types):
                raise TypeCheckFailure(f"expected {len(argument_types)}-parameter callable, "
                                       f"found {len(parameter_types)}-parameter callable at {context}")


def check_collection(arg, t, context: str) -> None:
    if not isinstance(arg, collections.Collection):
        raise TypeCheckFailure(f"expected 'Collection', found '{qualified_name(arg)}' at {context}: {repr(arg)}")
    if t is Collection:
        return
    else:
        return _check_collection(arg, t.__args__[0], context, indexed=False)


def check_t(arg, t, context: str) -> None:
    # the following motif is used to squelch the original (deeply recursive)
    # stack trace
    try:
        _check_t(arg, t, context)
    except TypeCheckFailure as e:
        raise TypeError(e.msg) from None


known_checkers = {
    Union: check_union,
    Dict: check_dict,
    Mapping: check_mapping,
    Sequence: check_sequence,
    Set: check_set,
    FrozenSet: check_frozenset,
    List: check_list,
    Tuple: check_tuple,
    Collection: check_collection,
    Callable: check_callable
}


def _check_t(arg, t, context: str) -> None:
    if t is Any:
        return

    # typing.Type subtypes have the '__origin__' attribute set
    typing_type = getattr(t, '__origin__', None)
    if typing_type is not None:
        f = known_checkers.get(typing_type)
        if f:
            f(arg, t, context)
        else:
            if typing_type not in warned_for:
                sys.stderr.write(f"WARN: typecheck: type '{typing_type}' is not currently supported\n")
                warned_for.add(typing_type)
    else:
        assert isinstance(t, type)
        if not isinstance(arg, t):
            if t not in implicit_conversions or all(
                    not isinstance(arg, compatible_type) for compatible_type in implicit_conversions[t]):
                raise TypeCheckFailure(f"expected type '{qualified_type(t)}', "
                                       f"found '{qualified_type(type(arg))}' at {context}: {repr(arg)}")


register_conversion(int, float)
