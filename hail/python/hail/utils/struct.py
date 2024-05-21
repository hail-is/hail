import pprint
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Dict

from hail.typecheck import anytype, typecheck, typecheck_method
from hail.utils.misc import get_nice_attr_error, get_nice_field_error


class Struct(Mapping):
    """
    Nested annotation structure.

    >>> bar = hl.Struct(**{'foo': 5, '1kg': 10})

    Struct elements are treated as both 'items' and 'attributes', which
    allows either syntax for accessing the element "foo" of struct "bar":

    >>> bar.foo
    >>> bar['foo']

    Field names that are not valid Python identifiers, such as fields that
    start with numbers or contain spaces, must be accessed with the latter
    syntax:

    >>> bar['1kg']

    The ``pprint`` module can be used to print nested Structs in a more
    human-readable fashion:

    >>> from pprint import pprint
    >>> pprint(bar)

    Parameters
    ----------
    attributes
        Field names and values.

    Note
    ----
    This object refers to the Python value returned by taking or collecting
    Hail expressions, e.g. ``mt.info.take(5)``. This is rare; it is much
    more common to manipulate the :class:`.StructExpression` object, which is
    constructed using the :func:`.struct` function.
    """

    def __init__(self, **kwargs):
        # Set this way to avoid an infinite recursion in `__getattr__`.
        self.__dict__["_fields"] = kwargs

    def __contains__(self, item):
        return item in self._fields

    def __getstate__(self) -> Dict[str, Any]:
        return self._fields

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__["_fields"] = state

    def _get_field(self, item):
        if item in self._fields:
            return self._fields[item]
        else:
            raise KeyError(get_nice_field_error(self, item))

    @typecheck_method(item=str)
    def __getitem__(self, item):
        return self._get_field(item)

    def __setattr__(self, key, value):
        if key in self._fields:
            raise ValueError("Structs are immutable, cannot overwrite a field.")
        else:
            super().__setattr__(key, value)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif item in self._fields:
            return self._fields[item]
        else:
            raise AttributeError(get_nice_attr_error(self, item))

    def __len__(self):
        return len(self._fields)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if all(k.isidentifier() for k in self._fields):
            return 'Struct(' + ', '.join(f'{k}={v!r}' for k, v in self._fields.items()) + ')'
        return 'Struct(**{' + ', '.join(f'{k!r}: {v!r}' for k, v in self._fields.items()) + '})'

    def __eq__(self, other):
        return self._fields == other._fields if isinstance(other, Struct) else NotImplemented

    def __hash__(self):
        return 37 + hash(tuple(sorted(self._fields.items())))

    def __iter__(self):
        return iter(self._fields)

    def __dir__(self):
        super_dir = super().__dir__()
        return super_dir + list(self._fields.keys())

    def annotate(self, **kwargs):
        """Add new fields or recompute existing fields.

        Notes
        -----
        If an expression in `kwargs` shares a name with a field of the
        struct, then that field will be replaced but keep its position in
        the struct. New fields will be appended to the end of the struct.

        Parameters
        ----------
        kwargs : keyword args
            Fields to add.

        Returns
        -------
        :class:`.Struct`
            Struct with new or updated fields.

        Examples
        --------

        Define a Struct `s`

        >>> s = hl.Struct(food=8, fruit=5)

        Add a new field to `s`

        >>> s.annotate(bar=2)
        Struct(food=8, fruit=5, bar=2)

        Add multiple fields to `s`

        >>> s.annotate(banana=2, apple=3)
        Struct(food=8, fruit=5, banana=2, apple=3)

        Recompute an existing field in `s`

        >>> s.annotate(bar=4, fruit=2)
        Struct(food=8, fruit=2, bar=4)
        """
        d = OrderedDict(self.items())
        for k, v in kwargs.items():
            d[k] = v
        return Struct(**d)

    @typecheck_method(fields=str, kwargs=anytype)
    def select(self, *fields, **kwargs):
        """Select existing fields and compute new ones.

        Notes
        -----
        The `fields` argument is a list of field names to keep. These fields
        will appear in the resulting struct in the order they appear in
        `fields`.

        The `kwargs` arguments are new fields to add.

        Parameters
        ----------
        fields : varargs of :class:`str`
            Field names to keep.
        named_exprs : keyword args
            New field.

        Returns
        -------
        :class:`.Struct`
            Struct containing specified existing fields and computed fields.

        Examples
        --------
        Define a Struct 's'

        >>> s = hl.Struct(foo=5, apple=10)

        Keep just one original field

        >>> s.select('foo')
        Struct(foo=5)

        Add one new field and keeps one old field

        >>> s.select('apple', bar=123)
        Struct(apple=10, bar=123)

        Adds two new fields and replaces old fields

        >>> s.select(bar=123, banana=1)
        Struct(bar=123, banana=1)

        """
        d = OrderedDict()
        for a in fields:
            d[a] = self[a]
        for k, v in kwargs.items():
            if k in d:
                raise ValueError("Cannot select and assign field '{}' in the same statement".format(k))
            d[k] = v
        return Struct(**d)

    @typecheck_method(args=str)
    def drop(self, *args):
        """Drop fields from the struct.

        Parameters
        ----------
        fields: varargs of :class:`str`
            Fields to drop.

        Returns
        -------
        :class:`.Struct`
            Struct without certain fields.

        Examples
        --------

        Define a Struct `s`

        >>> s = hl.Struct(food=8, fruit=5, bar=2, apple=10)

        Drop one field from `s`

        >>> s.drop('bar')
        Struct(food=8, fruit=5, apple=10)

        Drop two fields from `s`

        >>> s.drop('food', 'fruit')
        Struct(bar=2, apple=10)
        """
        d = OrderedDict((k, v) for k, v in self.items() if k not in args)
        return Struct(**d)


@typecheck(struct=Struct)
def to_dict(struct):
    return dict(struct.items())


_old_printer = pprint.PrettyPrinter


class StructPrettyPrinter(pprint.PrettyPrinter):
    def _format(self, obj, stream, indent, allowance, context, level, *args, **kwargs):
        if isinstance(obj, Struct):
            rep = self._repr(obj, context, level)
            max_width = self._width - indent - allowance
            if len(rep) <= max_width:
                stream.write(rep)
                return

            stream.write('Struct(')
            indent += len('Struct(')
            if all(k.isidentifier() for k in obj):
                n = len(obj.items())
                for i, (k, v) in enumerate(obj.items()):
                    is_first = i == 0
                    is_last = i == n - 1

                    if not is_first:
                        stream.write(' ' * indent)
                    stream.write(k)
                    stream.write('=')
                    this_indent = indent + len(k) + len('=')
                    self._format(v, stream, this_indent, allowance, context, level, *args, **kwargs)
                    if not is_last:
                        stream.write(',\n')
            else:
                stream.write('**{')
                indent += len('**{')
                n = len(obj.items())
                for i, (k, v) in enumerate(obj.items()):
                    is_first = i == 0
                    is_last = i == n - 1

                    if not is_first:
                        stream.write(' ' * indent)
                    stream.write(repr(k))
                    stream.write(': ')
                    this_indent = indent + len(repr(k)) + len(': ')
                    self._format(v, stream, this_indent, allowance, context, level, *args, **kwargs)
                    if not is_last:
                        stream.write(',\n')
                stream.write('}')
            stream.write(')')
        else:
            _old_printer._format(self, obj, stream, indent, allowance, context, level, *args, **kwargs)


pprint.PrettyPrinter = StructPrettyPrinter  # monkey-patch pprint
