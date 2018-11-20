from hail.typecheck import *
from collections import Mapping, OrderedDict
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
    """

    def __init__(self, **kwargs):
        self._fields = kwargs
        for k, v in kwargs.items():
            if not k in self.__dict__:
                self.__dict__[k] = v

    def __contains__(self, item):
        return item in self._fields

    def _get_field(self, item):
        if item in self._fields:
            return self._fields[item]
        else:
            raise KeyError(get_nice_field_error(self, item))

    @typecheck_method(item=str)
    def __getitem__(self, item):
        return self._get_field(item)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise AttributeError(get_nice_attr_error(self, item))

    def __len__(self):
        return len(self._fields)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Struct({})'.format(', '.join('{}={}'.format(k, repr(v)) for k, v in self._fields.items()))

    def __eq__(self, other):
        return isinstance(other, Struct) and self._fields == other._fields

    def __hash__(self):
        return 37 + hash(tuple(sorted(self._fields.items())))

    def __iter__(self):
        return iter(self._fields)

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
        fields : varargs of :obj:`str`
            Field names to keep.
        named_exprs : keyword args
            New field.

        Returns
        -------
        :class:`.Struct`
            Struct containing specified existing fields and computed fields.
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
        fields: varargs of :obj:`str`
            Fields to drop.

        Returns
        -------
        :class:`.Struct`
            Struct without certain fields.
        """
        d = OrderedDict((k, v) for k, v in self.items() if not k in args)
        return Struct(**d)


@typecheck(struct=Struct)
def to_dict(struct):
    return dict(struct.items())


import pprint

_old_printer = pprint.PrettyPrinter


class StructPrettyPrinter(pprint.PrettyPrinter):
    def _format(self, obj, *args, **kwargs):
        if isinstance(obj, Struct):
            obj = to_dict(obj)
        return _old_printer._format(self, obj, *args, **kwargs)


pprint.PrettyPrinter = StructPrettyPrinter  # monkey-patch pprint
