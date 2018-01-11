from hail.typecheck import *
from hail.history import HistoryMixin, record_init, record_method
from collections import Mapping

class Struct(Mapping, HistoryMixin):
    """
    Nested annotation structure.

    >>> bar = Struct(**{'foo': 5, '1kg': 10})

    Struct elements are treated as both 'items' and 'attributes', which
    allows either syntax for accessing the element "foo" of struct "bar":

    >>> bar.foo
    >>> bar['foo']

    Note that it is possible to use Hail to define struct fields inside
    of a key table or variant dataset that do not match python syntax.
    The name "1kg", for example, will not parse to python because it
    begins with an integer, which is not an acceptable leading character
    for an identifier.  There are two ways to access this field:

    >>> getattr(bar, '1kg')
    >>> bar['1kg']

    The ``pprint`` module can be used to print nested Structs in a more
    human-readable fashion:

    >>> from pprint import pprint
    >>> pprint(bar)

    :param dict attributes: struct members.
    """

    @record_init
    def __init__(self, **kwargs):
        self._fields = kwargs
        for k, v in kwargs.items():
            if not k in self.__dict__:
                self.__dict__[k] = v

    def __contains__(self, item):
        return item in self._fields

    def __getitem__(self, item):
        if not item in self._fields:
            raise KeyError("Struct has no field '{}'\n"
                           "    Fields: [ {} ]".format(item, ', '.join("'{}'".format(x) for x in self._fields)))
        return self._fields[item]

    def __len__(self):
        return len(self._fields)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Struct({})'.format(', '.join('{}={}'.format(k, v) for k, v in self._fields.items()))

    def __eq__(self, other):
        return isinstance(other, Struct) and self._fields == other._fields

    def __hash__(self):
        return 37 + hash(tuple(sorted(self._fields.items())))

    def __iter__(self):
        return iter(self._fields)

@typecheck(struct=Struct)
def to_dict(struct):
    d = {}
    for k, v in struct._attrs.items():
        if isinstance(v, Struct):
            d[k] = to_dict(v)
        else:
            d[k] = v
    return d


import pprint

_old_printer = pprint.PrettyPrinter


class StructPrettyPrinter(pprint.PrettyPrinter):
    def _format(self, obj, *args, **kwargs):
        if isinstance(obj, Struct):
            obj = to_dict(obj)
        return _old_printer._format(self, obj, *args, **kwargs)


pprint.PrettyPrinter = StructPrettyPrinter  # monkey-patch pprint
