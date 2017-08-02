from hail.typecheck import *
from hail.history import *


class Struct(HistoryMixin):
    """
    Nested annotation structure.

    >>> bar = Struct({'foo': 5, '1kg': 10})

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
    def __init__(self, attributes):
        self._attrs = attributes
        super(Struct, self).__init__()

    def __getattr__(self, item):
        assert (self._attrs)
        if item not in self._attrs:
            raise AttributeError("Struct instance has no attribute '%s'" % item)
        return self._attrs[item]

    def __contains__(self, item):
        return item in self._attrs

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __len__(self):
        return len(self._attrs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Struct' + str(self._attrs)

    def __eq__(self, other):
        if isinstance(other, Struct):
            return self._attrs == other._attrs
        else:
            return False

    def __hash__(self):
        return 37 + hash(tuple(sorted(self._attrs.items())))

    @record_method
    @typecheck_method(item=strlike,
                      default=anytype)
    def get(self, item, default=None):
        """Get an item, or return a default value if the item is not found.
        
        :param str item: Name of attribute.
        
        :param default: Default value.
        
        :returns: Value of item if found, or default value if not.
        """
        return self._attrs.get(item, default)


@typecheck(struct=Struct)
def to_dict(struct):
    d = {}
    for k, v in struct._attrs.iteritems():
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
