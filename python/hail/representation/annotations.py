from hail.java import strip_option


class Struct(object):
    """Nested annotation structure"""

    def __init__(self, hail_type, attributes):
        self._type = hail_type
        self._attrs = attributes

    def __getattr__(self, item):
        idx = strip_option(self._jtype.index(item))
        if not idx:
            raise AttributeError("Struct instance has no attribute '%s'" % item)
        return self._attrs[item]

    def __getitem__(self, item):
        self.__getattr__(item)

    def __len__(self):
        return len(self._attrs)

    def __str__(self):
        return self._jtype.str(self._jrow)

    def __repr__(self):
        return self.__repr__()
