

class Struct(object):
    """Nested annotation structure"""

    def __init__(self, attributes):
        self._attrs = attributes

    def __getattr__(self, item):
        assert(self._attrs)
        if item not in self._attrs:
            raise AttributeError("Struct instance has no attribute '%s'" % item)
        return self._attrs[item]

    def __getitem__(self, item):
        self.__getattr__(item)

    def __len__(self):
        return len(self._attrs)

    def __str__(self):
        return str(self._attrs)

    def __repr__(self):
        return self.__str__()
