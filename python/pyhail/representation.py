
class Struct(object):
    """Nested annotation structure"""
    def __init__(self, attributes):
        self._attrs = attributes

    def __getattribute__(self, item):
        return self._attrs[item]