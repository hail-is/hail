class Struct(object):
    """
    Nested annotation structure.

    :param dict attributes: struct members, must contain every field in ``fields`` as a key.
    :param fields: list of field names, used for ordering
    :type fields: list of str

    :ivar fields: ordered list of fields
    :vartype fields: list of str
    """

    def __init__(self, attributes, fields):
        if not len(fields) == len(attributes):
            raise ValueError('length of fields and size of attributes is not the same: %d and %d' %
                             (len(fields), len(attributes)))
        self._attrs = attributes
        self.fields = fields

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
        return len(self.fields)

    def __str__(self):
        return 'Struct{%s}' % ', '.join(["'%s': %s" % (f, self._attrs[f]) for f in self.fields])

    def __repr__(self):
        return '{%s}' % ', '.join(["'%s':%s" % (f, repr(self._attrs[f])) for f in self.fields])

    def __eq__(self, other):
        return self.fields == other.fields and self._attrs == other._attrs
