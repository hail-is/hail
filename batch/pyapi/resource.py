class Resource(object):
    _counter = 0

    def __init__(self, source=None, value=None):
        self._value = value
        self._source = source
        self._uid = "__RESOURCE__{}".format(Resource._counter)
        Resource._counter += 1

    def __add__(self, other):
        assert isinstance(other, str)
        if isinstance(self._value, str):
            return Resource(self._source, str(self._value) + other)
        else:
            print(self._value)
            assert(isinstance(self._value, list))
            return Resource(self._source, [x + other for x in self._value])

    def __radd__(self, other):
        assert isinstance(other, str)
        if isinstance(self._value, str):
            return Resource(self._source, other + str(self._value))
        else:
            assert(isinstance(self._value, list))
            return Resource(self._source, [other + x for x in self._value])

    def __str__(self):
        return f"Resource(_source={self._source},_uid={self._uid}, _value={self._value})"


