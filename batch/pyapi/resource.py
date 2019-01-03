class Resource(object):
    _counter = 0
    _uid_prefix = "__RESOURCE__"
    _regex_pattern = r"(?P<RESOURCE>{}\d+)".format(_uid_prefix)

    @classmethod
    def _new_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, source=None, value=None):
        from .task import Task
        assert isinstance(source, Task) or source is None
        assert value is None or isinstance(value, str)
        self._value = value
        self._source = source
        self._uid = Resource._new_uid()

    def __str__(self):
        return self._uid


class ResourceGroup(object):
    _counter = 0
    _uid_prefix = "__RESOURCE_GROUP__"
    _regex_pattern = r"(?P<RESOURCE_GROUP>{}\d+)".format(_uid_prefix)

    @classmethod
    def _new_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, root, **values):
        self._namespace = {}
        self._root = root
        self._uid = ResourceGroup._new_uid()

        for name, resource in values.items():
            assert isinstance(resource, Resource)
            self._namespace[name] = resource

    def _get_resource(self, item):
        if item not in self._namespace:
            raise KeyError(f"'{item}' not found in the resource group. Hint: you must declare each attribute when constructing the resource group.")
        r = self._namespace[item]
        return r._uid

    def __getitem__(self, item):
        return self._get_resource(item)

    def __getattr__(self, item):
        return self._get_resource(item)

    def __add__(self, other):
        assert isinstance(other, str)
        return str(self._uid) + other

    def __radd__(self, other):
        assert isinstance(other, str)
        return other + str(self._uid)

    def __str__(self):
        return self._uid


class ResourceGroupBuilder(object):
    def __init__(self, **mappings):
        def f(root):
            output = {}
            for name, code in mappings.items():
                output[name] = eval(f'f"""{code}"""')
            return output
        self._f = f


def resource_group_builder(**mappings):
    return ResourceGroupBuilder(**mappings)