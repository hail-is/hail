class Resource:
    pass


class ResourceFile(Resource):
    _counter = 0
    _uid_prefix = "__RESOURCE_FILE__"
    _regex_pattern = r"(?P<RESOURCE_FILE>{}\d+)".format(_uid_prefix)

    @classmethod
    def _new_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, source, value):
        from .task import Task
        assert isinstance(source, Task) or source is None
        assert value is None or isinstance(value, str)
        self._value = value
        self._source = source
        self._uid = ResourceFile._new_uid()
        self._output_paths = set()

    def add_output_path(self, path):
        self._output_paths.add(path)

    def __str__(self):
        return self._uid


class InputResourceFile(ResourceFile):
    def __init__(self, input_path, value):
        self._input_path = input_path
        super().__init__(None, value)


class TaskResourceFile(ResourceFile):
    def __init__(self, source, value):
        super().__init__(source, value)


class ResourceGroup(Resource):
    _counter = 0
    _uid_prefix = "__RESOURCE_GROUP__"
    _regex_pattern = r"(?P<RESOURCE_GROUP>{}\d+)".format(_uid_prefix)

    @classmethod
    def _new_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, source, root, **values):
        self._source = source
        self._resources = {} # dict of name to resource uid
        self._root = root
        self._uid = ResourceGroup._new_uid()
        self._output_paths = set()

        for name, resource in values.items():
            self._resources[name] = resource

    def add_output_path(self, path):
        self._output_paths.add(path)

    def _get_resource(self, item):
        if item not in self._resources:
            raise KeyError(f"'{item}' not found in the resource group.\n"
                           f"Hint: you must declare each attribute when constructing the resource group.")
        r_uid = self._resources[item]
        return r_uid

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
