import abc

from .utils import escape_string


class Resource:
    _uid: str

    @property
    @abc.abstractmethod
    def file_name(self) -> str:
        pass

    def declare(self, directory=None):
        directory = directory + '/' if directory else ''
        return f"{self._uid}={escape_string(directory + self.file_name)}"


class ResourceFile(Resource, str):
    _counter = 0
    _uid_prefix = "__RESOURCE_FILE__"
    _regex_pattern = r"(?P<RESOURCE_FILE>{}\d+)".format(_uid_prefix)

    @classmethod
    def _new_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, value):
        super(ResourceFile, self).__init__()
        assert value is None or isinstance(value, str)
        self._value = value
        self._source = None
        self._uid = ResourceFile._new_uid()
        self._output_paths = set()

    def add_source(self, source):
        from .task import Task
        assert isinstance(source, Task)
        self._source = source
        return self

    def add_output_path(self, path):
        self._output_paths.add(path)

    @property
    def file_name(self):
        assert self._value is not None
        return self._value

    def __str__(self):
        return self._uid


class InputResourceFile(ResourceFile):
    def __init__(self, value):
        self._input_path = None
        super().__init__(value)

    def add_input_path(self, path):
        self._input_path = path
        return self


class TaskResourceFile(ResourceFile):
    pass


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
        self._resources = {}  # dict of name to resource uid
        self._root = root
        self._uid = ResourceGroup._new_uid()
        self._output_paths = set()

        for name, resource in values.items():
            self._resources[name] = resource

    @property
    def file_name(self):
        return self._root

    def add_output_path(self, path):
        self._output_paths.add(path)

    def _get_resource(self, item):
        if item not in self._resources:
            raise KeyError(f"'{item}' not found in the resource group.\n"
                           f"Hint: you must declare each attribute when constructing the resource group.")
        return self._resources[item]

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
