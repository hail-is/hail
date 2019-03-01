import abc

from .utils import escape_string


class Resource:
    _uid: str

    @abc.abstractmethod
    def _get_path(self, directory) -> str:
        pass

    @abc.abstractmethod
    def _add_output_path(self, path):
        pass

    def _declare(self, directory):
        return f"{self._uid}={escape_string(self._get_path(directory))}"


class ResourceFile(Resource, str):
    _counter = 0
    _uid_prefix = "__RESOURCE_FILE__"
    _regex_pattern = r"(?P<RESOURCE_FILE>{}\d+)".format(_uid_prefix)

    @classmethod
    def _new_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __new__(cls, value):  # pylint: disable=W0613
        uid = ResourceFile._new_uid()
        r = str.__new__(cls, uid)
        r._uid = uid
        return r

    def __init__(self, value):
        super(ResourceFile, self).__init__()
        assert value is None or isinstance(value, str)
        self._value = value
        self._source = None
        self._output_paths = set()
        self._resource_group = None
        self._has_extension = False

    def _get_path(self, directory):
        raise NotImplementedError

    def _add_source(self, source):
        from .task import Task
        assert isinstance(source, Task)
        self._source = source
        return self

    def _add_output_path(self, path):
        self._output_paths.add(path)
        if self._source is not None:
            self._source._add_outputs(self)

    def _add_resource_group(self, rg):
        self._resource_group = rg

    def _has_resource_group(self):
        return self._resource_group is not None

    def _get_resource_group(self):
        return self._resource_group

    def add_extension(self, extension):
        if self._has_extension:
            raise Exception("Resource already has a file extension added.")
        self._value += extension
        self._has_extension = True
        return self

    def __str__(self):
        return self._uid

    def __repr__(self):
        return self._uid


class InputResourceFile(ResourceFile):
    def __init__(self, value):
        self._input_path = None
        super().__init__(value)

    def _add_input_path(self, path):
        self._input_path = path
        return self

    def _get_path(self, directory):
        assert self._value is not None
        return directory + '/inputs/' + self._value


class TaskResourceFile(ResourceFile):
    def _get_path(self, directory):
        assert self._source is not None
        assert self._value is not None
        return directory + '/' + self._source._uid + '/' + self._value


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

        for name, resource_file in values.items():
            assert isinstance(resource_file, ResourceFile)
            self._resources[name] = resource_file
            resource_file._add_resource_group(self)

    def _get_path(self, directory):
        subdir = self._source._uid if self._source else 'inputs'
        return directory + '/' + subdir + '/' + self._root

    def _add_output_path(self, path):
        self._output_paths.add(path)
        if self._source is not None:
            self._source._add_outputs(self)

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
