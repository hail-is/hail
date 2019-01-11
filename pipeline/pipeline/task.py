import re

from .resource import Resource, ResourceGroup


class Task:
    _counter = 0
    _uid_prefix = "__TASK__"
    _regex_pattern = r"(?P<TASK>{}\d+)".format(_uid_prefix)

    @classmethod
    def _new_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, pipeline, label=None):
        self._pipeline = pipeline
        self._label = label
        self._cpu = None
        self._memory = None
        self._docker = None
        self._command = []
        self._namespace = {}
        self._resources = {}
        self._dependencies = set()
        self._inputs = set()
        self._uid = Task._new_uid()

    def _get_resource(self, item):
        if item not in self._namespace:
            self._namespace[item] = self._pipeline._new_resource(self)
        r = self._namespace[item]
        if isinstance(r, Resource):
            return str(r)
        else:
            assert isinstance(r, ResourceGroup)
            return r

    def __getitem__(self, item):
        return self._get_resource(item)

    def __getattr__(self, item):
        return self._get_resource(item)

    def declare_resource_group(self, **mappings):
        for name, d in mappings.items():
            assert name not in self._namespace
            if not isinstance(d, dict):
                raise ValueError(f"value for name '{name}' is not a dict. Found '{type(d)}' instead.")
            self._namespace[name] = self._pipeline._new_resource_group(self, d)
        return self

    def depends_on(self, *tasks):
        for t in tasks:
            self._dependencies.add(t)

    def command(self, command):
        from .pipeline import Pipeline

        def add_dependencies(r):
            if isinstance(r, ResourceGroup):
                for _, resource in r._namespace.items():
                    add_dependencies(resource)
            else:
                assert isinstance(r, Resource)
                if r._source is not None and r._source != self:
                    self._dependencies.add(r._source)

        def handler(match_obj):
            groups = match_obj.groupdict()
            if groups['TASK']:
                raise ValueError(f"found a reference to a Task object in command '{command}'.")
            elif groups['PIPELINE']:
                raise ValueError(f"found a reference to a Pipeline object in command '{command}'.")
            else:
                assert groups['RESOURCE'] or groups['RESOURCE_GROUP']
                r_uid = match_obj.group()
                r = self._pipeline._resource_map.get(r_uid)
                if r is None:
                    raise KeyError(f"undefined resource '{r_uid}' in command '{command}'. "
                                   f"Hint: resources must be from the same pipeline as the current task.")
                add_dependencies(r)
                self._resources[r._uid] = r
                return f"${{{r_uid}}}"

        subst_command = re.sub(f"({Resource._regex_pattern})|({ResourceGroup._regex_pattern})"
                               f"|({Task._regex_pattern})|({Pipeline._regex_pattern})",
                               handler,
                               command)
        self._command.append(subst_command)
        return self

    def label(self, label):
        self._label = label
        return self

    def memory(self, memory):
        self._memory = memory
        return self

    def cpu(self, cpu):
        self._cpu = cpu
        return self

    def docker(self, docker):
        self._docker = docker
        return self

    def __str__(self):
        return self._uid
