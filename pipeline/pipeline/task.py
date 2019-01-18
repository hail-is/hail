import re

from .resource import ResourceFile, ResourceGroup


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

        self._resources = {}  # dict of name to resource
        self._uid = Task._new_uid()

        self._inputs = set()
        self._outputs = set()
        self._dependencies = set()

    def _get_resource(self, item):
        if item not in self._resources:
            self._resources[item] = self._pipeline._new_task_resource_file(self)
        return self._resources[item]

    def __getitem__(self, item):
        return self._get_resource(item)

    def __getattr__(self, item):
        return self._get_resource(item)

    def declare_resource_group(self, **mappings):
        for name, d in mappings.items():
            assert name not in self._resources
            if not isinstance(d, dict):
                raise ValueError(f"value for name '{name}' is not a dict. Found '{type(d)}' instead.")
            self._resources[name] = self._pipeline._new_resource_group(self, d)
        return self

    def depends_on(self, *tasks):
        for t in tasks:
            self._dependencies.add(t)

    def command(self, command):
        from .pipeline import Pipeline

        def handler(match_obj):
            groups = match_obj.groupdict()
            if groups['TASK']:
                raise ValueError(f"found a reference to a Task object in command '{command}'.")
            elif groups['PIPELINE']:
                raise ValueError(f"found a reference to a Pipeline object in command '{command}'.")
            else:
                assert groups['RESOURCE_FILE'] or groups['RESOURCE_GROUP']
                r_uid = match_obj.group()
                r = self._pipeline._resource_map.get(r_uid)
                if r is None:
                    raise KeyError(f"undefined resource '{r_uid}' in command '{command}'.\n"
                                   f"Hint: resources must be from the same pipeline as the current task.")
                if r._source != self:
                    self._inputs.add(r)
                    if r._source is not None:
                        self._dependencies.add(r._source)
                else:
                    self._outputs.add(r)
                return f"${{{r_uid}}}"

        subst_command = re.sub(f"({ResourceFile._regex_pattern})|({ResourceGroup._regex_pattern})"
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
