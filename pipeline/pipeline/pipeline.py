import re

from .backend import LocalBackend
from .task import Task
from .resource import Resource, ResourceGroup
from .utils import get_sha


class Pipeline(object):
    _counter = 0
    _uid_prefix = "__PIPELINE__"
    _regex_pattern = r"(?P<PIPELINE>{}\d+)".format(_uid_prefix)

    @classmethod
    def _get_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, backend=None):
        self._tasks = []
        self._resource_map = {}
        self._allocated_files = set()
        self._backend = backend if backend else LocalBackend()
        self._uid = Pipeline._get_uid()

    def new_task(self):
        t = Task(pipeline=self)
        self._tasks.append(t)
        return t

    def _tmp_file(self, prefix=None, suffix=None):
        def _get_random_file():
            file = '{}{}{}'.format(prefix if prefix else '',
                                   get_sha(8),
                                   suffix if suffix else '')
            if file not in self._allocated_files:
                self._allocated_files.add(file)
                return file
            else:
                _get_random_file()

        return _get_random_file()

    def _new_resource(self, source=None, value=None):
        r = Resource(source, value if value else self._tmp_file())
        self._resource_map[r._uid] = r
        return r

    def _new_resource_group(self, source, mappings):
        assert isinstance(mappings, dict)
        root = self._tmp_file()
        d = {}
        new_resource_map = {}
        for name, code in mappings.items():
            if not isinstance(code, str):
                raise ValueError(f"value for name '{name}' is not a string. Found '{type(code)}' instead.")
            r = self._new_resource(source=source, value=eval(f'f"""{code}"""'))
            d[name] = r
            new_resource_map[r._uid] = r

        self._resource_map.update(new_resource_map)
        rg = ResourceGroup(root, **d)
        self._resource_map.update({rg._uid: rg})
        return rg

    def _read_input(self, source, dest=None):
        dest = dest if dest else self._tmp_file()
        cp_task = (self.new_task()
                   .label('read_input')
                   .command(self._backend.cp(source, dest)))
        return self._new_resource(source=cp_task, value=dest)

    def read_input(self, source):
        return str(self._read_input(source))

    def read_input_group(self, **kwargs):
        root = self._tmp_file()
        added_resources = {name:self._read_input(file, root + '.' + name) for name, file in kwargs.items()}
        rg = ResourceGroup(root, **added_resources)
        self._resource_map.update({rg._uid: rg})
        return rg

    def write_output(self, resource, dest):
        if isinstance(resource, str):
            resource = self._resource_map[resource]
        else:
            assert isinstance(resource, Resource)
            assert resource._uid in self._resource_map
        cp_task = (self.new_task()
                   .label('write_output')
                   .command(self._backend.cp(resource, dest)))

    def select_tasks(self, pattern):
        return [task for task in self._tasks if task._label is not None and re.match(pattern, task._label) is not None]

    def run(self, dry_run=False, verbose=True, delete_on_exit=True):
        dependencies = {task:task._dependencies for task in self._tasks}
        ordered_tasks = []
        niter = 0
        while dependencies:
            for task, deps in dependencies.items():
                if len(deps) == 0:
                    ordered_tasks.append(task)
                    niter = 0
            for task, _ in dependencies.items():
                dependencies[task] = dependencies[task].difference(set(ordered_tasks))
            for task in ordered_tasks:
                if task in dependencies:
                    del dependencies[task]
            niter += 1

            if niter == 100:
                raise ValueError("cycle detected in dependency graph")

        self._tasks = ordered_tasks
        self._backend.run(self, dry_run, verbose, False, delete_on_exit) # FIXME: expose bg option when implemented!

    def __str__(self):
        return self._uid
