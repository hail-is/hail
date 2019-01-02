from .resource import Resource
import jinja2
from jinja2 import meta


class TaskSettings(object):
    def __init__(self, cpu=None, memory=None, docker=None, env=None):
        self.cpu = cpu
        self.memory = memory
        self.docker = docker
        self.env = env

    def __str__(self):
        return f"TaskSettings(cpu={self.cpu}, memory={self.memory}, docker={self.docker}, env={self.env})"


default_task_settings = TaskSettings(cpu=1, memory=1, docker=None, env=None)


class Task(object):
    _counter = 0

    def __init__(self, pipeline, label=None, settings=None):
        self._settings = settings if settings else default_task_settings
        assert(isinstance(self._settings, TaskSettings))

        self._pipeline = pipeline
        self._label = label
        self._command = []
        self._resources = {}
        self._dependencies = set()

        self._uid = "__TASK__{}".format(Task._counter)
        Task._counter += 1

    def __getitem__(self, item):
        if item in self._resources:
            return self._resources[item]

    def __getattr__(self, item):
        if item in self._resources:
            return self._resources[item]

    def command(self, command):
        # parse command and extract identifiers
        ast = jinja2.Environment().parse(command)
        var_names = jinja2.meta.find_undeclared_variables(ast)

        # add the new var names to the environment
        for var in var_names:
            if var not in self._resources:
                self._resources[var] = Resource(self, value=self._pipeline._backend.temp_file())

        # create template for variable subst later on
        template = jinja2.Template(command)
        self._command.append(template)

        return self

    def _render_command(self, resource_to_str):
        rendered = []
        for cmd_template in self._command:
            resources = {name:resource_to_str(resource) for name, resource in self._resources.items()}
            rendered.append(cmd_template.render(**resources))
        return rendered

    def label(self, label):
        self._label = label
        return self

    def memory(self, memory):
        self._settings.memory = memory
        return self

    def cpu(self, cpu):
        self._settings.cpu = cpu
        return self

    def docker(self, docker):
        self._settings.docker = docker
        return self

    def env(self, env):
        self._settings.env = env
        return self

    def copy(self):
        new_task = Task(self._label, self._settings)
        new_task._command = self._command
        new_task._resources = self._resources
        new_task._dependencies = self._dependencies
        new_task._pipeline = self._pipeline
        return new_task

    def inputs(self, **kwargs):
        from pyapi.pipeline import Pipeline

        def to_resource(item, depth=0):
            if isinstance(item, Task):
                raise ValueError("cannot have a task input")
            elif isinstance(item, Pipeline):
                raise ValueError("cannot have a pipeline input")
            elif isinstance(item, Resource):
                if item._source == self:
                    raise ValueError("cannot have an input from the same task")
                else:
                    self.add_dependency(item._source)
            elif isinstance(item, list) and depth == 0:
                return Resource(value=[to_resource(x, depth=depth+1) for x in item])
            elif isinstance(item, list) and depth > 0:
                raise ValueError("cannot have nested lists as inputs")
            else:
                assert isinstance(item, str)
                item = Resource(value=item)
            return item

        for name, input in kwargs.items():
            if name not in self._resources:
                raise ValueError(f"undefined identifier '{name}'.")
            self._resources[name]._value = to_resource(input)._value
        return self

    def add_dependency(self, *tasks):
        self._dependencies = self._dependencies.union(set(tasks))

    def remove_dependency(self, *tasks):
        self._dependencies = self._dependencies.difference(set(tasks))

    def __str__(self):
        return f"Task(_label={self._label}, _settings={self._settings}," \
                f"_command={self._command}, _dependencies={self._dependencies})"
