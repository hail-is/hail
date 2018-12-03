from .task import Task, default_task_settings
from .backend import LocalBackend, get_sha


class Pipeline(object):
    def __init__(self, backend=LocalBackend(), task_settings=None, tmp_dir='/tmp/'):
        self._tasks = []
        self._backend = backend
        self._task_settings = task_settings if task_settings else default_task_settings
        self._tmp_dir = tmp_dir + '/pipeline_{}/'.format(get_sha(6))

    def new_task(self):
        t = Task(pipeline=self, settings=self._task_settings)
        self._tasks.append(t)
        return t

    def write_output(self, resource, dest):
        cp_task = (self.new_task()
                   .command(self._backend.cp_cmd_template())
                   .inputs(src=resource, dest=dest))
        if resource._source is not None:
            cp_task.add_dependency(resource._source)

    def run(self):
        dependencies = {task:len(task._dependencies) for task in self._tasks}
        ordered_tasks = []
        niter = 0
        while dependencies:
            for task, ndep in dependencies.items():
                if ndep == 0:
                    ordered_tasks.append(task)
            for task, _ in dependencies.items():
                task.remove_dependency(*ordered_tasks)
            for task in ordered_tasks:
                if task in dependencies:
                    del dependencies[task]
            dependencies = {task:len(task._dependencies) for task, _ in dependencies.items()}

            if niter == 100:
                raise ValueError("cycle detected in dependency graph")

        self._tasks = ordered_tasks
        self._backend.run(self)
