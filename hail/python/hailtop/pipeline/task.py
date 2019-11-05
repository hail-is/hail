import re

from .resource import ResourceFile, ResourceGroup
from .utils import PipelineException


def _add_resource_to_set(resource_set, resource, include_rg=True):
    if isinstance(resource, ResourceGroup):
        rg = resource
        if include_rg:
            resource_set.add(resource)
    else:
        resource_set.add(resource)
        if isinstance(resource, ResourceFile) and resource._has_resource_group():
            rg = resource._get_resource_group()
        else:
            rg = None

    if rg is not None:
        for _, resource_file in rg._resources.items():
            resource_set.add(resource_file)


class Task:
    """
    Object representing a single job to execute.

    Examples
    --------

    Create a pipeline object:

    >>> p = Pipeline()

    Create a new pipeline task that prints hello to a temporary file `t.ofile`:

    >>> t = p.new_task()
    >>> t.command(f'echo "hello" > {t.ofile}')

    Write the temporary file `t.ofile` to a permanent location

    >>> p.write_output(t.ofile, 'hello.txt')

    Execute the DAG:

    >>> p.run()

    Notes
    -----
    This class should never be created directly by the user. Use `Pipeline.new_task` instead.
    """

    _counter = 0
    _uid_prefix = "__TASK__"
    _regex_pattern = r"(?P<TASK>{}\d+)".format(_uid_prefix)

    @classmethod
    def _new_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, pipeline, name=None, attributes=None):
        self._pipeline = pipeline
        self.name = name
        self.attributes = attributes
        self._cpu = None
        self._memory = None
        self._storage = None
        self._image = None
        self._command = []

        self._resources = {}  # dict of name to resource
        self._resources_inverse = {}  # dict of resource to name
        self._uid = Task._new_uid()

        self._inputs = set()
        self._internal_outputs = set()
        self._external_outputs = set()
        self._mentioned = set()  # resources used in the command
        self._valid = set()  # resources declared in the appropriate place
        self._dependencies = set()

    def _get_resource(self, item):
        if item not in self._resources:
            r = self._pipeline._new_task_resource_file(self)
            self._resources[item] = r
            self._resources_inverse[r] = item

        return self._resources[item]

    def __getitem__(self, item):
        return self._get_resource(item)

    def __getattr__(self, item):
        return self._get_resource(item)

    def _add_internal_outputs(self, resource):
        _add_resource_to_set(self._internal_outputs, resource, include_rg=False)

    def _add_inputs(self, resource):
        _add_resource_to_set(self._inputs, resource, include_rg=False)

    def declare_resource_group(self, **mappings):
        """
        Declare a resource group for a task.

        Examples
        --------

        Declare a resource group:

        >>> input = p.read_input_group(bed='data/example.bed',
        ...                            bim='data/example.bim',
        ...                            fam='data/example.fam')

        >>> t = p.new_task()
        >>> t.declare_resource_group(tmp1={'bed': '{root}.bed',
        ...                                'bim': '{root}.bim',
        ...                                'fam': '{root}.fam',
        ...                                'log': '{root}.log'})
        >>> t.command(f"plink --bfile {input} --make-bed --out {t.tmp1}")

        Caution
        -------
        Be careful when specifying the expressions for each file as this is Python
        code that is executed with `eval`!

        Parameters
        ----------
        mappings: :obj:`dict` of :obj:`str` to :obj:`dict` of :obj:`str` to :obj:`str`
            Keywords are the name(s) of the resource group(s). The value is a dict
            mapping the individual file identifier to a string expression representing
            how to transform the resource group root name into a file. Use `{root}`
            for the file root.

        Returns
        -------
        :class:`.Task`
            Same task object with resource groups set.
        """

        for name, d in mappings.items():
            assert name not in self._resources
            if not isinstance(d, dict):
                raise PipelineException(f"value for name '{name}' is not a dict. Found '{type(d)}' instead.")
            rg = self._pipeline._new_resource_group(self, d)
            self._resources[name] = rg
            _add_resource_to_set(self._valid, rg)
        return self

    def depends_on(self, *tasks):
        """
        Explicitly set dependencies on other tasks.

        Examples
        --------

        Create the first task:

        >>> t1 = p.new_task()
        >>> t1.command(f'echo "hello"')

        Create the second task that depends on `t1`:

        >>> t2 = p.new_task()
        >>> t2.depends_on(t1)
        >>> t2.command(f'echo "world"')

        Notes
        -----
        Dependencies between tasks are automatically created when resources from
        one task are used in a subsequent task. This method is only needed when
        no intermediate resource exists and the dependency needs to be explicitly
        set.

        Parameters
        ----------
        tasks: :class:`.Task`, varargs
            Sequence of tasks to depend on.

        Returns
        -------
        :class:`.Task`
            Same task object with dependencies set.
        """

        for t in tasks:
            self._dependencies.add(t)

    def command(self, command):
        """
        Set the task's command to execute.

        Examples
        --------

        Simple task with no output files:

        >>> p = Pipeline()
        >>> t1 = p.new_task()
        >>> t1.command(f'echo "hello"')
        >>> p.run()

        Simple task with one temporary file `t2.ofile` that is written to a
        permanent location:

        >>> p = Pipeline()
        >>> t2 = p.new_task()
        >>> t2.command(f'echo "hello world" > {t2.ofile}')
        >>> p.write_output(t2.ofile, 'output/hello.txt')
        >>> p.run()

        Two tasks with a file interdependency:

        >>> p = Pipeline()
        >>> t3 = p.new_task()
        >>> t3.command(f'echo "hello" > {t3.ofile}')
        >>> t4 = p.new_task()
        >>> t4.command(f'cat {t3.ofile} > {t4.ofile}')
        >>> p.write_output(t4.ofile, 'output/cat_output.txt')
        >>> p.run()

        Specify multiple commands in the same task:

        >>> p = Pipeline()
        >>> t5 = p.new_task()
        >>> t5.command(f'echo "hello" > {t5.tmp1}')
        >>> t5.command(f'echo "world" > {t5.tmp2}')
        >>> t5.command(f'echo "!" > {t5.tmp3}')
        >>> t5.command(f'cat {t5.tmp1} {t5.tmp2} {t5.tmp3} > {t5.ofile}')
        >>> p.write_output(t5.ofile, 'output/concatenated.txt')
        >>> p.run()

        Notes
        -----
        This method can be called more than once. It's behavior is to
        append commands to run to the set of previously defined commands
        rather than overriding an existing command.

        To declare a resource file of type :class:`.TaskResourceFile`, use either
        the get attribute syntax of `task.{identifier}` or the get item syntax of
        `task['identifier']`. If an object for that identifier doesn't exist,
        then one will be created automatically (only allowed in the :meth:`.command`
        method). The identifier name can be any valid Python identifier
        such as `ofile5000`.

        All :class:`.TaskResourceFile` are temporary files and must be written
        to a permanent location using :func:`.Pipeline.write_output` if the output needs
        to be saved.

        Only Resources can be referred to in commands. Referencing a :class:`.Pipeline`
        or :class:`.Task` will result in an error.

        Parameters
        ----------
        command: :obj:`str`

        Returns
        -------
        :class:`.Task`
            Same task object with command appended.
        """

        def handler(match_obj):
            groups = match_obj.groupdict()
            if groups['TASK']:
                raise PipelineException(f"found a reference to a Task object in command '{command}'.")
            if groups['PIPELINE']:
                raise PipelineException(f"found a reference to a Pipeline object in command '{command}'.")

            assert groups['RESOURCE_FILE'] or groups['RESOURCE_GROUP']
            r_uid = match_obj.group()
            r = self._pipeline._resource_map.get(r_uid)
            if r is None:
                raise PipelineException(f"undefined resource '{r_uid}' in command '{command}'.\n"
                                        f"Hint: resources must be from the same pipeline as the current task.")
            if r._source != self:
                self._add_inputs(r)
                if r._source is not None:
                    if r not in r._source._valid:
                        name = r._source._resources_inverse[r]
                        raise PipelineException(f"undefined resource '{name}'\n"
                                                f"Hint: resources must be defined within "
                                                "the task methods 'command' or 'declare_resource_group'")
                    self._dependencies.add(r._source)
                    r._source._add_internal_outputs(r)
            else:
                _add_resource_to_set(self._valid, r)
            self._mentioned.add(r)
            return f"${{{r_uid}}}"

        from .pipeline import Pipeline  # pylint: disable=cyclic-import

        subst_command = re.sub(f"({ResourceFile._regex_pattern})|({ResourceGroup._regex_pattern})"
                               f"|({Task._regex_pattern})|({Pipeline._regex_pattern})",
                               handler,
                               command)
        self._command.append(subst_command)
        return self

    def storage(self, storage):
        """
        Set the task's storage size.

        Examples
        --------

        Set the task's disk requirements to 1 Gi:

        >>> t1 = p.new_task()
        >>> (t1.storage('1Gi')
        ...    .command(f'echo "hello"'))

        Parameters
        ----------
        storage: :obj:`str`

        Returns
        -------
        :class:`.Task`
            Same task object with storage set.
        """
        self._storage = storage
        return self

    def memory(self, memory):
        """
        Set the task's memory requirements.

        Examples
        --------

        Set the task's memory requirement to 5GB:

        >>> t1 = p.new_task()
        >>> (t1.memory(5)
        ...    .command(f'echo "hello"'))

        Parameters
        ----------
        memory: :obj:`str` or :obj:`float` or :obj:`int`
            Value is in GB.

        Returns
        -------
        :class:`.Task`
            Same task object with memory requirements set.
        """
        self._memory = str(memory)
        return self

    def cpu(self, cores):
        """
        Set the task's CPU requirements.

        Examples
        --------

        Set the task's CPU requirement to 2 cores:

        >>> t1 = p.new_task()
        >>> (t1.cpu(2)
        ...    .command(f'echo "hello"'))

        Parameters
        ----------
        cores: :obj:`str` or :obj:`float` or :obj:`int`

        Returns
        -------
        :class:`.Task`
            Same task object with CPU requirements set.
        """

        self._cpu = str(cores)
        return self

    def image(self, image):
        """
        Set the task's docker image.

        Examples
        --------

        Set the task's docker image to `alpine`:

        >>> t1 = p.new_task()
        >>> (t1.image('alpine:latest')
        ...    .command(f'echo "hello"'))

        Parameters
        ----------
        image: :obj:`str`
            Docker image to use.

        Returns
        -------
        :class:`.Task`
            Same task object with docker image set.
        """

        self._image = image
        return self

    def _pretty(self):
        s = f"Task '{self._uid}'" \
            f"\tName:\t'{self.name}'" \
            f"\tAttributes:\t'{self.attributes}'" \
            f"\tImage:\t'{self._image}'" \
            f"\tCPU:\t'{self._cpu}'" \
            f"\tMemory:\t'{self._memory}'" \
            f"\tStorage:\t'{self._storage}'" \
            f"\tCommand:\t'{self._command}'"
        return s

    def __str__(self):
        return self._uid
