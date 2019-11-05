import os
import re
import uuid

from .backend import LocalBackend, BatchBackend
from .task import Task
from .resource import Resource, InputResourceFile, TaskResourceFile, ResourceGroup
from .utils import PipelineException


class Pipeline:
    """
    Object representing the distributed acyclic graph (DAG) of jobs to run.

    Examples
    --------

    Create a pipeline object:

    >>> p = Pipeline()

    Create a new pipeline task that prints hello to a temporary file `t.ofile`:

    >>> t = p.new_task()
    >>> t.command(f'echo "hello" > {t.ofile}')

    Write the temporary file `t.ofile` to a permanent location

    >>> p.write_output(t.ofile, 'output/hello.txt')

    Execute the DAG:

    >>> p.run()

    Parameters
    ----------
    name: :obj:`str`, optional
        Name of the pipeline.
    backend: :func:`.Backend`, optional
        Backend used to execute the jobs. Default is :class:`.LocalBackend`
    attributes: :obj:`dict` of :obj:`str` to :obj:`str`, optional
        Key-value pairs of additional attributes. 'name' is not a valid keyword.
        Use the name argument instead.
    default_image: :obj:`str`, optional
        Docker image to use by default if not specified by a task.
    default_memory: :obj:`str`, optional
        Memory setting to use by default if not specified by a task. Only
        applicable if a docker image is specified for the :class:`.LocalBackend`
        or the :class:`.BatchBackend`. Value is in GB.
    default_cpu: :obj:`str`, optional
        CPU setting to use by default if not specified by a task. Only
        applicable if a docker image is specified for the :class:`.LocalBackend`
        or the :class:`.BatchBackend`.
    default_storage: :obj:`str`, optional
        Storage setting to use by default if not specified by a task. Only
        applicable for the :class:`.BatchBackend`.
    """

    _counter = 0
    _uid_prefix = "__PIPELINE__"
    _regex_pattern = r"(?P<PIPELINE>{}\d+)".format(_uid_prefix)

    @classmethod
    def _get_uid(cls):
        uid = "{}{}".format(cls._uid_prefix, cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, name=None, backend=None, attributes=None,
                 default_image=None, default_memory=None, default_cpu=None,
                 default_storage=None):
        self._tasks = []
        self._resource_map = {}
        self._allocated_files = set()
        self._input_resources = set()
        self._uid = Pipeline._get_uid()

        self.name = name

        if attributes is None:
            attributes = {}
        if 'name' in attributes:
            raise PipelineException("'name' is not a valid attribute. Use the name argument instead.")
        self.attributes = attributes

        self._default_image = default_image
        self._default_memory = default_memory
        self._default_cpu = default_cpu
        self._default_storage = default_storage

        if backend:
            self._backend = backend
        elif os.environ.get('BATCH_URL') is not None:
            self._backend = BatchBackend(os.environ.get('BATCH_URL'))
        else:
            self._backend = LocalBackend()

    def new_task(self, name=None, attributes=None):
        """
        Initialize a new task object with default memory, docker image,
        and CPU settings if specified upon pipeline creation.

        Examples
        --------

        >>> t = p.new_task()

        Parameters
        ----------
        name: :obj:`str`, optional
            Name of the task.
        attributes: :obj:`dict` of :obj:`str` to :obj:`str`, optional
            Key-value pairs of additional attributes. 'name' is not a valid keyword.
            Use the name argument instead.

        Returns
        -------
        :class:`.Task`
        """

        if attributes is None:
            attributes = {}

        t = Task(pipeline=self, name=name, attributes=attributes)

        if self._default_image is not None:
            t.image(self._default_image)
        if self._default_memory is not None:
            t.memory(self._default_memory)
        if self._default_cpu is not None:
            t.cpu(self._default_cpu)
        if self._default_storage is not None:
            t.storage(self._default_storage)

        self._tasks.append(t)
        return t

    def _tmp_file(self, prefix=None, suffix=None):
        def _get_random_file():
            file = '{}{}{}'.format(prefix if prefix else '',
                                   uuid.uuid4().hex[:8],
                                   suffix if suffix else '')
            if file not in self._allocated_files:
                self._allocated_files.add(file)
                return file
            return _get_random_file()

        return _get_random_file()

    def _new_task_resource_file(self, source, value=None):
        trf = TaskResourceFile(value if value else self._tmp_file())
        trf._add_source(source)
        self._resource_map[trf._uid] = trf  # pylint: disable=no-member
        return trf

    def _new_input_resource_file(self, input_path, value=None):
        irf = InputResourceFile(value if value else self._tmp_file())
        irf._add_input_path(input_path)
        self._resource_map[irf._uid] = irf  # pylint: disable=no-member
        self._input_resources.add(irf)
        return irf

    def _new_resource_group(self, source, mappings):
        assert isinstance(mappings, dict)
        root = self._tmp_file()
        d = {}
        new_resource_map = {}
        for name, code in mappings.items():
            if not isinstance(code, str):
                raise PipelineException(f"value for name '{name}' is not a string. Found '{type(code)}' instead.")
            r = self._new_task_resource_file(source=source, value=eval(f'f"""{code}"""'))  # pylint: disable=W0123
            d[name] = r
            new_resource_map[r._uid] = r  # pylint: disable=no-member

        self._resource_map.update(new_resource_map)
        rg = ResourceGroup(source, root, **d)
        self._resource_map.update({rg._uid: rg})
        return rg

    def read_input(self, path, extension=None):
        """
        Create a new input resource file object representing a single file.

        Examples
        --------

        Read the file `hello.txt`:

        >>> p = Pipeline()
        >>> input = p.read_input('hello.txt')
        >>> t = p.new_task()
        >>> t.command(f"cat {input}")
        >>> p.run()

        Parameters
        ----------
        path: :obj:`str`
            File path to read.
        extension: :obj:`str`, optional
            File extension to use.

        Returns
        -------
        :class:`.InputResourceFile`
        """

        irf = self._new_input_resource_file(path)
        if extension is not None:
            irf.add_extension(extension)
        return irf

    def read_input_group(self, **kwargs):
        """
        Create a new resource group representing a mapping of identifier to
        input resource files.

        Examples
        --------

        Read a binary PLINK file:

        >>> p = Pipeline()
        >>> bfile = p.read_input_group(bed="data/example.bed",
        ...                            bim="data/example.bim",
        ...                            fam="data/example.fam")
        >>> t = p.new_task()
        >>> t.command(f"plink --bfile {bfile} --geno --out {t.geno}")
        >>> t.command(f"wc -l {bfile.fam}")
        >>> t.command(f"wc -l {bfile.bim}")
        >>> p.run()

        Read a FASTA file and it's index (file extensions matter!):

        >>> fasta = p.read_input_group({'fasta': 'data/example.fasta',
        ...                             'fasta.idx': 'data/example.fasta.idx'})

        Create a resource group where the identifiers don't match the file extensions:

        >>> rg = p.read_input_group(foo='data/foo.txt',
        ...                         bar='data/bar.txt')

        `rg.foo` and `rg.bar` will not have the `.txt` file extension and
        instead will be `{root}.foo` and `{root}.bar` where `{root}` is a random
        identifier.

        Notes
        -----
        The identifier is used to refer to
        a specific resource file. For example, given the resource group `rg`, you
        can use the attribute notation `rg.identifier` or the get item notation
        `rg[identifier]`.

        The file extensions for each file are derived from the identifier.
        This is equivalent to `"{root}.identifier"` from
        :meth:`.Task.declare_resource_group`. We are planning on adding flexibility
        to incorporate more complicated extensions in the future such as `.vcf.bgz`.
        For now, use :func:`ResourceFile.add_extension` to add an extension to a
        resource file.

        Parameters
        ----------
        kwargs: :obj:`dict` of :obj:`str` to :obj:`str`
            Key word arguments where the name/key is the identifier and the value
            is the file path.

        Returns
        -------
        :class:`.InputResourceFile`
        """

        root = self._tmp_file()
        new_resources = {name: self._new_input_resource_file(file, root + '.' + name) for name, file in kwargs.items()}
        rg = ResourceGroup(None, root, **new_resources)
        self._resource_map.update({rg._uid: rg})
        return rg

    def write_output(self, resource, dest):  # pylint: disable=R0201
        """
        Write resource file or resource file group to an output destination.

        Examples
        --------

        Write a single task intermediate to a permanent location:

        >>> p = Pipeline()
        >>> t = p.new_task()
        >>> t.command(f'echo "hello" > {t.ofile}')
        >>> p.write_output(t.ofile, 'output/hello.txt')
        >>> p.run()

        Notes
        -----
        All :class:`.TaskResourceFile` are temporary files and must be written
        to a permanent location using :meth:`.write_output` if the output needs
        to be saved.

        Parameters
        ----------
        resource: :class:`.ResourceFile` or :class:`.ResourceGroup`
            Resource to be written to a file.
        dest: :obj:`str`
            Destination file path. For a single :class:`.ResourceFile`, this will
            simply be `dest`. For a :class:`.ResourceGroup`, `dest` is the file
            root and each resource file will be written to `{root}.identifier`
            where `identifier` is the identifier of the file in the
            :class:`.ResourceGroup` map.
        """

        if not isinstance(resource, Resource):
            raise PipelineException(f"'write_output' only accepts Resource inputs. Found '{type(resource)}'.")
        if isinstance(resource, TaskResourceFile) and resource not in resource._source._mentioned:
            name = resource._source._resources_inverse
            raise PipelineException(f"undefined resource '{name}'\n"
                                    f"Hint: resources must be defined within the "
                                    "task methods 'command' or 'declare_resource_group'")
        resource._add_output_path(dest)

    def select_tasks(self, pattern):
        """
        Select all tasks in the pipeline whose name matches `pattern`.

        Examples
        --------

        Select tasks in pipeline matching `qc`:

        >>> p = Pipeline()
        >>> t = p.new_task().name('qc')
        >>> qc_tasks = p.select_tasks('qc')
        >>> assert qc_tasks == [t]

        Parameters
        ----------
        pattern: :obj:`str`
            Regex pattern matching task names.

        Returns
        -------
        :obj:`list` of :class:`.Task`
        """

        return [task for task in self._tasks if task.name is not None and re.match(pattern, task.name) is not None]

    def run(self, dry_run=False, verbose=False, delete_scratch_on_exit=True):
        """
        Execute a pipeline.

        Examples
        --------

        Create a simple pipeline and execute it:

        >>> p = Pipeline()
        >>> t = p.new_task()
        >>> t.command('echo "hello"')
        >>> p.run()

        Parameters
        ----------
        dry_run: :obj:`bool`, optional
            If `True`, don't execute code.
        verbose: :obj:`bool`, optional
            If `True`, print debugging output.
        delete_scratch_on_exit: :obj:`bool`, optional
            If `True`, delete temporary directories with intermediate files.
        """

        seen = set()
        ordered_tasks = []

        def schedule_task(t):
            if t in seen:
                return
            seen.add(t)
            for p in t._dependencies:
                schedule_task(p)
            ordered_tasks.append(t)

        for t in self._tasks:
            schedule_task(t)

        assert len(seen) == len(self._tasks)

        task_index = {t: i for i, t in enumerate(ordered_tasks)}
        for t in ordered_tasks:
            i = task_index[t]
            for p in t._dependencies:
                j = task_index[p]
                if j >= i:
                    raise PipelineException("cycle detected in dependency graph")

        self._tasks = ordered_tasks
        self._backend._run(self, dry_run, verbose, delete_scratch_on_exit)

    def __str__(self):
        return self._uid
