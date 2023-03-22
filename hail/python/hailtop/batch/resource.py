import abc
from typing import Optional, Set, cast

from . import job  # pylint: disable=cyclic-import
from .exceptions import BatchException


class Resource:
    """
    Abstract class for resources.
    """

    _uid: str
    _source: Optional[job.Job]

    @abc.abstractmethod
    def _get_path(self, directory: str) -> str:
        pass

    @abc.abstractmethod
    def _add_output_path(self, path: str) -> None:
        pass


class ResourceFile(Resource, str):
    """
    Class representing a single file resource. There exist two subclasses:
    :class:`.InputResourceFile` and :class:`.JobResourceFile`.
    """
    _counter = 0
    _uid_prefix = "__RESOURCE_FILE__"
    _regex_pattern = r"(?P<RESOURCE_FILE>{}\d+)".format(_uid_prefix)  # pylint: disable=consider-using-f-string

    @classmethod
    def _new_uid(cls):
        uid = cls._uid_prefix + str(cls._counter)
        cls._counter += 1
        return uid

    def __new__(cls, *args, **kwargs):  # pylint: disable=W0613
        uid = ResourceFile._new_uid()
        r = str.__new__(cls, uid)
        r._uid = uid
        return r

    def __init__(self, value: Optional[str]):
        super().__init__()
        assert value is None or isinstance(value, str)
        self._value = value
        self._source: Optional[job.Job] = None
        self._output_paths: Set[str] = set()
        self._resource_group: Optional[ResourceGroup] = None

    def _get_path(self, directory: str):
        raise NotImplementedError

    def _add_output_path(self, path: str) -> None:
        self._output_paths.add(path)
        if self._source is not None:
            self._source._external_outputs.add(self)

    def _add_resource_group(self, rg: 'ResourceGroup') -> None:
        self._resource_group = rg

    def _has_resource_group(self) -> bool:
        return self._resource_group is not None

    def _get_resource_group(self) -> Optional['ResourceGroup']:
        return self._resource_group

    def __str__(self):
        return f'{self._uid}'  # pylint: disable=no-member

    def __repr__(self):
        return self._uid  # pylint: disable=no-member


class InputResourceFile(ResourceFile):
    """
    Class representing a resource from an input file.

    Examples
    --------
    `input` is an :class:`.InputResourceFile` of the batch `b`
    and is used in job `j`:

    >>> b = Batch()
    >>> input = b.read_input('data/hello.txt')
    >>> j = b.new_job(name='hello')
    >>> j.command(f'cat {input}')
    >>> b.run()
    """

    def __init__(self, value):
        self._input_path = None
        super().__init__(value)

    def _add_input_path(self, path: str) -> 'InputResourceFile':
        self._input_path = path
        return self

    def _get_path(self, directory: str) -> str:
        assert self._value is not None
        return directory + '/inputs/' + self._value


class JobResourceFile(ResourceFile):
    """
    Class representing an intermediate file from a job.

    Examples
    --------
    `j.ofile` is a :class:`.JobResourceFile` on the job`j`:

    >>> b = Batch()
    >>> j = b.new_job(name='hello-tmp')
    >>> j.command(f'echo "hello world" > {j.ofile}')
    >>> b.run()

    Notes
    -----
    All :class:`.JobResourceFile` are temporary files and must be written
    to a permanent location using :meth:`.Batch.write_output` if the output needs
    to be saved.
    """

    def __init__(self, value, source: job.Job):
        super().__init__(value)
        self._has_extension = False
        self._source: job.Job = source

    def _get_path(self, directory: str) -> str:
        assert self._source is not None
        assert self._value is not None
        return f'{directory}/{self._source._dirname}/{self._value}'

    def add_extension(self, extension: str) -> 'JobResourceFile':
        """
        Specify the file extension to use.

        Examples
        --------

        >>> b = Batch()
        >>> j = b.new_job()
        >>> j.command(f'echo "hello" > {j.ofile}')
        >>> j.ofile.add_extension('.txt')
        >>> b.run()

        Notes
        -----
        The default file name for a :class:`.JobResourceFile` is the name
        of the identifier.

        Parameters
        ----------
        extension: :obj:`str`
            File extension to use.

        Returns
        -------
        :class:`.JobResourceFile`
            Same resource file with the extension specified
        """
        if self._has_extension:
            raise BatchException("Resource already has a file extension added.")
        assert self._value is not None
        self._value += extension
        self._has_extension = True
        return self


class ResourceGroup(Resource):
    """
    Class representing a mapping of identifiers to a resource file.

    Examples
    --------

    Initialize a batch and create a new job:

    >>> b = Batch()
    >>> j = b.new_job()

    Read a set of input files as a resource group:

    >>> bfile = b.read_input_group(bed='data/example.bed',
    ...                            bim='data/example.bim',
    ...                            fam='data/example.fam')

    Create a resource group from a job intermediate:

    >>> j.declare_resource_group(ofile={'bed': '{root}.bed',
    ...                                 'bim': '{root}.bim',
    ...                                 'fam': '{root}.fam'})
    >>> j.command(f'plink --bfile {bfile} --make-bed --out {j.ofile}')

    Reference the entire file group:

    >>> j.command(f'plink --bfile {bfile} --geno 0.2 --make-bed --out {j.ofile}')

    Reference a single file:

    >>> j.command(f'wc -l {bfile.fam}')

    Execute the batch:

    >>> b.run() # doctest: +SKIP

    Notes
    -----
    All files in the resource group are copied between jobs even if only one
    file in the resource group is mentioned. This is to account for files that
    are implicitly assumed to always be together such as a FASTA file and its
    index.
    """

    _counter = 0
    _uid_prefix = "__RESOURCE_GROUP__"
    _regex_pattern = r"(?P<RESOURCE_GROUP>{}\d+)".format(_uid_prefix)  # pylint: disable=consider-using-f-string

    @classmethod
    def _new_uid(cls):
        uid = cls._uid_prefix + str(cls._counter)
        cls._counter += 1
        return uid

    def __init__(self, source: Optional[job.Job], root: str, **values: ResourceFile):
        self._source = source
        self._resources = {}  # dict of name to resource uid
        self._root = root
        self._uid = ResourceGroup._new_uid()

        for name, resource_file in values.items():
            assert isinstance(resource_file, ResourceFile)
            self._resources[name] = resource_file
            resource_file._add_resource_group(self)

    def _get_path(self, directory: str) -> str:
        subdir = str(self._source._dirname) if self._source else 'inputs'
        return directory + '/' + subdir + '/' + self._root

    def _add_output_path(self, path: str) -> None:
        for name, rf in self._resources.items():
            rf._add_output_path(path + '.' + name)

    def _get_resource(self, item: str) -> ResourceFile:
        if item not in self._resources:
            raise BatchException(f"'{item}' not found in the resource group.\n"
                                 f"Hint: you must declare each attribute when constructing the resource group.")
        return self._resources[item]

    def __getitem__(self, item: str) -> ResourceFile:
        return self._get_resource(item)

    def __getattr__(self, item: str) -> ResourceFile:
        return self._get_resource(item)

    def __add__(self, other: str):
        assert isinstance(other, str)
        return str(self._uid) + other

    def __radd__(self, other: str):
        assert isinstance(other, str)
        return other + str(self._uid)

    def __str__(self):
        return f'{self._uid}'


class PythonResult(Resource, str):
    """
    Class representing a result from a Python job.

    Examples
    --------

    Add two numbers and then square the result:

    .. code-block:: python

        def add(x, y):
            return x + y

        def square(x):
            return x ** 2


        b = Batch()
        j = b.new_python_job(name='add')
        result = j.call(add, 3, 2)
        result = j.call(square, result)
        b.write_output(result.as_str(), 'output/squared.txt')
        b.run()

    Notes
    -----
    All :class:`.PythonResult` are temporary Python objects and must be written
    to a permanent location using :meth:`.Batch.write_output` if the output needs
    to be saved. In most cases, you'll want to convert the :class:`.PythonResult`
    to a :class:`.JobResourceFile` in a human-readable format.
    """
    _counter = 0
    _uid_prefix = "__PYTHON_RESULT__"
    _regex_pattern = r"(?P<PYTHON_RESULT>{}\d+)".format(_uid_prefix)  # pylint: disable=consider-using-f-string

    @classmethod
    def _new_uid(cls):
        uid = cls._uid_prefix + str(cls._counter)
        cls._counter += 1
        return uid

    def __new__(cls, *args, **kwargs):  # pylint: disable=W0613
        uid = PythonResult._new_uid()
        r = str.__new__(cls, uid)
        r._uid = uid
        return r

    def __init__(self, value: str, source: job.PythonJob):
        super().__init__()
        assert value is None or isinstance(value, str)
        self._value = value
        self._source: job.PythonJob = source
        self._output_paths: Set[str] = set()
        self._json = None
        self._str = None
        self._repr = None

    def _get_path(self, directory: str) -> str:
        assert self._source is not None
        assert self._value is not None
        return f'{directory}/{self._source._dirname}/{self._value}'

    def _add_converted_resource(self, value):
        jrf = self._source._batch._new_job_resource_file(self._source, value)
        self._source._resources[value] = jrf
        self._source._resources_inverse[jrf] = value
        self._source._valid.add(jrf)
        self._source._mentioned.add(jrf)
        return jrf

    def _add_output_path(self, path: str) -> None:
        self._output_paths.add(path)
        if self._source is not None:
            self._source._external_outputs.add(self)

    def source(self) -> job.PythonJob:
        """
        Get the job that created the Python result.
        """
        return cast(job.PythonJob, self._source)

    def as_json(self) -> JobResourceFile:
        """
        Convert a Python result to a file with a JSON representation of the object.

        Examples
        --------

        .. code-block:: python

            def add(x, y):
                return {'result': x + y}


            b = Batch()
            j = b.new_python_job(name='add')
            result = j.call(add, 3, 2)
            b.write_output(result.as_json(), 'output/add.json')
            b.run()

        Returns
        -------
        :class:`.JobResourceFile`
            A new resource file where the contents are a Python object
            that has been converted to JSON.
        """
        if self._json is None:
            jrf = self._add_converted_resource(self._value + '-json')
            jrf.add_extension('.json')
            self._json = jrf
        return cast(JobResourceFile, self._json)

    def as_str(self) -> JobResourceFile:
        """
        Convert a Python result to a file with the str representation of the object.

        Examples
        --------

        .. code-block:: python

            def add(x, y):
                return x + y


            b = Batch()
            j = b.new_python_job(name='add')
            result = j.call(add, 3, 2)
            b.write_output(result.as_str(), 'output/add.txt')
            b.run()

        Returns
        -------
        :class:`.JobResourceFile`
            A new resource file where the contents are the str representation
            of a Python object.
        """
        if self._str is None:
            jrf = self._add_converted_resource(self._value + '-str')
            jrf.add_extension('.txt')
            self._str = jrf
        return cast(JobResourceFile, self._str)

    def as_repr(self) -> JobResourceFile:
        """
        Convert a Python result to a file with the repr representation of the object.

        Examples
        --------

        .. code-block:: python

            def add(x, y):
                return x + y


            b = Batch()
            j = b.new_python_job(name='add')
            result = j.call(add, 3, 2)
            b.write_output(result.as_repr(), 'output/add.txt')
            b.run()

        Returns
        -------
        :class:`.JobResourceFile`
            A new resource file where the contents are the repr representation
            of a Python object.
        """
        if self._repr is None:
            jrf = self._add_converted_resource(self._value + '-repr')
            jrf.add_extension('.txt')
            self._repr = jrf
        return cast(JobResourceFile, self._repr)

    def __str__(self):
        return f'{self._uid}'  # pylint: disable=no-member

    def __repr__(self):
        return self._uid  # pylint: disable=no-member
