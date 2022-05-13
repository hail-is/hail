from typing import Union, Optional, Dict, List, Set, Tuple, Callable, Any, KeysView
import abc
import re

import warnings
import dill
import os
import functools
import inspect
import textwrap
from io import BytesIO, StringIO

from . import backend, resource as _resource, batch  # pylint: disable=cyclic-import
from .exceptions import BatchException
from .globals import DEFAULT_SHELL


def opt_str(x: Optional[Any]) -> Optional[str]:
    if x is None:
        return x
    return str(x)


class ResourceToStringPickler(dill.Pickler):
    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)
        self.serialized_resources: List[_resource.Resource] = []
        self.dispatch = {
            **dill.Pickler.dispatch,
            **{k: ResourceToStringPickler.save_resource
               for k in _resource.ALL_RESOURCE_CLASSES}
        }

    def save_resource(self, obj: _resource.Resource):
        self.serialized_resources.append(obj)
        return self.save(obj.as_py())


class Job(abc.ABC):
    """
    Object representing a single job to execute.

    Notes
    -----
    This class should never be created directly by the user. Use :meth:`.Batch.new_job`,
    :meth:`.Batch.new_bash_job`, or :meth:`.Batch.new_python_job` instead.
    """

    def __init__(self,
                 batch: 'batch.Batch',
                 token: str,
                 *,
                 name: Optional[str] = None,
                 attributes: Optional[Dict[str, str]] = None,
                 shell: Optional[str] = None):
        self._batch = batch
        self._shell = shell
        self._token = token

        self.name = name
        self.attributes = attributes

        self._dependencies: Set['Job'] = set()
        self._produced_resource_by_name: Dict[str, _resource.Resource] = {}
        self._defined_resource_remoteness: Dict[_resource.Resource, bool] = {}
        self._need_to_copy_out: Set[_resource.Resource] = set()

        self._env: Dict[str, str] = {}
        self._storage: Optional[str] = None
        self._memory: Optional[str] = None
        self._cpu: Optional[str] = None
        self._always_run: bool = False
        self._timeout: Optional[Union[int, float]] = None
        self._cloudfuse: List[Tuple[str, str, bool]] = []
        self._wrapper_code: List[str] = []
        self._user_code: List[str] = []
        self._image: Optional[str] = None

        # unused?
        self._preemptible: Optional[bool] = None
        self._machine_type: Optional[str] = None

    def defined_resources(self) -> KeysView[_resource.Resource]:
        return self._defined_resource_remoteness.keys()

    def resource_defined_remotely(self, resource: _resource.Resource) -> bool:
        print(resource, self._defined_resource_remoteness, self)
        return self._defined_resource_remoteness[resource]

    def resource_is_defined(self, resource: _resource.Resource, *, remote: bool = False):
        if resource.is_remote():
            return self.resource_is_defined(resource.defining_resource(), remote=True)
        self._defined_resource_remoteness[resource] = remote

    def resource_is_needed(self, resource: _resource.Resource):
        if resource.is_remote():
            return self.resource_is_needed(resource.defining_resource())
        if self.resource_defined_remotely(resource):
            return
        self._need_to_copy_out.add(resource)

    def __getitem__(self, name: str) -> '_resource.Resource':
        resource = self._produced_resource_by_name.get(name)
        if resource is None:
            resource = _resource.JobResource(self._batch._remote_location, self, name, None)
            self._produced_resource_by_name[name] = resource
            self._batch._resource_by_uid[resource.uid()] = resource
        return resource

    def __getattr__(self, name: str) -> '_resource.Resource':
        return self.__getitem__(name)

    @abc.abstractmethod
    async def compile(self, *, dry_run=False) -> Tuple[str, str, List['_resource.Resource']]:
        pass

    def depends_on(self, *jobs: 'Job') -> 'Job':
        """
        Explicitly set dependencies on other jobs.

        Examples
        --------

        Initialize the batch:

        >>> b = Batch()

        Create the first job:

        >>> j1 = b.new_job()
        >>> j1.command(f'echo "hello"')

        Create the second job `j2` that depends on `j1`:

        >>> j2 = b.new_job()
        >>> j2.depends_on(j1)
        >>> j2.command(f'echo "world"')

        Execute the batch:

        >>> b.run()

        Notes
        -----
        Dependencies between jobs are automatically created when resources from
        one job are used in a subsequent job. This method is only needed when
        no intermediate resource exists and the dependency needs to be explicitly
        set.

        Parameters
        ----------
        jobs:
            Sequence of jobs to depend on.

        Returns
        -------
        Same job object with dependencies set.
        """

        self._dependencies.extend(jobs)
        return self

    def env(self, variable: str, value: str):
        self._env[variable] = value

    def storage(self, storage: Optional[Union[str, int]]) -> 'Job':
        """
        Set the job's storage size.

        Examples
        --------

        Set the job's disk requirements to 1 Gi:

        >>> b = Batch()
        >>> j = b.new_job()
        >>> (j.storage('10Gi')
        ...   .command(f'echo "hello"'))
        >>> b.run()

        Notes
        -----

        The storage expression must be of the form {number}{suffix}
        where valid optional suffixes are *K*, *Ki*, *M*, *Mi*,
        *G*, *Gi*, *T*, *Ti*, *P*, and *Pi*. Omitting a suffix means
        the value is in bytes.

        For the :class:`.ServiceBackend`, jobs requesting one or more cores receive
        5 GiB of storage for the root file system `/`. Jobs requesting a fraction of a core
        receive the same fraction of 5 GiB of storage. If you need additional storage, you
        can explicitly request more storage using this method and the extra storage space
        will be mounted at `/io`. Batch automatically writes all :class:`.ResourceFile` to
        `/io`.

        The default storage size is 0 Gi. The minimum storage size is 0 Gi and the
        maximum storage size is 64 Ti. If storage is set to a value between 0 Gi
        and 10 Gi, the storage request is rounded up to 10 Gi. All values are
        rounded up to the nearest Gi.

        Parameters
        ----------
        storage:
            Units are in bytes if `storage` is an :obj:`int`. If `None`, use the
            default storage size for the :class:`.ServiceBackend` (0 Gi).

        Returns
        -------
        Same job object with storage set.
        """

        self._storage = opt_str(storage)
        return self

    def memory(self, memory: Optional[Union[str, int]]) -> 'Job':
        """
        Set the job's memory requirements.

        Examples
        --------

        Set the job's memory requirement to be 3Gi:

        >>> b = Batch()
        >>> j = b.new_job()
        >>> (j.memory('3Gi')
        ...   .command(f'echo "hello"'))
        >>> b.run()

        Notes
        -----

        The memory expression must be of the form {number}{suffix}
        where valid optional suffixes are *K*, *Ki*, *M*, *Mi*,
        *G*, *Gi*, *T*, *Ti*, *P*, and *Pi*. Omitting a suffix means
        the value is in bytes.

        For the :class:`.ServiceBackend`, the values 'lowmem', 'standard',
        and 'highmem' are also valid arguments. 'lowmem' corresponds to
        approximately 1 Gi/core, 'standard' corresponds to approximately
        4 Gi/core, and 'highmem' corresponds to approximately 7 Gi/core.
        The default value is 'standard'.

        Parameters
        ----------
        memory:
            Units are in bytes if `memory` is an :obj:`int`. If `None`,
            use the default value for the :class:`.ServiceBackend` ('standard').

        Returns
        -------
        Same job object with memory requirements set.
        """

        self._memory = opt_str(memory)
        return self

    def cpu(self, cores: Optional[Union[str, int, float]]) -> 'Job':
        """
        Set the job's CPU requirements.

        Notes
        -----

        The string expression must be of the form {number}{suffix}
        where the optional suffix is *m* representing millicpu.
        Omitting a suffix means the value is in cpu.

        For the :class:`.ServiceBackend`, `cores` must be a power of
        two between 0.25 and 16.

        Examples
        --------

        Set the job's CPU requirement to 250 millicpu:

        >>> b = Batch()
        >>> j = b.new_job()
        >>> (j.cpu('250m')
        ...   .command(f'echo "hello"'))
        >>> b.run()

        Parameters
        ----------
        cores:
            Units are in cpu if `cores` is numeric. If `None`,
            use the default value for the :class:`.ServiceBackend`
            (1 cpu).

        Returns
        -------
        Same job object with CPU requirements set.
        """

        self._cpu = opt_str(cores)
        return self

    def always_run(self, always_run: bool = True) -> 'Job':
        """
        Set the job to always run, even if dependencies fail.

        Notes
        -----
        Can only be used with the :class:`.backend.ServiceBackend`.

        Warning
        -------
        Jobs set to always run are not cancellable!

        Examples
        --------

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> (j.always_run()
        ...   .command(f'echo "hello"'))

        Parameters
        ----------
        always_run:
            If True, set job to always run.

        Returns
        -------
        Same job object set to always run.
        """

        if not isinstance(self._batch._backend, backend.ServiceBackend):
            raise NotImplementedError("A ServiceBackend is required to use the 'always_run' option")

        self._always_run = always_run
        return self

    def timeout(self, timeout: Optional[Union[float, int]]) -> 'Job':
        """
        Set the maximum amount of time this job can run for.

        Notes
        -----
        Can only be used with the :class:`.backend.ServiceBackend`.

        Examples
        --------

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> (j.timeout(10)
        ...   .command(f'echo "hello"'))

        Parameters
        ----------
        timeout:
            Maximum amount of time for a job to run before being killed.
            If `None`, there is no timeout.

        Returns
        -------
        Same job object set with a timeout.
        """

        if not isinstance(self._batch._backend, backend.ServiceBackend):
            raise NotImplementedError("A ServiceBackend is required to use the 'timeout' option")

        self._timeout = timeout
        return self

    def gcsfuse(self, bucket, mount_point, read_only=True):
        """
        Add a bucket to mount with gcsfuse.

        Notes
        -----
        Can only be used with the :class:`.backend.ServiceBackend`. This method can
        be called more than once. This method has been deprecated. Use :meth:`.Job.cloudfuse`
        instead.

        Warning
        -------
        There are performance and cost implications of using `gcsfuse <https://cloud.google.com/storage/docs/gcs-fuse>`__.

        Examples
        --------

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> (j.gcsfuse('my-bucket', '/my-bucket')
        ...   .command(f'cat /my-bucket/my-file'))

        Parameters
        ----------
        bucket:
            Name of the google storage bucket to mount.
        mount_point:
            The path at which the bucket should be mounted to in the Docker
            container.
        read_only:
            If ``True``, mount the bucket in read-only mode.

        Returns
        -------
        Same job object set with a bucket to mount with gcsfuse.
        """

        warnings.warn("The 'gcsfuse' method has been deprecated. Use the 'cloudfuse' method instead.")
        return self.cloudfuse(bucket, mount_point, read_only=read_only)

    def cloudfuse(self, bucket: str, mount_point: str, *, read_only: bool = True):
        """
        Add a bucket to mount with gcsfuse in GCP or a storage container with blobfuse in Azure.

        Notes
        -----
        Can only be used with the :class:`.backend.ServiceBackend`. This method can
        be called more than once.

        Warning
        -------
        There are performance and cost implications of using `gcsfuse <https://cloud.google.com/storage/docs/gcs-fuse>`__
        or `blobfuse <https://github.com/Azure/azure-storage-fuse#considerations>`__.

        Examples
        --------

        Google Cloud Platform:

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> (j.cloudfuse('my-bucket', '/my-bucket')
        ...   .command(f'cat /my-bucket/my-blob-object'))

        Azure:

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> (j.cloudfuse('my-account/my-container', '/dest')
        ...   .command(f'cat /dest/my-blob-object'))

        Parameters
        ----------
        bucket:
            Name of the google storage bucket to mount or the path to an Azure container in the
            format of `<account>/<container>`.
        mount_point:
            The path at which the cloud blob storage should be mounted to in the Docker
            container.
        read_only:
            If ``True``, mount the cloud blob storage in read-only mode.

        Returns
        -------
        Same job object set with a cloud storage path to mount with either gcsfuse or blobfuse.
        """

        if not isinstance(self._batch._backend, backend.ServiceBackend):
            raise NotImplementedError("A ServiceBackend is required to use the 'cloudfuse' option")

        if bucket == '':
            raise BatchException('location cannot be the empty string')

        if mount_point == '':
            raise BatchException('mount_point cannot be the empty string')

        self._cloudfuse.append((bucket, mount_point, read_only))
        return self

    def image(self, image: str) -> 'Job':
        """
        Set the job's docker image.

        Examples
        --------

        Set the job's docker image to `ubuntu:20.04`:

        >>> b = Batch()
        >>> j = b.new_job()
        >>> (j.image('ubuntu:20.04')
        ...   .command(f'echo "hello"'))
        >>> b.run()  # doctest: +SKIP

        Parameters
        ----------
        image:
            Docker image to use.

        Returns
        -------
        Same job object with docker image set.
        """

        self._image = image
        return self


class BashJob(Job):
    """
    Object representing a single bash job to execute.

    Examples
    --------

    Create a batch object:

    >>> b = Batch()

    Create a new bash job that prints hello to a temporary file `t.ofile`:

    >>> j = b.new_job()
    >>> j.command(f'echo "hello" > {j.ofile}')

    Write the temporary file `t.ofile` to a permanent location

    >>> b.write_output(j.ofile, 'hello.txt')

    Execute the DAG:

    >>> b.run()

    Notes
    -----
    This class should never be created directly by the user. Use :meth:`.Batch.new_job`
    or :meth:`.Batch.new_bash_job` instead.
    """

    def __init__(self,
                 batch: 'batch.Batch',
                 token: str,
                 *,
                 name: Optional[str] = None,
                 attributes: Optional[Dict[str, str]] = None,
                 shell: Optional[str] = None):
        super().__init__(batch, token, name=name, attributes=attributes, shell=shell)
        self._command: List[str] = []

    def declare_resource_group(self, **mappings: Dict[str, str]) -> 'BashJob':
        """Declare a resource group for a job.

        Examples
        --------

        Declare a resource group:

        >>> b = Batch()
        >>> input = b.read_input_group(bed='data/example.bed',
        ...                            bim='data/example.bim',
        ...                            fam='data/example.fam')
        >>> j = b.new_job()
        >>> j.declare_resource_group(tmp1={'bed': '{root}.bed',
        ...                                'bim': '{root}.bim',
        ...                                'fam': '{root}.fam',
        ...                                'log': '{root}.log'})
        >>> j.command(f'plink --bfile {input} --make-bed --out {j.tmp1}')
        >>> b.run()  # doctest: +SKIP

        Warning
        -------
        Be careful when specifying the expressions for each file as this is Python
        code that is executed with `eval`!

        # FIXME: !!!!! eval?

        Parameters
        ----------
        mappings:
            Keywords (in the above example `tmp1`) are the name(s) of the
            resource group(s).


            FIXME: woah, eval?
            File names may contain arbitrary Python
            expressions, which will be evaluated by Python `eval`.  To use the
            keyword as the file name, use `{root}` (in the above example {root}
            will be replaced with `tmp1`).

        Returns
        -------
        Same job object with resource groups set.
        """

        for name, named_format_strings in mappings.items():
            rg = _resource.ResourceGroup(
                self._batch._remote_location,
                named_format_strings,
                self,
                f'ResourceGroup({",".join(mappings.keys())})'
            )
            self._produced_resource_by_name[name] = rg
            self._batch._resource_by_uid[rg.uid()] = rg

        return self

    def command(self, command: str) -> 'BashJob':
        """Set the job's command to execute.

        Examples
        --------

        Simple job with no output files:

        >>> b = Batch()
        >>> j = b.new_job()
        >>> j.command(f'echo "hello"')
        >>> b.run()

        Simple job with one temporary file `j.ofile` that is written to a
        permanent location:

        >>> b = Batch()
        >>> j = b.new_job()
        >>> j.command(f'echo "hello world" > {j.ofile}')
        >>> b.write_output(j.ofile, 'output/hello.txt')
        >>> b.run()

        Two jobs with a file interdependency:

        >>> b = Batch()
        >>> j1 = b.new_job()
        >>> j1.command(f'echo "hello" > {j1.ofile}')
        >>> j2 = b.new_bash_job()
        >>> j2.command(f'cat {j1.ofile} > {j2.ofile}')
        >>> b.write_output(j2.ofile, 'output/cat_output.txt')
        >>> b.run()

        Specify multiple commands in the same job:

        >>> b = Batch()
        >>> t = b.new_job()
        >>> j.command(f'echo "hello" > {j.tmp1}')
        >>> j.command(f'echo "world" > {j.tmp2}')
        >>> j.command(f'echo "!" > {j.tmp3}')
        >>> j.command(f'cat {j.tmp1} {j.tmp2} {j.tmp3} > {j.ofile}')
        >>> b.write_output(j.ofile, 'output/concatenated.txt')
        >>> b.run()

        Notes
        -----
        This method can be called more than once. It's behavior is to append
        commands to run to the set of previously defined commands rather than
        overriding an existing command.

        To declare a resource file of type :class:`.JobResourceFile`, use either
        the get attribute syntax of `job.{identifier}` or the get item syntax of
        `job['identifier']`. If an object for that identifier doesn't exist,
        then one will be created automatically (only allowed in the
        :meth:`.command` method). The identifier name can be any valid Python
        identifier such as `ofile5000`.

        All :class:`.JobResourceFile` are temporary files and must be written to
        a permanent location using :meth:`.Batch.write_output` if the output
        needs to be saved.

        Only resources can be referred to in commands. Referencing a
        :class:`.batch.Batch` or :class:`.Job` will result in an error.

        Parameters
        ----------
        command:
            A ``bash`` command.

        Returns
        -------
        Same job object with command appended.
        """

        for mo in re.finditer(_resource.Resource.REGEXP, command):
            uid = _resource.decode_uid(mo[2])
            try:
                resource = self._batch._resource_by_uid[uid]
            except KeyError as err:
                raise BatchException(
                    f"undefined resource '{mo[2]}' in command '{command}'.\n"
                    f"Hint: resources must be from the same batch as the current job.") from err
            maybe_source = resource.source()
            if maybe_source is not None:
                if maybe_source == self:
                    self.resource_is_defined(resource)
                else:
                    maybe_source.resource_is_needed(resource)
                    self._dependencies.add(maybe_source)

        self._command.append(command)
        return self

    async def compile(self, *, dry_run=False) -> Tuple[str, str, List['_resource.Resource']]:
        job_shell = self._shell if self._shell else DEFAULT_SHELL

        job_command = ['#! ' + job_shell]
        job_command.extend([
            'mkdir -p ' + os.path.dirname(resource.local_location())
            for resource in self.defined_resources()])
        user_code = [cmd.strip() for cmd in self._command]
        job_command.extend(user_code)
        job_command_str = '\n'.join(job_command)
        user_code_str = '\n'.join(user_code)
        job_command_bytes = job_command_str.encode('utf-8')

        resources_to_download = []
        for match in re.finditer(_resource.Resource.REGEXP, job_command_str):
            assert match[1] in 'LR'
            is_referenced_remote = match[1] == 'R'
            uid = _resource.decode_uid(match[2])
            resource = self._batch._resource_by_uid[uid]
            if resource not in self.defined_resources() and not is_referenced_remote:
                resources_to_download.append(resource)

        if len(job_command_bytes) <= 10 * 1024:
            return (job_command_str, user_code_str, resources_to_download)

        code_location = self._batch._remote_location + '/' + self._token + '/' + 'code.sh'
        if not dry_run:
            # FIXME: local fs .create should be responsibile for makedirs
            await self._batch._fs.makedirs(os.path.dirname(code_location), exist_ok=True)
            await self._batch._fs.write(code_location, job_command_bytes)
        code_resource = self._batch.read_input(code_location)
        resources_to_download.append(code_resource)

        job_command_str += f'\nchmod u+x {code_resource}; source {code_resource}'

        return (job_command_str, user_code_str, resources_to_download)


class PythonJob(Job):
    """
    Object representing a single Python job to execute.

    Examples
    --------

    Create a new Python job that multiplies two numbers and then adds 5 to the result:

    .. code-block:: python

        # Create a batch object with a default Python image

        b = Batch(default_python_image='gcr.io/hail-vdc/python-dill:3.7-slim')

        def multiply(x, y):
            return x * y

        def add(x, y):
            return x + y

        j = b.new_python_job()
        result = j.call(multiply, 2, 3)
        result = j.call(add, result, 5)

        # Write out the str representation of result to a file

        b.write_output(result.as_str(), 'hello.txt')

        b.run()

    Notes
    -----
    This class should never be created directly by the user. Use :meth:`.Batch.new_python_job`
    instead.
    """

    def __init__(self,
                 batch: 'batch.Batch',
                 token: str,
                 *,
                 name: Optional[str] = None,
                 attributes: Optional[Dict[str, str]] = None):
        super().__init__(batch, token, name=name, attributes=attributes, shell=None)
        self._resources: Dict[str, _resource.Resource] = {}
        self._resources_inverse: Dict[_resource.Resource, str] = {}
        self._functions: List[Tuple[_resource.PythonResult, bytes, str]] = []
        self.n_results = 0

    def image(self, image: str) -> 'PythonJob':
        """
        Set the job's docker image.

        Notes
        -----

        `image` must already exist and have the same version of Python as what is
        being used on the computer submitting the Batch. It also must have the
        `dill` Python package installed. You can use the function :func:`.docker.build_python_image`
        to build a new image containing `dill` and additional Python packages.

        Examples
        --------

        Set the job's docker image to `gcr.io/hail-vdc/python-dill:3.7-slim`:

        >>> b = Batch()
        >>> j = b.new_python_job()
        >>> (j.image('gcr.io/hail-vdc/python-dill:3.7-slim')
        ...   .call(print, 'hello'))
        >>> b.run()  # doctest: +SKIP

        Parameters
        ----------
        image:
            Docker image to use.

        Returns
        -------
        Same job object with docker image set.
        """

        self._image = image
        return self

    def call(self, unapplied: Callable, *args, **kwargs) -> '_resource.PythonResult':
        """Execute a Python function.

        Examples
        --------

        .. code-block:: python

            import json

            def add(x, y):
                return x + y

            def multiply(x, y):
                return x * y

            def format_as_csv(x, y, add_result, mult_result):
                return f'{x},{y},{add_result},{mult_result}'

            def csv_to_json(path):
                data = []
                with open(path) as f:
                    for line in f:
                        line = line.rstrip()
                        fields = line.split(',')
                        d = {'x': int(fields[0]),
                             'y': int(fields[1]),
                             'add': int(fields[2]),
                             'mult': int(fields[3])}
                        data.append(d)
                return json.dumps(data)


            # Get all the multiplication and addition table results

            b = Batch(name='add-mult-table')

            formatted_results = []

            for x in range(3):
                for y in range(3):
                    j = b.new_python_job(name=f'{x}-{y}')
                    add_result = j.call(add, x, y)
                    mult_result = j.call(multiply, x, y)
                    result = j.call(format_as_csv, x, y, add_result, mult_result)
                    formatted_results.append(result.as_str())

            cat_j = b.new_bash_job(name='concatenate')
            cat_j.command(f'cat {" ".join(formatted_results)} > {cat_j.output}')

            csv_to_json_j = b.new_python_job(name='csv-to-json')
            json_output = csv_to_json_j.call(csv_to_json, cat_j.output)

            b.write_output(j.as_str(), '/output/add_mult_table.json')
            b.run()

        Notes
        -----
        Unlike the :class:`.BashJob`, a :class:`.PythonJob` returns a new
        :class:`.PythonResult` for every invocation of :meth:`.PythonJob.call`. A
        :class:`.PythonResult` can be used as an argument in subsequent invocations of
        :meth:`.PythonJob.call`, as an argument in downstream python jobs,
        or as inputs to other bash jobs. Likewise, :class:`.InputResourceFile`,
        :class:`.JobResourceFile`, and :class:`.ResourceGroup` can be passed to
        :meth:`.PythonJob.call`. Batch automatically detects dependencies between jobs
        including between python jobs and bash jobs.

        When a :class:`.ResourceFile` is passed as an argument, it is passed to the
        function as a string to the local file path. When a :class:`.ResourceGroup`
        is passed as an argument, it is passed to the function as a dict where the
        keys are the resource identifiers in the original :class:`.ResourceGroup`
        and the values are the local file paths.

        Like :class:`.JobResourceFile`, all :class:`.PythonResult` are stored as
        temporary files and must be written to a permanent location using
        :meth:`.Batch.write_output` if the output needs to be saved. A
        PythonResult is saved as a dill serialized object. However, you
        can use one of the methods :meth:`.PythonResult.as_str`, :meth:`.PythonResult.as_repr`,
        or :meth:`.PythonResult.as_json` to convert a `PythonResult` to a
        `JobResourceFile` with the desired output.

        Warning
        -------

        You must have any non-builtin packages that are used by `unapplied` installed
        in your image. You can use :func:`.docker.build_python_image` to build a
        Python image with additional Python packages installed that is compatible
        with Python jobs.

        Here are some tips to make sure your function can be used with Batch:

         - Only reference top-level modules in your functions: like numpy or pandas.
         - If you get a serialization error, try moving your imports into your function.
         - Instead of serializing a complex class, determine what information is essential
           and only serialize that, perhaps as a dict or array.

        Parameters
        ----------
        unapplied:
            A reference to a Python function to execute.
        args:
            Positional arguments to the Python function. Must be either a builtin
            Python object, a :class:`.Resource`, or a Dill serializable object.
        kwargs:
            Key-word arguments to the Python function. Must be either a builtin
            Python object, a :class:`.Resource`, or a Dill serializable object.

        Returns
        -------
        :class:`.resource.PythonResult`
        """

        if not callable(unapplied):
            raise BatchException(f'unapplied must be a callable function. Found {type(unapplied)}.')

        for arg in args:
            if isinstance(arg, Job):
                raise BatchException('arguments to a PythonJob cannot be other job objects.')

        for value in kwargs.values():
            if isinstance(value, Job):
                raise BatchException('arguments to a PythonJob cannot be other job objects.')

        self.n_results += 1
        result_name = f'result{self.n_results}'
        result = _resource.PythonResult(self._batch._remote_location, self, result_name)
        self._produced_resource_by_name[result_name] = result
        self._batch._resource_by_uid[result.uid()] = result
        self.resource_is_defined(result)

        pipe = BytesIO()
        pickler = ResourceToStringPickler(pipe, recurse=True)
        pickler.dump(functools.partial(unapplied, *args, **kwargs))
        pipe.seek(0)

        for resource in pickler.serialized_resources:
            maybe_source = resource.source()
            if maybe_source is not None:
                if maybe_source == self:
                    self.resource_is_defined(resource)
                else:
                    maybe_source.resource_is_needed(resource)
                    self._dependencies.add(maybe_source)

        user_code_builder = StringIO()
        user_code_builder.write(textwrap.dedent(inspect.getsource(unapplied)) + '\n')
        args_str = ', '.join([repr(arg) for arg in args])
        kwargs_str = ', '.join([f'{k}={v!r}' for k, v in kwargs.items()])
        separator = ', ' if args and kwargs else ''
        func_call = f'{unapplied.__name__}({args_str}{separator}{kwargs_str})'
        user_code_builder.write(func_call + '\n')

        self._functions.append((result, pipe.getvalue(), user_code_builder.getvalue()))

        return result

    async def compile(self, *, dry_run=False) -> Tuple[str, str, List['_resource.Resource']]:
        resources_to_download: List[_resource.Resource] = []
        command = StringIO()
        user_code_builder = StringIO()
        for i, (result, pickled, user_code) in enumerate(self._functions):
            user_code_builder.write(user_code)
            py_code_location = self._batch._remote_location + '/' + self._token + '/' + f'code{i}.dill'
            if not dry_run:
                # FIXME: local fs .create should be responsibile for makedirs
                await self._batch._fs.makedirs(os.path.dirname(py_code_location), exist_ok=True)
                await self._batch._fs.write(py_code_location, pickled)
            py_code_resource = self._batch.read_input(py_code_location)
            resources_to_download.append(py_code_resource)

            json_write = ''
            if result._as_json:
                resources_to_download.append(result._as_json)
                json_write = f'''
            os.makedirs(os.path.dirname(\\"result._as_json\\"), exist_ok=True)
            with open(\\"{result._as_json}\\", \\"w\\") as out:
                out.write(json.dumps(result) + \\"\\n\\")
'''

            str_write = ''
            if result._as_str:
                resources_to_download.append(result._as_str)
                str_write = f'''
            os.makedirs(os.path.dirname(\\"result._as_str\\"), exist_ok=True)
            with open(\\"{result._as_str}\\", \\"w\\") as out:
                out.write(str(result) + \\"\\n\\")
'''

            repr_write = ''
            if result._as_repr:
                resources_to_download.append(result._as_repr)
                repr_write = f'''
            os.makedirs(os.path.dirname(\\"result._as_repr\\"), exist_ok=True)
            with open(\\"{result._as_repr}\\", \\"w\\") as out:
                out.write(repr(result) + \\"\\n\\")
'''

            command.write(f'''python3 -c "
import os
import base64
import dill
import traceback
import json
import sys

class PythonResultUnpickler(dill.Unpickler):
    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)
        self.dispatch = {{
            **dill.Pickler.dispatch,
            **{k: ResourceToStringPickler.save_resource
           for k in _resource.ALL_RESOURCE_CLASSES}
        }}

    def save_resource(self, obj: _resource.Resource):
        return self.save(obj.as_py())

os.makedirs(os.path.dirname(\\"{result}\\"), exist_ok=True)
with open(\\"{result}\\", \\"wb\\") as dill_out:
    try:
        with open(\\"{py_code_resource}\\", \\"rb\\") as f:
            result = dill.load(f)()
            dill.dump(result, dill_out, recurse=True)
            {json_write}
            {str_write}
            {repr_write}
    except Exception as e:
        traceback.print_exc()
        dill.dump((e, traceback.format_exception(type(e), e, e.__traceback__)), dill_out, recurse=True)
        raise e
"''')

        return command.getvalue(), user_code_builder.getvalue(), resources_to_download
