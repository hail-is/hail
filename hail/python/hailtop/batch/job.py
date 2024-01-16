import asyncio
import inspect
import os
import re
import textwrap
import warnings
from shlex import quote as shq
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union, cast

from typing_extensions import Self

import hailtop.batch_client.client as bc

from . import backend, batch  # pylint: disable=cyclic-import
from . import resource as _resource  # pylint: disable=cyclic-import
from .exceptions import BatchException


def _add_resource_to_set(resource_set, resource, include_rg=True):
    rg: Optional[_resource.ResourceGroup]
    if isinstance(resource, _resource.ResourceGroup):
        rg = resource
        if include_rg:
            resource_set.add(resource)
    else:
        resource_set.add(resource)
        if isinstance(resource, _resource.ResourceFile) and resource._has_resource_group():
            rg = resource._get_resource_group()
        else:
            rg = None

    if rg is not None:
        for _, resource_file in rg._resources.items():
            resource_set.add(resource_file)


def opt_str(x):
    if x is None:
        return x
    return str(x)


class Job:
    """
    Object representing a single job to execute.

    Notes
    -----
    This class should never be created directly by the user. Use :meth:`.Batch.new_job`,
    :meth:`.Batch.new_bash_job`, or :meth:`.Batch.new_python_job` instead.
    """

    _counter = 1
    _uid_prefix = "__JOB__"
    _regex_pattern = r"(?P<JOB>{}\d+)".format(_uid_prefix)  # pylint: disable=consider-using-f-string

    @classmethod
    def _new_uid(cls):
        uid = cls._uid_prefix + str(cls._counter)
        cls._counter += 1
        return uid

    def __init__(
        self,
        batch: 'batch.Batch',
        token: str,
        *,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
        shell: Optional[str] = None,
    ):
        self._batch = batch
        self._shell = shell
        self._token = token

        self.name = name
        self.attributes = attributes

        self._cpu: Optional[str] = None
        self._memory: Optional[str] = None
        self._storage: Optional[str] = None
        self._image: Optional[str] = None
        self._always_run: bool = False
        self._preemptible: Optional[bool] = None
        self._machine_type: Optional[str] = None
        self._timeout: Optional[Union[int, float]] = None
        self._cloudfuse: List[Tuple[str, str, bool]] = []
        self._always_copy_output: bool = False
        self._env: Dict[str, str] = {}
        self._wrapper_code: List[str] = []
        self._user_code: List[str] = []
        self._regions: Optional[List[str]] = None

        self._resources: Dict[str, _resource.Resource] = {}
        self._resources_inverse: Dict[_resource.Resource, str] = {}
        self._uid = Job._new_uid()
        self._job_id: Optional[int] = None

        self._inputs: Set[_resource.Resource] = set()
        self._internal_outputs: Set[Union[_resource.ResourceFile, _resource.PythonResult]] = set()
        self._external_outputs: Set[Union[_resource.ResourceFile, _resource.PythonResult]] = set()
        self._mentioned: Set[_resource.Resource] = set()  # resources used in the command
        self._valid: Set[_resource.Resource] = set()  # resources declared in the appropriate place
        self._dependencies: Set[Job] = set()
        self._submitted: bool = False
        self._client_job: Optional[bc.Job] = None

        def safe_str(s):
            new_s = []
            for c in s:
                if c.isalnum() or c == '-':
                    new_s.append(c)
                else:
                    new_s.append('_')
            return ''.join(new_s)

        self._dirname = f'{safe_str(name)}-{self._token}' if name else self._token

    def _get_resource(self, item: str) -> '_resource.Resource':
        if item not in self._resources:
            r = self._batch._new_job_resource_file(self, value=item)
            self._resources[item] = r
            self._resources_inverse[r] = item

        return self._resources[item]

    def __getitem__(self, item: str) -> '_resource.Resource':
        return self._get_resource(item)

    def __getattr__(self, item: str) -> '_resource.Resource':
        return self._get_resource(item)

    def _add_internal_outputs(self, resource: '_resource.Resource') -> None:
        _add_resource_to_set(self._internal_outputs, resource, include_rg=False)

    def _add_inputs(self, resource: '_resource.Resource') -> None:
        _add_resource_to_set(self._inputs, resource, include_rg=False)

    def depends_on(self, *jobs: 'Job') -> Self:
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

        for j in jobs:
            self._dependencies.add(j)
        return self

    def env(self, variable: str, value: str):
        self._env[variable] = value

    def storage(self, storage: Optional[Union[str, int]]) -> Self:
        """
        Set the job's storage size.

        Examples
        --------

        Set the job's disk requirements to 10 Gi:

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

    def memory(self, memory: Optional[Union[str, int]]) -> Self:
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

    def cpu(self, cores: Optional[Union[str, int, float]]) -> Self:
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

    def always_run(self, always_run: bool = True) -> Self:
        """
        Set the job to always run, even if dependencies fail.

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
        self._always_run = always_run
        return self

    def spot(self, is_spot: bool) -> Self:
        """
        Set whether a job is run on spot instances. By default, all jobs run on spot instances.

        Examples
        --------

        Ensure a job only runs on non-spot instances:

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> j = j.spot(False)
        >>> j = j.command(f'echo "hello"')

        Parameters
        ----------
        is_spot:
            If False, this job will be run on non-spot instances.

        Returns
        -------
        Same job object.
        """
        self._preemptible = is_spot
        return self

    def regions(self, regions: Optional[List[str]]) -> Self:
        """
        Set the cloud regions a job can run in.

        Notes
        -----
        Can only be used with the :class:`.backend.ServiceBackend`.

        This method may be used to ensure code executes in the same region as the data it reads.
        This can avoid egress charges as well as improve latency.

        Examples
        --------

        Require the job to run in 'us-central1':

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> (j.regions(['us-central1'])
        ...   .command(f'echo "hello"'))

        Specify the job can run in any region:

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> (j.regions(None)
        ...   .command(f'echo "hello"'))

        Parameters
        ----------
        regions:
            The cloud region(s) to run this job in. Use `None` to signify
            the job can run in any available region. Use py:staticmethod:`.ServiceBackend.supported_regions`
            to list the available regions to choose from. The default is the job can run in
            any region.

        Returns
        -------
        Same job object with the cloud regions the job can run in set.
        """

        if not isinstance(self._batch._backend, backend.ServiceBackend):
            raise NotImplementedError("A ServiceBackend is required to use the 'regions' option")

        self._regions = regions
        return self

    def timeout(self, timeout: Optional[Union[float, int]]) -> Self:
        """
        Set the maximum amount of time this job can run for in seconds.

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
            Maximum amount of time in seconds for a job to run before being killed.
            If `None`, there is no timeout.

        Returns
        -------
        Same job object set with a timeout in seconds.
        """

        if not isinstance(self._batch._backend, backend.ServiceBackend):
            raise NotImplementedError("A ServiceBackend is required to use the 'timeout' option")

        self._timeout = timeout
        return self

    def gcsfuse(self, bucket, mount_point, read_only=True) -> Self:
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

    def cloudfuse(self, bucket: str, mount_point: str, *, read_only: bool = True) -> Self:
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

    def always_copy_output(self, always_copy_output: bool = True) -> Self:
        """
        Set the job to always copy output to cloud storage, even if the job failed.

        Notes
        -----
        Can only be used with the :class:`.backend.ServiceBackend`.

        Examples
        --------

        >>> b = Batch(backend=backend.ServiceBackend('test'))
        >>> j = b.new_job()
        >>> (j.always_copy_output()
        ...   .command(f'echo "hello" > {j.ofile} && false'))

        Parameters
        ----------
        always_copy_output:
            If True, set job to always copy output to cloud storage regardless
            of whether the job succeeded.

        Returns
        -------
        Same job object set to always copy output.
        """

        if not isinstance(self._batch._backend, backend.ServiceBackend):
            raise NotImplementedError("A ServiceBackend is required to use the 'always_copy_output' option")

        self._always_copy_output = always_copy_output
        return self

    async def _compile(self, local_tmpdir, remote_tmpdir, *, dry_run=False):
        raise NotImplementedError

    def _interpolate_command(self, command, allow_python_results=False):
        def handler(match_obj):
            groups = match_obj.groupdict()

            if groups['JOB']:
                raise BatchException(f"found a reference to a Job object in command '{command}'.")
            if groups['BATCH']:
                raise BatchException(f"found a reference to a Batch object in command '{command}'.")
            if groups['PYTHON_RESULT'] and not allow_python_results:
                raise BatchException(
                    f"found a reference to a PythonResult object. hint: Use one of the methods `as_str`, `as_json` or `as_repr` on a PythonResult. command: '{command}'"
                )

            assert groups['RESOURCE_FILE'] or groups['RESOURCE_GROUP'] or groups['PYTHON_RESULT']
            r_uid = match_obj.group()
            r = self._batch._resource_map.get(r_uid)

            if r is None:
                raise BatchException(
                    f"undefined resource '{r_uid}' in command '{command}'.\n"
                    f"Hint: resources must be from the same batch as the current job."
                )

            source = r.source()
            if source != self:
                self._add_inputs(r)
                if source is not None:
                    if r not in source._valid:
                        name = source._resources_inverse[r]
                        raise BatchException(
                            f"undefined resource '{name}'\n"
                            f"Hint: resources must be defined within "
                            f"the job methods 'command' or 'declare_resource_group'"
                        )

                    if self._always_run:
                        warnings.warn(
                            'A job marked as always run has a resource file dependency on another job. If the dependent job fails, '
                            f'the always run job with the following command may not succeed:\n{command}'
                        )

                    self._dependencies.add(source)
                    source._add_internal_outputs(r)
            else:
                _add_resource_to_set(self._valid, r)

            self._mentioned.add(r)
            return '${BATCH_TMPDIR}' + shq(r._get_path(''))

        regexes = [
            _resource.ResourceFile._regex_pattern,
            _resource.ResourceGroup._regex_pattern,
            _resource.PythonResult._regex_pattern,
            Job._regex_pattern,
            batch.Batch._regex_pattern,
        ]

        subst_command = re.sub('(' + ')|('.join(regexes) + ')', handler, command)

        return subst_command

    def _pretty(self):
        s = (
            f"Job '{self._uid}'"
            f"\tName:\t'{self.name}'"
            f"\tAttributes:\t'{self.attributes}'"
            f"\tImage:\t'{self._image}'"
            f"\tCPU:\t'{self._cpu}'"
            f"\tMemory:\t'{self._memory}'"
            f"\tStorage:\t'{self._storage}'"
            f"\tCommand:\t'{self._command}'"
        )
        return s

    def __str__(self):
        return self._uid


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

    def __init__(
        self,
        batch: 'batch.Batch',
        token: str,
        *,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
        shell: Optional[str] = None,
    ):
        super().__init__(batch, token, name=name, attributes=attributes, shell=shell)
        self._command: List[str] = []

    def declare_resource_group(self, **mappings: Dict[str, Any]) -> 'BashJob':
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

        Parameters
        ----------
        mappings:
            Keywords (in the above example `tmp1`) are the name(s) of the
            resource group(s).  File names may contain arbitrary Python
            expressions, which will be evaluated by Python `eval`.  To use the
            keyword as the file name, use `{root}` (in the above example {root}
            will be replaced with `tmp1`).

        Returns
        -------
        Same job object with resource groups set.
        """

        for name, d in mappings.items():
            assert name not in self._resources
            if not isinstance(d, dict):
                raise BatchException(f"value for name '{name}' is not a dict. Found '{type(d)}' instead.")
            rg = self._batch._new_resource_group(self, d, root=name)
            self._resources[name] = rg
            _add_resource_to_set(self._valid, rg)
        return self

    def image(self, image: str) -> 'BashJob':
        """
        Set the job's docker image.

        Examples
        --------

        Set the job's docker image to `ubuntu:22.04`:

        >>> b = Batch()
        >>> j = b.new_job()
        >>> (j.image('ubuntu:22.04')
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

        command = self._interpolate_command(command)
        self._command.append(command)
        return self

    async def _compile(self, local_tmpdir, remote_tmpdir, *, dry_run=False):
        if len(self._command) == 0:
            return False

        job_command = [cmd.strip() for cmd in self._command]
        job_command = [f'{{\n{x}\n}}' for x in job_command]
        job_command_str = '\n'.join(job_command)

        job_command_bytes = job_command_str.encode()

        if len(job_command_bytes) <= 10 * 1024:
            self._wrapper_code.append(job_command_str)
            return False

        self._user_code.append(job_command_str)

        job_path = f'{remote_tmpdir}/{self._dirname}'
        code_path = f'{job_path}/code.sh'
        code = self._batch.read_input(code_path)

        wrapper_command = f"""
chmod u+x {code}
source {code}
"""
        wrapper_command = self._interpolate_command(wrapper_command)
        self._wrapper_code.append(wrapper_command)

        if not dry_run:
            await self._batch._fs.makedirs(os.path.dirname(code_path), exist_ok=True)
            await self._batch._fs.write(code_path, job_command_bytes)

        return True


UnpreparedArg = Union[
    '_resource.ResourceType', List['UnpreparedArg'], Tuple['UnpreparedArg', ...], Dict[str, 'UnpreparedArg'], Any
]

PreparedArg = Union[
    Tuple[Literal['py_path'], str],
    Tuple[Literal['path'], str],
    Tuple[Literal['dict_path'], Dict[str, str]],
    Tuple[Literal['list'], List['PreparedArg']],
    Tuple[Literal['dict'], Dict[str, 'PreparedArg']],
    Tuple[Literal['tuple'], Tuple['PreparedArg', ...]],
    Tuple[Literal['value'], Any],
]


class PythonJob(Job):
    """
    Object representing a single Python job to execute.

    Examples
    --------

    Create a new Python job that multiplies two numbers and then adds 5 to the result:

    .. code-block:: python

        # Create a batch object with a default Python image

        b = Batch(default_python_image='hailgenetics/python-dill:3.9-slim')

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

    def __init__(
        self,
        batch: 'batch.Batch',
        token: str,
        *,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
    ):
        super().__init__(batch, token, name=name, attributes=attributes, shell=None)
        self._resources: Dict[str, _resource.Resource] = {}
        self._resources_inverse: Dict[_resource.Resource, str] = {}
        self._function_calls: List[
            Tuple[_resource.PythonResult, int, Tuple[UnpreparedArg, ...], Dict[str, UnpreparedArg]]
        ] = []
        self.n_results = 0

    def _get_python_resource(self, item: str) -> '_resource.PythonResult':
        if item not in self._resources:
            r = self._batch._new_python_result(self, value=item)
            self._resources[item] = r
            self._resources_inverse[r] = item
        return cast(_resource.PythonResult, self._resources[item])

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

        Set the job's docker image to `hailgenetics/python-dill:3.9-slim`:

        >>> b = Batch()
        >>> j = b.new_python_job()
        >>> (j.image('hailgenetics/python-dill:3.9-slim')
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

    def call(self, unapplied: Callable, *args: UnpreparedArg, **kwargs: UnpreparedArg) -> '_resource.PythonResult':
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

        if asyncio.iscoroutinefunction(unapplied):
            unapplied_copy = unapplied

            def run_async(*args, **kwargs):
                return asyncio.run(unapplied_copy(*args, **kwargs))

            unapplied = run_async

        for arg in args:
            if isinstance(arg, Job):
                raise BatchException('arguments to a PythonJob cannot be other job objects.')

        for value in kwargs.values():
            if isinstance(value, Job):
                raise BatchException('arguments to a PythonJob cannot be other job objects.')

        # Some builtins like `print` do not have signatures
        try:
            inspect.signature(unapplied).bind(*args, **kwargs)
        except ValueError as e:
            # Some builtins like `print` don't have a signature that inspect can read
            if 'no signature found for builtin' not in e.args[0]:
                raise e
        except TypeError as e:
            raise BatchException(f'Cannot call {unapplied.__name__} with the supplied arguments') from e

        def handle_arg(r):
            if r._source != self:
                self._add_inputs(r)
                if r._source is not None:
                    if r not in r._source._valid:
                        name = r._source._resources_inverse[r]
                        raise BatchException(f"undefined resource '{name}'\n")
                    self._dependencies.add(r._source)
                    r._source._add_internal_outputs(r)
            else:
                _add_resource_to_set(self._valid, r)

            self._mentioned.add(r)

        def handle_args(r):
            if isinstance(r, _resource.Resource):
                handle_arg(r)
            elif isinstance(r, (list, tuple)):
                for elt in r:
                    handle_args(elt)
            elif isinstance(r, dict):
                for v in r.values():
                    handle_args(v)

        handle_args(args)
        handle_args(kwargs)

        self.n_results += 1
        result = self._get_python_resource(f'result{self.n_results}')
        handle_arg(result)

        unapplied_id = self._batch._register_python_function(unapplied)

        self._function_calls.append((result, unapplied_id, args, kwargs))

        return result

    async def _compile(self, local_tmpdir, remote_tmpdir, *, dry_run=False):
        def preserialize(arg: UnpreparedArg) -> PreparedArg:
            if isinstance(arg, _resource.PythonResult):
                return ('py_path', arg._get_path(local_tmpdir))
            if isinstance(arg, _resource.ResourceFile):
                return ('path', arg._get_path(local_tmpdir))
            if isinstance(arg, _resource.ResourceGroup):
                return (
                    'dict_path',
                    {name: resource._get_path(local_tmpdir) for name, resource in arg._resources.items()},
                )
            if isinstance(arg, list):
                return ('list', [preserialize(elt) for elt in arg])
            if isinstance(arg, tuple):
                return ('tuple', tuple((preserialize(elt) for elt in arg)))
            if isinstance(arg, dict):
                return ('dict', {k: preserialize(v) for k, v in arg.items()})
            return ('value', arg)

        for i, (result, unapplied_id, args, kwargs) in enumerate(self._function_calls):
            func_file = self._batch._python_function_files[unapplied_id]

            preserialized_args = [preserialize(arg) for arg in args]
            del args
            preserialized_kwargs = {keyword: preserialize(arg) for keyword, arg in kwargs.items()}
            del kwargs

            args_file = await self._batch._serialize_python_to_input_file(
                os.path.dirname(result._get_path(remote_tmpdir)),
                "args",
                i,
                (preserialized_args, preserialized_kwargs),
                dry_run,
            )

            json_write, str_write, repr_write = [
                ''
                if not output
                else f"""
        with open('{output}', 'w') as out:
            out.write({formatter}(result) + '\\n')
"""
                for output, formatter in [(result._json, "json.dumps"), (result._str, "str"), (result._repr, "repr")]
            ]

            wrapper_code = f'''python3 -c "
import os
import base64
import dill
import traceback
import json
import sys

def deserialize_argument(arg):
    typ, val = arg
    if typ == 'py_path':
        return dill.load(open(val, 'rb'))
    if typ in ('path', 'dict_path'):
        return val
    if typ == 'list':
        return [deserialize_argument(elt) for elt in val]
    if typ == 'tuple':
        return tuple((deserialize_argument(elt) for elt in val))
    if typ == 'dict':
        return {{k: deserialize_argument(v) for k, v in val.items()}}
    assert typ == 'value'
    return val

with open('{result}', 'wb') as dill_out:
    try:
        with open('{func_file}', 'rb') as func_file:
            func = dill.load(func_file)
        with open('{args_file}', 'rb') as arg_file:
            args, kwargs = dill.load(arg_file)
            args = [deserialize_argument(arg) for arg in args]
            kwargs = {{k: deserialize_argument(v) for k, v in kwargs.items()}}
        result = func(*args, **kwargs)
        dill.dump(result, dill_out, recurse=True)
        {json_write}
        {str_write}
        {repr_write}
    except Exception as e:
        traceback.print_exc()
        dill.dump((e, traceback.format_exception(type(e), e, e.__traceback__)), dill_out, recurse=True)
        raise e
"'''

            wrapper_code = self._interpolate_command(wrapper_code, allow_python_results=True)
            self._wrapper_code.append(wrapper_code)

            unapplied = self._batch._python_function_defs[unapplied_id]
            self._user_code.append(textwrap.dedent(inspect.getsource(unapplied)))
            args_str = ', '.join([f'{arg!r}' for _, arg in preserialized_args])
            kwargs_str = ', '.join([f'{k}={v!r}' for k, (_, v) in preserialized_kwargs.items()])
            separator = ', ' if args_str and kwargs_str else ''
            func_call = f'{unapplied.__name__}({args_str}{separator}{kwargs_str})'
            self._user_code.append(self._interpolate_command(func_call, allow_python_results=True))

        return True
