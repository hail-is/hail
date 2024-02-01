from typing import Optional, Dict, Any, TypeVar, Generic, List, Union, ClassVar
import abc
import asyncio
import collections
import orjson
import os
import subprocess as sp
import uuid
import time
import functools
import copy
from shlex import quote as shq
import webbrowser
import warnings
from rich.progress import track

from hailtop import pip_version
from hailtop.config import ConfigVariable, configuration_of, get_deploy_config, get_remote_tmpdir
from hailtop.utils.rich_progress_bar import SimpleCopyToolProgressBar
from hailtop.utils.gcs_requester_pays import GCSRequesterPaysFSCache
from hailtop.utils import parse_docker_image_reference, async_to_blocking, bounded_gather, url_scheme
from hailtop.batch.hail_genetics_images import HAIL_GENETICS_IMAGES, hailgenetics_hail_image_for_current_python_version

from hailtop.batch_client.parse import parse_cpu_in_mcpu
import hailtop.batch_client.client as bc
from hailtop.batch_client.client import BatchClient
from hailtop.batch_client.aioclient import BatchClient as AioBatchClient
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration

from . import resource, batch  # pylint: disable=unused-import
from .job import PythonJob
from .exceptions import BatchException
from .globals import DEFAULT_SHELL
from hailtop.aiotools.validators import validate_file


HAIL_GENETICS_HAILTOP_IMAGE = os.environ.get('HAIL_GENETICS_HAILTOP_IMAGE', f'hailgenetics/hailtop:{pip_version()}')
HAIL_GENETICS_HAIL_IMAGE = (
    os.environ.get('HAIL_GENETICS_HAIL_IMAGE') or hailgenetics_hail_image_for_current_python_version()
)


RunningBatchType = TypeVar('RunningBatchType')
"""
The type of value returned by :py:meth:`.Backend._run`. The value returned by some backends
enables the user to monitor the asynchronous execution of a Batch.
"""

SelfType = TypeVar('SelfType')


class Backend(abc.ABC, Generic[RunningBatchType]):
    """
    Abstract class for backends.
    """

    _closed = False

    def __init__(self, requester_pays_fses: GCSRequesterPaysFSCache):
        self._requester_pays_fses = requester_pays_fses

    def requester_pays_fs(self, requester_pays_config: Optional[GCSRequesterPaysConfiguration]) -> RouterAsyncFS:
        return self._requester_pays_fses[requester_pays_config]

    def validate_file(self, uri: str, requester_pays_config: Optional[GCSRequesterPaysConfiguration] = None) -> None:
        self._validate_file(uri, self.requester_pays_fs(requester_pays_config))

    @abc.abstractmethod
    def _validate_file(self, uri: str, fs: RouterAsyncFS) -> None:
        raise NotImplementedError

    def _run(self, batch, dry_run, verbose, delete_scratch_on_exit, **backend_kwargs) -> Optional[RunningBatchType]:
        """
        See :meth:`._async_run`.

        Warning
        -------
        This method should not be called directly. Instead, use :meth:`.batch.Batch.run`.
        """
        return async_to_blocking(self._async_run(batch, dry_run, verbose, delete_scratch_on_exit, **backend_kwargs))

    @abc.abstractmethod
    async def _async_run(
        self, batch, dry_run, verbose, delete_scratch_on_exit, **backend_kwargs
    ) -> Optional[RunningBatchType]:
        """
        Execute a batch.

        Warning
        -------
        This method should not be called directly. Instead, use :meth:`.batch.Batch.run`.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _fs(self) -> RouterAsyncFS:
        raise NotImplementedError()

    @abc.abstractmethod
    async def _async_close(self):
        raise NotImplementedError()

    def close(self):
        """
        Close a Hail Batch Backend.

        Notes
        -----
        This method should be called after executing your batches at the
        end of your script.
        """
        async_to_blocking(self.async_close())

    async def async_close(self):
        if not self._closed:
            await self._async_close()
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self: SelfType) -> SelfType:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LocalBackend(Backend[None]):
    """
    Backend that executes batches on a local computer.
    Examples
    --------

    >>> local_backend = LocalBackend(tmp_dir='/tmp/user/')
    >>> b = Batch(backend=local_backend)

    Parameters
    ----------
    tmp_dir:
        Temporary directory to use.
    gsa_key_file:
        Mount a file with a gsa key to `/gsa-key/key.json`. Only used if a
        job specifies a docker image. This option will override the value set by
        the environment variable `HAIL_BATCH_GSA_KEY_FILE`.
    extra_docker_run_flags:
        Additional flags to pass to `docker run`. Only used if a job specifies
        a docker image. This option will override the value set by the environment
        variable `HAIL_BATCH_EXTRA_DOCKER_RUN_FLAGS`.
    """

    def __init__(
        self, tmp_dir: str = '/tmp/', gsa_key_file: Optional[str] = None, extra_docker_run_flags: Optional[str] = None
    ):
        super().__init__(GCSRequesterPaysFSCache(fs_constructor=RouterAsyncFS))
        self._tmp_dir = tmp_dir.rstrip('/')

        flags = ''

        if extra_docker_run_flags is not None:
            flags += extra_docker_run_flags
        elif os.environ.get('HAIL_BATCH_EXTRA_DOCKER_RUN_FLAGS') is not None:
            flags += os.environ['HAIL_BATCH_EXTRA_DOCKER_RUN_FLAGS']

        if gsa_key_file is None:
            gsa_key_file = os.environ.get('HAIL_BATCH_GSA_KEY_FILE')
        if gsa_key_file is not None:
            flags += f' -v {gsa_key_file}:/gsa-key/key.json'

        self._extra_docker_run_flags = flags
        self.__fs = self._requester_pays_fses[None]

    @property
    def _fs(self) -> RouterAsyncFS:
        return self.__fs

    def _validate_file(self, uri: str, fs: RouterAsyncFS) -> None:
        validate_file(uri, fs)

    async def _async_run(
        self, batch: 'batch.Batch', dry_run: bool, verbose: bool, delete_scratch_on_exit: bool, **backend_kwargs
    ) -> None:  # pylint: disable=R0915
        """
        Execute a batch.

        Warning
        -------
        This method should not be called directly. Instead, use :meth:`.batch.Batch.run`.

        Parameters
        ----------
        batch:
            Batch to execute.
        dry_run:
            If `True`, don't execute code.
        verbose:
            If `True`, print debugging output.
        delete_scratch_on_exit:
            If `True`, delete temporary directories with intermediate files.
        """

        if backend_kwargs:
            raise ValueError(f'LocalBackend does not support any of these keywords: {backend_kwargs}')

        tmpdir = self._get_scratch_dir()

        def new_code_block():
            return ['set -e' + ('x' if verbose else ''), '\n', '# change cd to tmp directory', f"cd {tmpdir}", '\n']

        def run_code(code) -> Optional[sp.CalledProcessError]:
            code = '\n'.join(code)
            if dry_run:
                print(code)
            else:
                try:
                    sp.check_call(code, shell=True)
                except sp.CalledProcessError as e:
                    print(e)
                    print(e.output)
                    return e
            return None

        copied_input_resource_files = set()
        os.makedirs(tmpdir + '/inputs/', exist_ok=True)

        requester_pays_project_json = orjson.dumps(batch.requester_pays_project).decode('utf-8')

        def copy_input(job, r):
            if isinstance(r, resource.InputResourceFile):
                if r not in copied_input_resource_files:
                    copied_input_resource_files.add(r)

                    assert r._input_path
                    input_scheme = url_scheme(r._input_path)
                    if input_scheme != '':
                        transfers_bytes = orjson.dumps([{"from": r._input_path, "to": r._get_path(tmpdir)}])
                        transfers = transfers_bytes.decode('utf-8')
                        return [f'python3 -m hailtop.aiotools.copy {shq(requester_pays_project_json)} {shq(transfers)}']

                    absolute_input_path = os.path.realpath(os.path.expanduser(r._input_path))

                    dest = r._get_path(os.path.expanduser(tmpdir))
                    dir = os.path.dirname(dest)
                    os.makedirs(dir, exist_ok=True)

                    if job._image is not None:  # pylint: disable-msg=W0640
                        return [f'cp {shq(absolute_input_path)} {shq(dest)}']

                    return [f'ln -sf {shq(absolute_input_path)} {shq(dest)}']

                return []

            assert isinstance(r, (resource.JobResourceFile, resource.PythonResult))
            return []

        def symlink_input_resource_group(r):
            symlinks = []
            if isinstance(r, resource.ResourceGroup) and r._source is None:
                for name, irf in r._resources.items():
                    src = irf._get_path(tmpdir)
                    dest = f'{r._get_path(tmpdir)}.{name}'
                    symlinks.append(f'ln -sf {shq(src)} {shq(dest)}')
            return symlinks

        def transfer_dicts_for_resource_file(
            res_file: Union[resource.ResourceFile, resource.PythonResult],
        ) -> List[dict]:
            if isinstance(res_file, resource.InputResourceFile):
                source = res_file._input_path
            else:
                assert isinstance(res_file, (resource.JobResourceFile, resource.PythonResult))
                source = res_file._get_path(tmpdir)

            return [{"from": source, "to": dest} for dest in res_file._output_paths]

        try:
            input_transfer_dicts = [
                transfer_dict
                for input_resource in batch._input_resources
                for transfer_dict in transfer_dicts_for_resource_file(input_resource)
            ]

            if input_transfer_dicts:
                input_transfers = orjson.dumps(input_transfer_dicts).decode('utf-8')
                code = new_code_block()
                code += ["# Write input resources to output destinations"]
                code += [f'python3 -m hailtop.aiotools.copy {shq(requester_pays_project_json)} {shq(input_transfers)}']
                code += ['\n']
                run_code(code)

            await batch._serialize_python_functions_to_input_files(tmpdir, dry_run=dry_run)

            jobs = batch._unsubmitted_jobs
            child_jobs = collections.defaultdict(set)

            for j in jobs:
                for parent in j._dependencies:
                    child_jobs[parent].add(j)

            cancelled_jobs = set()

            def cancel_child_jobs(j):
                for child in child_jobs[j]:
                    if not child._always_run:
                        cancelled_jobs.add(child)

            first_exc = None
            for job in jobs:
                if job in cancelled_jobs:
                    print(f'Job {job} was cancelled. Not running')
                    cancel_child_jobs(job)
                    continue
                await job._compile(tmpdir, tmpdir, dry_run=dry_run)

                os.makedirs(f'{tmpdir}/{job._dirname}/', exist_ok=True)

                code = new_code_block()

                code.append(f"# {job._job_id}: {job.name if job.name else ''}")

                if job._user_code:
                    code.append('# USER CODE')
                    user_code = [f'# {line}' for cmd in job._user_code for line in cmd.split('\n')]
                    code.append('\n'.join(user_code))

                code += [x for r in job._inputs for x in copy_input(job, r)]
                code += [x for r in job._mentioned for x in symlink_input_resource_group(r)]

                env = {**job._env, 'BATCH_TMPDIR': tmpdir}
                env_declarations = [f'export {k}={v}' for k, v in env.items()]
                joined_env = '; '.join(env_declarations) + '; ' if env else ''

                job_shell = job._shell if job._shell else DEFAULT_SHELL

                cmd = " && ".join(f'{{\n{x}\n}}' for x in job._wrapper_code)

                quoted_job_script = shq(joined_env + cmd)

                if job._image:
                    cpu = f'--cpus={job._cpu}' if job._cpu else ''

                    memory = job._memory
                    if memory is not None:
                        memory_ratios = {'lowmem': 1024**3, 'standard': 4 * 1024**3, 'highmem': 7 * 1024**3}
                        if memory in memory_ratios:
                            if job._cpu is not None:
                                mcpu = parse_cpu_in_mcpu(job._cpu)
                                if mcpu is not None:
                                    memory = str(int(memory_ratios[memory] * (mcpu / 1000)))
                                else:
                                    raise BatchException(f'invalid value for cpu: {job._cpu}')
                            else:
                                raise BatchException(f'must specify cpu when using {memory} to specify the memory')
                        memory = f'-m {memory}' if memory else ''
                    else:
                        memory = ''

                    code.append(
                        f"docker run "
                        "--entrypoint=''"
                        f"{self._extra_docker_run_flags} "
                        f"-v {tmpdir}:{tmpdir} "
                        f"{memory} "
                        f"{cpu} "
                        f"{job._image} "
                        f"{job_shell} -c {quoted_job_script}"
                    )
                else:
                    code.append(f"{job_shell} -c {quoted_job_script}")

                output_transfer_dicts = [
                    transfer_dict
                    for output_resource in job._external_outputs
                    for transfer_dict in transfer_dicts_for_resource_file(output_resource)
                ]

                if output_transfer_dicts:
                    output_transfers = orjson.dumps(output_transfer_dicts).decode('utf-8')
                    code += [
                        f'python3 -m hailtop.aiotools.copy {shq(requester_pays_project_json)} {shq(output_transfers)}'
                    ]
                code += ['\n']

                exc = run_code(code)
                if exc is not None:
                    first_exc = exc if first_exc is None else first_exc
                    cancel_child_jobs(job)
                job._submitted = True
        finally:
            batch._python_function_defs.clear()
            batch._python_function_files.clear()
            if delete_scratch_on_exit:
                sp.run(f'rm -rf {tmpdir}', shell=True, check=False)

        if first_exc is not None:
            raise first_exc
        print('Batch completed successfully!')

    def _get_scratch_dir(self):
        def _get_random_name():
            dir = f'{self._tmp_dir}/batch/{uuid.uuid4().hex[:6]}'
            if os.path.isdir(dir):
                return _get_random_name()
            os.makedirs(dir, exist_ok=True)
            return dir

        return _get_random_name()

    async def _async_close(self):
        await self._fs.close()


class ServiceBackend(Backend[bc.Batch]):
    """Backend that executes batches on Hail's Batch Service on Google Cloud.

    Examples
    --------

    Create and use a backend that bills to the Hail Batch billing project named "my-billing-account"
    and stores temporary intermediate files in "gs://my-bucket/temporary-files".

    >>> import hailtop.batch as hb
    >>> service_backend = hb.ServiceBackend(
    ...     billing_project='my-billing-account',
    ...     remote_tmpdir='gs://my-bucket/temporary-files/'
    ... )  # doctest: +SKIP
    >>> b = hb.Batch(backend=service_backend)  # doctest: +SKIP
    >>> j = b.new_job()  # doctest: +SKIP
    >>> j.command('echo hello world!')  # doctest: +SKIP
    >>> b.run() # doctest: +SKIP

    Same as above, but set the billing project and temporary intermediate folders via a
    configuration file::

        cat >my-batch-script.py >>EOF
        import hailtop.batch as hb
        b = hb.Batch(backend=ServiceBackend())
        j = b.new_job()
        j.command('echo hello world!')
        b.run()
        EOF
        hailctl config set batch/billing_project my-billing-account
        hailctl config set batch/remote_tmpdir gs://my-bucket/temporary-files/
        python3 my-batch-script.py

    Same as above, but also specify the use of the :class:`.ServiceBackend` via configuration file::

        cat >my-batch-script.py >>EOF
        import hailtop.batch as hb
        b = hb.Batch()
        j = b.new_job()
        j.command('echo hello world!')
        b.run()
        EOF
        hailctl config set batch/billing_project my-billing-account
        hailctl config set batch/remote_tmpdir gs://my-bucket/temporary-files/
        hailctl config set batch/backend service
        python3 my-batch-script.py

    Create a backend which stores temporary intermediate files in
    "https://my-account.blob.core.windows.net/my-container/tempdir".

    >>> service_backend = hb.ServiceBackend(
    ...     billing_project='my-billing-account',
    ...     remote_tmpdir='https://my-account.blob.core.windows.net/my-container/tempdir'
    ... )  # doctest: +SKIP

    Require all jobs in all batches in this backend to execute in us-central1::

    >>> b = hb.Batch(backend=hb.ServiceBackend(regions=['us-central1']))

    Same as above, but using a configuration file::

        hailctl config set batch/regions us-central1
        python3 my-batch-script.py

    Same as above, but using the ``HAIL_BATCH_REGIONS`` environment variable::

        export HAIL_BATCH_REGIONS=us-central1
        python3 my-batch-script.py

    Permit jobs to execute in *either* us-central1 or us-east1::

    >>> b = hb.Batch(backend=hb.ServiceBackend(regions=['us-central1', 'us-east1']))

    Same as a bove, but using a configuration file::

        hailctl config set batch/regions us-central1,us-east1

    Allow reading or writing to buckets even though they are "cold" storage:

    >>> b = hb.Batch(
    ...     backend=hb.ServiceBackend(
    ...         gcs_bucket_allow_list=['cold-bucket', 'cold-bucket2'],
    ...     ),
    ... )

    Parameters
    ----------
    billing_project:
        Name of billing project to use.
    bucket:
        This argument is deprecated. Use `remote_tmpdir` instead.
    remote_tmpdir:
        Temporary data will be stored in this cloud storage folder.
    google_project:
        This argument is deprecated. Use `gcs_requester_pays_configuration` instead.
    gcs_requester_pays_configuration : either :class:`str` or :class:`tuple` of :class:`str` and :class:`list` of :class:`str`, optional
        If a string is provided, configure the Google Cloud Storage file system to bill usage to the
        project identified by that string. If a tuple is provided, configure the Google Cloud
        Storage file system to bill usage to the specified project for buckets specified in the
        list.
    token:
        The authorization token to pass to the batch client.
        Should only be set for user delegation purposes.
    regions:
        Cloud regions in which jobs may run. :attr:`.ServiceBackend.ANY_REGION` indicates jobs may
        run in any region. If unspecified or ``None``, the ``batch/regions`` Hail configuration
        variable is consulted. See examples above. If none of these variables are set, then jobs may
        run in any region. :meth:`.ServiceBackend.supported_regions` lists the available regions.
    gcs_bucket_allow_list:
        A list of buckets that the :class:`.ServiceBackend` should be permitted to read from or write to, even if their
        default policy is to use "cold" storage.

    """

    ANY_REGION: ClassVar[List[str]] = ['any_region']
    """A special value that indicates a job may run in any region."""

    @staticmethod
    def supported_regions():
        """
        Get the supported cloud regions

        Examples
        --------
        >>> regions = ServiceBackend.supported_regions()

        Returns
        -------
        A list of the supported cloud regions
        """
        with BatchClient('dummy') as dummy_client:
            return dummy_client.supported_regions()

    def __init__(
        self,
        *args,
        billing_project: Optional[str] = None,
        bucket: Optional[str] = None,
        remote_tmpdir: Optional[str] = None,
        google_project: Optional[str] = None,
        token: Optional[str] = None,
        regions: Optional[List[str]] = None,
        gcs_requester_pays_configuration: Optional[GCSRequesterPaysConfiguration] = None,
        gcs_bucket_allow_list: Optional[List[str]] = None,
    ):
        if len(args) > 2:
            raise TypeError(f'ServiceBackend() takes 2 positional arguments but {len(args)} were given')
        if len(args) >= 1:
            if billing_project is not None:
                raise TypeError('ServiceBackend() got multiple values for argument \'billing_project\'')
            warnings.warn(
                'Use of deprecated positional argument \'billing_project\' in ServiceBackend(). Specify \'billing_project\' as a keyword argument instead.'
            )
            billing_project = args[0]
        if len(args) >= 2:
            if bucket is not None:
                raise TypeError('ServiceBackend() got multiple values for argument \'bucket\'')
            warnings.warn(
                'Use of deprecated positional argument \'bucket\' in ServiceBackend(). Specify \'bucket\' as a keyword argument instead.'
            )
            bucket = args[1]

        billing_project = configuration_of(ConfigVariable.BATCH_BILLING_PROJECT, billing_project, None)
        if billing_project is None:
            raise ValueError(
                'the billing_project parameter of ServiceBackend must be set '
                'or run `hailctl config set batch/billing_project '
                'MY_BILLING_PROJECT`'
            )
        self._billing_project = billing_project
        self._token = token

        self.remote_tmpdir = get_remote_tmpdir('ServiceBackend', bucket=bucket, remote_tmpdir=remote_tmpdir)

        gcs_kwargs: Dict[str, Any]
        if google_project is not None:
            if gcs_requester_pays_configuration is not None:
                raise ValueError(
                    'Both google_project and gcs_requester_pays_configuration were '
                    'specified. Please remove the google_project argument.'
                )
            warnings.warn(
                'The google_project parameter of ServiceBackend is deprecated. '
                'Please replace with gcs_requester_pays_configuration.'
            )
            gcs_kwargs = {'gcs_requester_pays_configuration': google_project}
        else:
            gcs_kwargs = {'gcs_requester_pays_configuration': gcs_requester_pays_configuration}

        super().__init__(
            GCSRequesterPaysFSCache(
                fs_constructor=RouterAsyncFS,
                default_kwargs={"gcs_kwargs": gcs_kwargs, "gcs_bucket_allow_list": gcs_bucket_allow_list},
            )
        )

        self.__fs = self._requester_pays_fses[None]

        self.validate_file(self.remote_tmpdir)

        if regions is None:
            regions_from_conf = configuration_of(ConfigVariable.BATCH_REGIONS, None, None)
            if regions_from_conf is not None:
                assert isinstance(regions_from_conf, str)
                regions = regions_from_conf.split(',')
        elif regions == ServiceBackend.ANY_REGION:
            regions = None
        self.regions = regions
        self.__batch_client: Optional[AioBatchClient] = None

    async def _batch_client(self) -> AioBatchClient:
        if self.__batch_client is None:
            self.__batch_client = await AioBatchClient.create(self._billing_project, _token=self._token)
        return self.__batch_client

    @property
    def _fs(self) -> RouterAsyncFS:
        return self.__fs

    def _validate_file(self, uri: str, fs: RouterAsyncFS) -> None:
        validate_file(uri, fs, validate_scheme=True)

    def _close(self):
        async_to_blocking(self._async_close())

    async def _async_close(self):
        if self.__batch_client is not None:
            await self.__batch_client.close()
        await self._fs.close()

    async def _async_run(
        self,
        batch: 'batch.Batch',
        dry_run: bool,
        verbose: bool,
        delete_scratch_on_exit: bool,
        wait: bool = True,
        open: bool = False,
        disable_progress_bar: bool = False,
        callback: Optional[str] = None,
        token: Optional[str] = None,
        **backend_kwargs,
    ) -> Optional[bc.Batch]:  # pylint: disable-msg=too-many-statements
        """Execute a batch.

        Warning
        -------
        This method should not be called directly. Instead, use :meth:`.batch.Batch.run`
        and pass :class:`.ServiceBackend` specific arguments as key-word arguments.

        Parameters
        ----------
        batch:
            Batch to execute.
        dry_run:
            If `True`, don't execute code.
        verbose:
            If `True`, print debugging output.
        delete_scratch_on_exit:
            If `True`, delete temporary directories with intermediate files.
        wait:
            If `True`, wait for the batch to finish executing before returning.
        open:
            If `True`, open the UI page for the batch.
        disable_progress_bar:
            If `True`, disable the progress bar.
        callback:
            If not `None`, a URL that will receive at most one POST request
            after the entire batch completes.
        token:
            If not `None`, a string used for idempotency of batch submission.
        """
        if backend_kwargs:
            raise ValueError(f'ServiceBackend does not support any of these keywords: {backend_kwargs}')

        build_dag_start = time.time()

        uid = uuid.uuid4().hex[:6]
        batch_remote_tmpdir = f'{self.remote_tmpdir}{uid}'
        local_tmpdir = f'/io/batch/{uid}'

        default_image = 'ubuntu:22.04'

        attributes = copy.deepcopy(batch.attributes)
        if batch.name is not None:
            attributes['name'] = batch.name

        if batch._async_batch is None:
            batch._async_batch = (await self._batch_client()).create_batch(
                attributes=attributes,
                callback=callback,
                token=token,
                cancel_after_n_failures=batch._cancel_after_n_failures,
            )

        async_batch = batch._async_batch

        n_jobs_submitted = 0
        used_remote_tmpdir = False

        jobs_to_command = {}
        commands = []

        bash_flags = 'set -e' + ('x' if verbose else '')

        def copy_input(r):
            if isinstance(r, resource.InputResourceFile):
                return [(r._input_path, r._get_path(local_tmpdir))]
            assert isinstance(r, (resource.JobResourceFile, resource.PythonResult))
            return [(r._get_path(batch_remote_tmpdir), r._get_path(local_tmpdir))]

        def copy_internal_output(r):
            assert isinstance(r, (resource.JobResourceFile, resource.PythonResult))
            return [(r._get_path(local_tmpdir), r._get_path(batch_remote_tmpdir))]

        def copy_external_output(r):
            if isinstance(r, resource.InputResourceFile):
                return [(r._input_path, dest) for dest in r._output_paths]
            assert isinstance(r, (resource.JobResourceFile, resource.PythonResult))
            return [(r._get_path(local_tmpdir), dest) for dest in r._output_paths]

        def symlink_input_resource_group(r):
            symlinks = []
            if isinstance(r, resource.ResourceGroup) and r._source is None:
                for name, irf in r._resources.items():
                    src = irf._get_path(local_tmpdir)
                    dest = f'{r._get_path(local_tmpdir)}.{name}'
                    symlinks.append(f'ln -sf {shq(src)} {shq(dest)}')
            return symlinks

        write_external_inputs = [x for r in batch._input_resources for x in copy_external_output(r)]
        if write_external_inputs:
            transfers_bytes = orjson.dumps([{"from": src, "to": dest} for src, dest in write_external_inputs])
            transfers = transfers_bytes.decode('utf-8')
            write_cmd = ['python3', '-m', 'hailtop.aiotools.copy', 'null', transfers]
            if dry_run:
                commands.append(' '.join(shq(x) for x in write_cmd))
            else:
                j = async_batch.create_job(
                    image=HAIL_GENETICS_HAILTOP_IMAGE, command=write_cmd, attributes={'name': 'write_external_inputs'}
                )
                jobs_to_command[j] = ' '.join(shq(x) for x in write_cmd)
                n_jobs_submitted += 1

        unsubmitted_jobs = batch._unsubmitted_jobs

        pyjobs = [j for j in unsubmitted_jobs if isinstance(j, PythonJob)]
        for pyjob in pyjobs:
            if pyjob._image is None:
                pyjob._image = HAIL_GENETICS_HAIL_IMAGE
        await batch._serialize_python_functions_to_input_files(batch_remote_tmpdir, dry_run=dry_run)

        disable_setup_steps_progress_bar = disable_progress_bar or len(unsubmitted_jobs) < 10_000
        with SimpleCopyToolProgressBar(
            total=len(unsubmitted_jobs), description='upload code', disable=disable_setup_steps_progress_bar
        ) as pbar:

            async def compile_job(job):
                used_remote_tmpdir = await job._compile(local_tmpdir, batch_remote_tmpdir, dry_run=dry_run)
                pbar.update(1)
                return used_remote_tmpdir

            used_remote_tmpdir_results = await bounded_gather(
                *[functools.partial(compile_job, j) for j in unsubmitted_jobs],
                parallelism=150,
                cancel_on_error=True,
            )
            used_remote_tmpdir |= any(used_remote_tmpdir_results)

        for job in track(unsubmitted_jobs, description='create job objects', disable=disable_setup_steps_progress_bar):
            inputs = [x for r in job._inputs for x in copy_input(r)]

            outputs = [x for r in job._internal_outputs for x in copy_internal_output(r)]
            if outputs:
                used_remote_tmpdir = True
            outputs += [x for r in job._external_outputs for x in copy_external_output(r)]

            symlinks = [x for r in job._mentioned for x in symlink_input_resource_group(r)]

            if job._image is None:
                if verbose:
                    print(f"Using image '{default_image}' since no image was specified.")

            make_local_tmpdir = f'mkdir -p {local_tmpdir}/{job._dirname}'

            job_command = [cmd.strip() for cmd in job._wrapper_code]
            prepared_job_command = (f'{{\n{x}\n}}' for x in job_command)
            cmd = f"""
{bash_flags}
{make_local_tmpdir}
{"; ".join(symlinks)}
{" && ".join(prepared_job_command)}
"""

            user_code = '\n\n'.join(job._user_code) if job._user_code else None

            if dry_run:
                formatted_command = f"""
================================================================================
# Job {job._job_id} {f": {job.name}" if job.name else ''}

--------------------------------------------------------------------------------
## USER CODE
--------------------------------------------------------------------------------
{user_code}

--------------------------------------------------------------------------------
## COMMAND
--------------------------------------------------------------------------------
{cmd}
================================================================================
"""
                commands.append(formatted_command)
                continue

            parents = [j._client_job for j in job._dependencies]

            attributes = copy.deepcopy(job.attributes) if job.attributes else {}
            if job.name:
                attributes['name'] = job.name

            resources: Dict[str, Any] = {}
            if job._cpu:
                resources['cpu'] = job._cpu
            if job._memory:
                resources['memory'] = job._memory
            if job._storage:
                resources['storage'] = job._storage
            if job._machine_type:
                resources['machine_type'] = job._machine_type
            if job._preemptible is not None:
                resources['preemptible'] = job._preemptible

            image = job._image if job._image else default_image
            image_ref = parse_docker_image_reference(image)
            if image_ref.hosted_in('dockerhub') and image_ref.name() not in HAIL_GENETICS_IMAGES:
                warnings.warn(
                    f'Using an image {image} from Docker Hub. ' f'Jobs may fail due to Docker Hub rate limits.'
                )

            env = {**job._env, 'BATCH_TMPDIR': local_tmpdir}

            j = async_batch.create_job(
                image=image,
                command=[job._shell if job._shell else DEFAULT_SHELL, '-c', cmd],
                parents=[parent._async_job for parent in parents],  # type: ignore
                attributes=attributes,
                resources=resources,
                input_files=inputs if len(inputs) > 0 else None,
                output_files=outputs if len(outputs) > 0 else None,
                always_run=job._always_run,
                timeout=job._timeout,
                cloudfuse=job._cloudfuse if len(job._cloudfuse) > 0 else None,
                env=env,
                requester_pays_project=batch.requester_pays_project,
                user_code=user_code,
                regions=job._regions,
                always_copy_output=job._always_copy_output,
            )

            n_jobs_submitted += 1

            job._client_job = bc.Job(j)
            jobs_to_command[j] = cmd

        if dry_run:
            print("\n\n".join(commands))
            return None

        if delete_scratch_on_exit and used_remote_tmpdir:
            parents = list(jobs_to_command.keys())
            j = async_batch.create_job(
                image=HAIL_GENETICS_HAILTOP_IMAGE,
                command=['python3', '-m', 'hailtop.aiotools.delete', batch_remote_tmpdir],
                parents=parents,
                resources={'cpu': '0.25'},
                attributes={'name': 'remove_tmpdir'},
                always_run=True,
            )
            n_jobs_submitted += 1

        if verbose:
            print(f'Built DAG with {n_jobs_submitted} jobs in {round(time.time() - build_dag_start, 3)} seconds.')

        submit_batch_start = time.time()
        await async_batch.submit(disable_progress_bar=disable_progress_bar)

        batch_id = async_batch.id

        for job in batch._unsubmitted_jobs:
            job._submitted = True

        jobs_to_command = {j.id: cmd for j, cmd in jobs_to_command.items()}

        if verbose:
            print(
                f'Submitted batch {batch_id} with {n_jobs_submitted} jobs in {round(time.time() - submit_batch_start, 3)} seconds:'
            )
            for jid, cmd in jobs_to_command.items():
                print(f'{jid}: {cmd}')
            print('')

        deploy_config = get_deploy_config()
        url = deploy_config.external_url('batch', f'/batches/{batch_id}')

        if open:
            webbrowser.open(url)
        if wait and len(unsubmitted_jobs) > 0:
            if verbose:
                print(f'Waiting for batch {batch_id}...')
            starting_job_id: int = min(j._client_job.job_id for j in unsubmitted_jobs)  # type: ignore
            await asyncio.sleep(0.6)  # it is not possible for the batch to be finished in less than 600ms
            status = await async_batch.wait(disable_progress_bar=disable_progress_bar, starting_job=starting_job_id)
            print(f'batch {batch_id} complete: {status["state"]}')

        batch._python_function_defs.clear()
        batch._python_function_files.clear()
        return bc.Batch(async_batch)
