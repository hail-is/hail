from typing import Optional, Dict, Any, TypeVar, Generic, List, Tuple
import asyncio
import sys
import tempfile
import abc
import orjson
import os
import subprocess as sp
import uuid
import time
import functools
from shlex import quote as shq
import webbrowser
import warnings
from io import StringIO

import hailtop
from hailtop import pip_version
from hailtop.config import get_deploy_config, get_user_config
from hailtop.utils import parse_docker_image_reference, async_to_blocking, bounded_gather, tqdm
from hailtop.batch.hail_genetics_images import HAIL_GENETICS_IMAGES
from hailtop.batch_client.parse import parse_cpu_in_mcpu
import hailtop.batch_client.aioclient as aiobc
import hailtop.batch_client.client as bc
from hailtop.aiotools import AsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.aiotools.copy import copy_from_dict

from . import resource, batch, job as _job  # pylint: disable=unused-import
from .exceptions import BatchException
from .globals import DEFAULT_SHELL

HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE',
                                          f'hailgenetics/hail:{pip_version()}')


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

    @abc.abstractmethod
    def _run(self, batch, dry_run, verbose, delete_scratch_on_exit, **backend_kwargs) -> RunningBatchType:
        """
        Execute a batch.

        Warning
        -------
        This method should not be called directly. Instead, use :meth:`.batch.Batch.run`.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _fs(self) -> AsyncFS:
        raise NotImplementedError()

    @abc.abstractmethod
    def local_tmpdir(self) -> str:
        pass

    @abc.abstractmethod
    def remote_tmpdir(self) -> str:
        pass

    def _close(self):  # pylint: disable=R0201
        return

    def close(self):  # pylint: disable=R0201
        """
        Close a Hail Batch Backend.

        Notes
        -----
        This method should be called after executing your batches at the
        end of your script.
        """
        if not self._closed:
            self._close()
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

    def __init__(self,
                 tmp_dir: str = '/tmp/',
                 gsa_key_file: Optional[str] = None,
                 extra_docker_run_flags: Optional[str] = None):
        self._tmp_dir = tmp_dir.rstrip('/')  # FIXME: what is this for?
        self._remote_tmpdir = self._get_scratch_dir()

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
        self.__fs: AsyncFS = RouterAsyncFS(default_scheme='file')

    def remote_tmpdir(self) -> str:
        return self._remote_tmpdir

    @property
    def _fs(self):
        return self.__fs

    def _run(self,
             batch: 'batch.Batch',
             dry_run: bool,
             verbose: bool,
             delete_scratch_on_exit: bool,
             **backend_kwargs) -> None:  # pylint: disable=R0915
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
        async_to_blocking(self._async_run(batch, dry_run, verbose, delete_scratch_on_exit, **backend_kwargs))

    async def _async_run(self,
                         batch: 'batch.Batch',
                         dry_run: bool,
                         verbose: bool,
                         delete_scratch_on_exit: bool,
                         **backend_kwargs) -> None:  # pylint: disable=R0915
        if backend_kwargs:
            raise ValueError(f'LocalBackend does not support any of these keywords: {backend_kwargs}')

        requester_pays_project_json = orjson.dumps(batch.requester_pays_project).decode('utf-8')

        async def generate_job_code(job) -> str:
            main_code, user_code, resources_to_download = await job.compile()
            code = StringIO()
            for r in resources_to_download:
                transfers_bytes = orjson.dumps([{
                    "from": r.remote_location(),
                    "to": r.local_location()}])
                transfers = transfers_bytes.decode('utf-8')
                code.write(
                    f'/Users/dking/miniconda3/bin/python3 -m hailtop.aiotools.copy {shq(requester_pays_project_json)} {shq(transfers)}\n')
            code.write(main_code)
            return code.getvalue()

        try:
            job_codes = await asyncio.gather(*[generate_job_code(job) for job in batch._jobs])
            for code, job in zip(job_codes, batch._jobs):
                if dry_run:
                    print(code)
                elif job._image is None:
                    with tempfile.TemporaryDirectory() as workdir:
                        try:
                            sp.run(
                                [job._shell or DEFAULT_SHELL, '-c', code],
                                check=True,
                                cwd=workdir,
                                env=(job._env or {})  # FIXME: do I need BATCH_TMPDIR ?
                            )
                        except sp.CalledProcessError as err:
                            raise ValueError((err.output, err.stderr)) from err
                elif job._image:
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

                    command = [
                        'docker',
                        'run',
                        '--entrypoint=' + (job._shell or DEFAULT_SHELL),
                        *self._extra_docker_run_flags.split(' '),
                        '-v', batch._remote_location + ':' + batch._remote_location,
                    ]
                    if job._cpu is not None:
                        command.extend(['--cpus', job._cpu])
                    if memory is not None:
                        command.extend(['-m', memory])
                    command.extend(['--', '-c', code])

                    with tempfile.TemporaryDirectory() as workdir:
                        sp.run(command, check=True, cwd=workdir, env=job._env)  # FIXME: do I need BATCH_TMPDIR ?

                await copy_from_dict(
                    max_simultaneous_transfers=300,
                    files=[
                        {'from': resource.local_location(), 'to': destination}
                        for resource, destinations in batch._outputs.items()
                        for destination in destinations])
        finally:
            if delete_scratch_on_exit:
                sp.run(['rm', '-rf', batch._remote_location], check=False)

        print('Batch completed successfully!')

    def _get_scratch_dir(self):
        def _get_random_name():
            dir = f'{self._tmp_dir}/batch/{uuid.uuid4().hex[:6]}'
            if os.path.isdir(dir):
                return _get_random_name()
            os.makedirs(dir, exist_ok=True)
            return dir

        return _get_random_name() + '/'

    def local_tmpdir(self) -> str:
        return self._get_scratch_dir()

    def _close(self):
        async_to_blocking(self._fs.close())


class ServiceBackend(Backend[bc.Batch]):
    """Backend that executes batches on Hail's Batch Service on Google Cloud.

    Examples
    --------

    >>> service_backend = ServiceBackend(billing_project='my-billing-account', remote_tmpdir='gs://my-bucket/temporary-files/') # doctest: +SKIP
    >>> b = Batch(backend=service_backend) # doctest: +SKIP
    >>> b.run() # doctest: +SKIP
    >>> service_backend.close() # doctest: +SKIP

    If the Hail configuration parameters batch/billing_project and
    batch/remote_tmpdir were previously set with ``hailctl config set``, then
    one may elide the `billing_project` and `remote_tmpdir` parameters.

    >>> service_backend = ServiceBackend()
    >>> b = Batch(backend=service_backend)
    >>> b.run() # doctest: +SKIP
    >>> service_backend.close()


    Parameters
    ----------
    billing_project:
        Name of billing project to use.
    bucket:
        Name of bucket to use. Should not include the ``gs://`` prefix. Cannot be used with
        `remote_tmpdir`. Temporary data will be stored in the "/batch" folder of this
        bucket. This argument is deprecated. Use `remote_tmpdir` instead.
    remote_tmpdir:
        Temporary data will be stored in this cloud storage folder. Cannot be used with deprecated
        argument `bucket`. Paths should start with one of gs://, hail-az://, or s3://.
    google_project:
        If specified, the project to use when authenticating with Google
        Storage. Google Storage is used to transfer serialized values between
        this computer and the cloud machines that execute Python jobs.
    token:
        The authorization token to pass to the batch client.
        Should only be set for user delegation purposes.
    """

    def __init__(self,
                 *args,
                 billing_project: Optional[str] = None,
                 bucket: Optional[str] = None,
                 remote_tmpdir: Optional[str] = None,
                 google_project: Optional[str] = None,
                 token: Optional[str] = None
                 ):
        if len(args) > 2:
            raise TypeError(f'ServiceBackend() takes 2 positional arguments but {len(args)} were given')
        if len(args) >= 1:
            if billing_project is not None:
                raise TypeError('ServiceBackend() got multiple values for argument \'billing_project\'')
            warnings.warn('Use of deprecated positional argument \'billing_project\' in ServiceBackend(). Specify \'billing_project\' as a keyword argument instead.')
            billing_project = args[0]
        if len(args) >= 2:
            if bucket is not None:
                raise TypeError('ServiceBackend() got multiple values for argument \'bucket\'')
            warnings.warn('Use of deprecated positional argument \'bucket\' in ServiceBackend(). Specify \'bucket\' as a keyword argument instead.')
            bucket = args[1]

        if billing_project is None:
            billing_project = get_user_config().get('batch', 'billing_project', fallback=None)
        if billing_project is None:
            raise ValueError(
                'the billing_project parameter of ServiceBackend must be set '
                'or run `hailctl config set batch/billing_project '
                'MY_BILLING_PROJECT`')

        self.batch_client = async_to_blocking(aiobc.BatchClient.create(billing_project, _token=token))
        user_config = get_user_config()

        if bucket is not None:
            warnings.warn('Use of deprecated argument \'bucket\' in ServiceBackend(). Specify \'remote_tmpdir\' as a keyword argument instead.')

        if remote_tmpdir is not None and bucket is not None:
            raise ValueError('Cannot specify both \'remote_tmpdir\' and \'bucket\' in ServiceBackend(). Specify \'remote_tmpdir\' as a keyword argument instead.')

        if bucket is None and remote_tmpdir is None:
            remote_tmpdir = user_config.get('batch', 'remote_tmpdir', fallback=None)

        if remote_tmpdir is None:
            if bucket is None:
                bucket = user_config.get('batch', 'bucket', fallback=None)
                warnings.warn('Using deprecated configuration setting \'batch/bucket\'. Run `hailctl config set batch/remote_tmpdir` '
                              'to set the default for \'remote_tmpdir\' instead.')
            if bucket is None:
                raise ValueError(
                    'The \'remote_tmpdir\' parameter of ServiceBackend must be set. '
                    'Run `hailctl config set batch/remote_tmpdir REMOTE_TMPDIR`')
            if 'gs://' in bucket:
                raise ValueError(
                    'The bucket parameter to ServiceBackend() should be a bucket name, not a path. '
                    'Use the remote_tmpdir parameter to specify a path.')
            remote_tmpdir = f'gs://{bucket}/batch'
        else:
            schemes = {'gs', 'hail-az'}
            found_scheme = any(remote_tmpdir.startswith(f'{scheme}://') for scheme in schemes)
            if not found_scheme:
                raise ValueError(
                    f'remote_tmpdir must be a storage uri path like gs://bucket/folder. Possible schemes include {schemes}')
        if remote_tmpdir[-1] != '/':
            remote_tmpdir += '/'
        self._remote_tmpdir = remote_tmpdir

        gcs_kwargs = {'project': google_project}
        self.__fs: AsyncFS = RouterAsyncFS(default_scheme='file', gcs_kwargs=gcs_kwargs)

    def remote_tmpdir(self) -> str:
        return self._remote_tmpdir

    def local_tmpdir(self) -> str:
        return '/io/'

    @property
    def _fs(self):
        return self.__fs

    def _close(self):
        async_to_blocking(self.batch_client.close())
        async_to_blocking(self._fs.close())

    def _run(self,
             batch: 'batch.Batch',
             dry_run: bool,
             verbose: bool,
             delete_scratch_on_exit: bool,
             wait: bool = True,
             open: bool = False,
             disable_progress_bar: bool = False,
             callback: Optional[str] = None,
             token: Optional[str] = None,
             **backend_kwargs) -> bc.Batch:  # pylint: disable-msg=too-many-statements
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
        return async_to_blocking(
            self._async_run(batch, dry_run, verbose, delete_scratch_on_exit, wait, open, disable_progress_bar, callback, token, **backend_kwargs))

    async def _async_run(self,
                         batch: 'batch.Batch',
                         dry_run: bool,
                         verbose: bool,
                         delete_scratch_on_exit: bool,
                         wait: bool = True,
                         open: bool = False,
                         disable_progress_bar: bool = False,
                         callback: Optional[str] = None,
                         token: Optional[str] = None,
                         **backend_kwargs):  # pylint: disable-msg=too-many-statements
        if backend_kwargs:
            raise ValueError(f'ServiceBackend does not support any of these keywords: {backend_kwargs}')

        build_dag_start = time.time()

        pyjobs = [j for j in batch._jobs if isinstance(j, _job.PythonJob)]
        for job in pyjobs:
            if job._image is None:
                version = sys.version_info
                if version.major != 3 or version.minor not in (6, 7, 8):
                    raise BatchException(
                        f"You must specify 'image' for Python jobs if you are using a Python version other than 3.6, 3.7, or 3.8 (you are using {version})")
                job._image = f'hailgenetics/python-dill:{version.major}.{version.minor}-slim'

        default_image = 'ubuntu:20.04'
        attributes = batch.attributes
        if batch.name is not None:
            attributes = {'name': batch.name, **attributes}
        job_to_client_job_mapping: Dict[_job.Job, aiobc.Job] = {}
        batch_builder = self.batch_client.create_batch(
            attributes=attributes,
            callback=callback,
            token=token,
            cancel_after_n_failures=batch._cancel_after_n_failures
        )

        with tqdm(total=len(batch._jobs), desc='upload code', disable=disable_progress_bar) as pbar:
            async def compile_job(job: _job.Job) -> Tuple[_job.Job, str, str, List[resource.Resource]]:
                # FIXME: data type for this triplet
                code, user_code, resources_to_download = await job.compile(dry_run=dry_run)
                pbar.update(1)
                return job, code, user_code, resources_to_download
            compiled_jobs = await bounded_gather(*[functools.partial(compile_job, j) for j in batch._jobs], parallelism=150)

        used_remote_tmpdir = any(len(j._need_to_copy_out) > 0 for j in batch._jobs)

        for job, code, user_code, resources_to_download in tqdm(compiled_jobs, desc='create job objects', disable=disable_progress_bar):
            inputs = [(r.remote_location(), r.local_location()) for r in resources_to_download]

            def output_src_dest_pairs(r: resource.Resource) -> List[Tuple[str, str]]:
                external_locations = batch._outputs[r]
                if job.resource_defined_remotely(r):
                    return [(r.remote_location(), dest) for dest in external_locations]
                return [
                    (r.local_location(), dest)
                    for dest in [r.remote_location(), *external_locations]
                ]

            outputs = [
                src_dest
                for r in job._need_to_copy_out
                for src_dest in output_src_dest_pairs(r)
            ]

            if job._image is None:
                if verbose:
                    print(f"Using image '{default_image}' since no image was specified.")

            if dry_run:
                print('''
================================================================================
# Job {job._job_id} {f": {job.name}" if job.name else ''}
''')
                print(code)
                print('''
================================================================================
''')
                continue

            parents = [job_to_client_job_mapping[j] for j in job._dependencies]

            attributes = job.attributes or dict()
            if job.name:
                attributes = {'name': job.name, **attributes}

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
                warnings.warn(f'Using an image, {image}, from Docker Hub. '
                              f'Jobs may fail due to Docker Hub rate limits.')

            env = job._env

            j = batch_builder.create_job(
                image=image,
                command=[job._shell if job._shell else DEFAULT_SHELL, '-c', code],
                parents=parents,
                attributes=attributes,
                resources=resources,
                input_files=inputs if len(inputs) > 0 else None,
                output_files=outputs if len(outputs) > 0 else None,
                always_run=job._always_run,
                timeout=job._timeout,
                cloudfuse=job._cloudfuse if len(job._cloudfuse) > 0 else None,
                env=env,
                requester_pays_project=batch.requester_pays_project,
                mount_tokens=True,
                user_code=user_code
            )

            job_to_client_job_mapping[job] = j

        if delete_scratch_on_exit and used_remote_tmpdir:
            j = batch_builder.create_job(
                image=HAIL_GENETICS_HAIL_IMAGE,
                command=['python3', '-m', 'hailtop.aiotools.delete', batch._remote_location],
                parents=batch_builder._jobs,
                attributes={'name': 'remove_tmpdir'},
                always_run=True)

        if verbose:
            print(f'Built DAG with {batch_builder.n_jobs} jobs in {round(time.time() - build_dag_start, 3)} seconds.')

        submit_batch_start = time.time()
        batch_handle = await batch_builder.submit(disable_progress_bar=disable_progress_bar)

        if verbose:
            print(f'Submitted batch {batch_handle.id} with {batch_builder.n_jobs} jobs in {round(time.time() - submit_batch_start, 3)} seconds.')

        deploy_config = get_deploy_config()
        url = deploy_config.external_url('batch', f'/batches/{batch_handle.id}')
        print(f'Submitted batch {batch_handle.id}, see {url}')

        if open:
            webbrowser.open(url)
        if wait:
            print(f'Waiting for batch {batch_handle.id}...')
            status = await batch_handle.wait()
            print(f'batch {batch_handle.id} complete: {status["state"]}')
        return bc.Batch.from_async_batch(batch_handle)
