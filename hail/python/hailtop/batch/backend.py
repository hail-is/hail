from typing import Optional, Dict, Any, TypeVar, Generic
import sys
import abc
import os
import subprocess as sp
import uuid
import time
import functools
import copy
from shlex import quote as shq
import webbrowser
import warnings
from concurrent.futures import ThreadPoolExecutor

from hailtop.config import get_deploy_config, get_user_config
from hailtop.utils import is_google_registry_domain, parse_docker_image_reference, async_to_blocking, bounded_gather, tqdm
from hailtop.batch.hail_genetics_images import HAIL_GENETICS_IMAGES
from hailtop.batch_client.parse import parse_cpu_in_mcpu
import hailtop.batch_client.client as bc
from hailtop.batch_client.client import BatchClient
from hailtop.aiotools import RouterAsyncFS, LocalAsyncFS, AsyncFS
from hailtop.aiogoogle import GoogleStorageAsyncFS

from . import resource, batch, job as _job  # pylint: disable=unused-import
from .exceptions import BatchException
from .globals import DEFAULT_SHELL


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
        self.__fs: AsyncFS = LocalAsyncFS(ThreadPoolExecutor())

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

        if backend_kwargs:
            raise ValueError(f'LocalBackend does not support any of these keywords: {backend_kwargs}')

        tmpdir = self._get_scratch_dir()

        def new_code_block():
            return ['set -e' + ('x' if verbose else ''),
                    '\n',
                    '# change cd to tmp directory',
                    f"cd {tmpdir}",
                    '\n']

        def run_code(code):
            code = '\n'.join(code)
            if dry_run:
                print(code)
            else:
                try:
                    sp.check_call(code, shell=True)
                except sp.CalledProcessError as e:
                    print(e)
                    print(e.output)
                    raise

        copied_input_resource_files = set()
        os.makedirs(tmpdir + '/inputs/', exist_ok=True)

        if batch.requester_pays_project:
            requester_pays_project = f'-u {batch.requester_pays_project}'
        else:
            requester_pays_project = ''

        def copy_input(job, r):
            if isinstance(r, resource.InputResourceFile):
                if r not in copied_input_resource_files:
                    copied_input_resource_files.add(r)

                    if r._input_path.startswith('gs://'):
                        return [f'gsutil {requester_pays_project} cp -r {shq(r._input_path)} {shq(r._get_path(tmpdir))}']

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

        def copy_external_output(r):
            def _cp(dest):
                if not dest.startswith('gs://'):
                    dest = os.path.expanduser(dest)
                    dest = os.path.abspath(dest)
                    directory = os.path.dirname(dest)
                    os.makedirs(directory, exist_ok=True)
                    return 'cp'
                return f'gsutil {requester_pays_project} cp -r'

            if isinstance(r, resource.InputResourceFile):
                return [f'{_cp(dest)} {shq(r._input_path)} {shq(dest)}'
                        for dest in r._output_paths]

            assert isinstance(r, (resource.JobResourceFile, resource.PythonResult))
            return [f'{_cp(dest)} {r._get_path(tmpdir)} {shq(dest)}'
                    for dest in r._output_paths]

        def symlink_input_resource_group(r):
            symlinks = []
            if isinstance(r, resource.ResourceGroup) and r._source is None:
                for name, irf in r._resources.items():
                    src = irf._get_path(tmpdir)
                    dest = f'{r._get_path(tmpdir)}.{name}'
                    symlinks.append(f'ln -sf {shq(src)} {shq(dest)}')
            return symlinks

        try:
            write_inputs = [x for r in batch._input_resources for x in copy_external_output(r)]
            if write_inputs:
                code = new_code_block()
                code += ["# Write input resources to output destinations"]
                code += write_inputs
                code += ['\n']
                run_code(code)

            for job in batch._jobs:
                async_to_blocking(job._compile(tmpdir, tmpdir))

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

                    code.append(f"docker run "
                                "--entrypoint=''"
                                f"{self._extra_docker_run_flags} "
                                f"-v {tmpdir}:{tmpdir} "
                                f"-w {tmpdir} "
                                f"{memory} "
                                f"{cpu} "
                                f"{job._image} "
                                f"{job_shell} -c {quoted_job_script}")
                else:
                    code.append(f"{job_shell} -c {quoted_job_script}")

                code += [x for r in job._external_outputs for x in copy_external_output(r)]
                code += ['\n']

                run_code(code)
        finally:
            if delete_scratch_on_exit:
                sp.run(f'rm -rf {tmpdir}', shell=True, check=False)

        print('Batch completed successfully!')

    def _get_scratch_dir(self):
        def _get_random_name():
            dir = f'{self._tmp_dir}/batch/{uuid.uuid4().hex[:6]}'
            if os.path.isdir(dir):
                return _get_random_name()
            os.makedirs(dir, exist_ok=True)
            return dir

        return _get_random_name()

    def _close(self):
        async_to_blocking(self._fs.close())


class ServiceBackend(Backend[bc.Batch]):
    """Backend that executes batches on Hail's Batch Service on Google Cloud.

    Examples
    --------

    >>> service_backend = ServiceBackend('my-billing-account', bucket='my-bucket') # doctest: +SKIP
    >>> b = Batch(backend=service_backend) # doctest: +SKIP
    >>> b.run() # doctest: +SKIP
    >>> service_backend.close() # doctest: +SKIP

    If the Hail configuration parameters batch/billing_project and
    batch/bucket were previously set with ``hailctl config set``, then
    one may elide the `billing_project` and `bucket` parameters.

    >>> service_backend = ServiceBackend()
    >>> b = Batch(backend=service_backend)
    >>> b.run() # doctest: +SKIP
    >>> service_backend.close()

    Instead of a bucket, a full path may be specified for the remote temporary directory:

    >>> service_backend = ServiceBackend('my-billing-account',
    ...                                  remote_tmpdir='gs://my-bucket/temporary-files/')
    >>> b = Batch(backend=service_backend)
    >>> b.run() # doctest: +SKIP
    >>> service_backend.close()

    Parameters
    ----------
    billing_project:
        Name of billing project to use.
    bucket:
        Name of bucket to use. Should not include the ``gs://`` prefix. Cannot be used with
        remote_tmpdir. Temporary data will be stored in the "/batch" folder of this
        bucket. Using this parameter as a positional argument is deprecated.
    remote_tmpdir:
        Temporary data will be stored in this google cloud storage folder. Cannot be used with
        bucket.
    google_project:
        If specified, the project to use when authenticating with Google
        Storage. Google Storage is used to transfer serialized values between
        this computer and the cloud machines that execute Python jobs.

    """

    def __init__(self,
                 *args,
                 billing_project: Optional[str] = None,
                 bucket: Optional[str] = None,
                 remote_tmpdir: Optional[str] = None,
                 google_project: Optional[str] = None
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

        if remote_tmpdir is not None and bucket is not None:
            raise ValueError('Cannot specify both remote_tmpdir and bucket in ServiceBackend()')

        if billing_project is None:
            billing_project = get_user_config().get('batch', 'billing_project', fallback=None)
        if billing_project is None:
            raise ValueError(
                'the billing_project parameter of ServiceBackend must be set '
                'or run `hailctl config set batch/billing_project '
                'MY_BILLING_PROJECT`')
        self._batch_client = BatchClient(billing_project)
        self.__fs: AsyncFS = RouterAsyncFS('file', [LocalAsyncFS(ThreadPoolExecutor()),
                                                    GoogleStorageAsyncFS(project=google_project)])

        if remote_tmpdir is None:
            if bucket is None:
                bucket = get_user_config().get('batch', 'bucket', fallback=None)
            if bucket is None:
                raise ValueError(
                    'either the bucket or remote_tmpdir parameter of ServiceBackend '
                    'must be set or run `hailctl config set batch/bucket MY_BUCKET`')
            if 'gs://' in bucket:
                raise ValueError(
                    'The bucket parameter to ServiceBackend() should be a bucket name, not a path. '
                    'Use the remote_tmpdir parameter to specify a path.')
            remote_tmpdir = f'gs://{bucket}/batch'
        else:
            if not remote_tmpdir.startswith('gs://'):
                raise ValueError(
                    'remote_tmpdir must be a google storage path like gs://bucket/folder')
        if remote_tmpdir[-1] != '/':
            remote_tmpdir += '/'
        self.remote_tmpdir = remote_tmpdir

    @property
    def _fs(self):
        return self.__fs

    def _close(self):
        self._batch_client.close()
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

        uid = uuid.uuid4().hex[:6]
        batch_remote_tmpdir = f'{self.remote_tmpdir}{uid}'
        local_tmpdir = f'/io/batch/{uid}'

        default_image = 'ubuntu:18.04'

        attributes = copy.deepcopy(batch.attributes)
        if batch.name is not None:
            attributes['name'] = batch.name

        bc_batch = self._batch_client.create_batch(attributes=attributes, callback=callback,
                                                   token=token, cancel_after_n_failures=batch._cancel_after_n_failures)

        n_jobs_submitted = 0
        used_remote_tmpdir = False

        job_to_client_job_mapping: Dict[_job.Job, bc.Job] = {}
        jobs_to_command = {}
        commands = []

        bash_flags = 'set -e' + ('x' if verbose else '')

        activate_service_account = 'gcloud -q auth activate-service-account ' \
                                   '--key-file=/gsa-key/key.json'

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
            def _cp(src, dst):
                return f'gsutil -m cp -R {shq(src)} {shq(dst)}'

            write_cmd = f'''
{bash_flags}
{activate_service_account}
{' && '.join([_cp(*files) for files in write_external_inputs])}
'''

            if dry_run:
                commands.append(write_cmd)
            else:
                j = bc_batch.create_job(image='gcr.io/google.com/cloudsdktool/cloud-sdk:310.0.0-alpine',
                                        command=['/bin/bash', '-c', write_cmd],
                                        attributes={'name': 'write_external_inputs'})
                jobs_to_command[j] = write_cmd
                n_jobs_submitted += 1

        pyjobs = [j for j in batch._jobs if isinstance(j, _job.PythonJob)]
        for job in pyjobs:
            if job._image is None:
                version = sys.version_info
                if version.major != 3 or version.minor not in (6, 7, 8):
                    raise BatchException(
                        f"You must specify 'image' for Python jobs if you are using a Python version other than 3.6, 3.7, or 3.8 (you are using {version})")
                job._image = f'hailgenetics/python-dill:{version.major}.{version.minor}-slim'

        with tqdm(total=len(batch._jobs), desc='upload code', disable=disable_progress_bar) as pbar:
            async def compile_job(job):
                used_remote_tmpdir = await job._compile(local_tmpdir, batch_remote_tmpdir, dry_run=dry_run)
                pbar.update(1)
                return used_remote_tmpdir
            used_remote_tmpdir_results = await bounded_gather(*[functools.partial(compile_job, j) for j in batch._jobs], parallelism=150)
            used_remote_tmpdir |= any(used_remote_tmpdir_results)

        for job in tqdm(batch._jobs, desc='create job objects', disable=disable_progress_bar):
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
            cmd = f'''
{bash_flags}
{make_local_tmpdir}
{"; ".join(symlinks)}
{" && ".join(prepared_job_command)}
'''

            user_code = '\n\n'.join(job._user_code) if job._user_code else None

            if dry_run:
                formatted_command = f'''
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
'''
                commands.append(formatted_command)
                continue

            parents = [job_to_client_job_mapping[j] for j in job._dependencies]

            attributes = copy.deepcopy(job.attributes) if job.attributes else dict()
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
            if not is_google_registry_domain(image_ref.domain) and image_ref.name() not in HAIL_GENETICS_IMAGES:
                warnings.warn(f'Using an image {image} not in GCR. '
                              f'Jobs may fail due to Docker Hub rate limits.')

            env = {**job._env, 'BATCH_TMPDIR': local_tmpdir}

            j = bc_batch.create_job(image=image,
                                    command=[job._shell if job._shell else DEFAULT_SHELL, '-c', cmd],
                                    parents=parents,
                                    attributes=attributes,
                                    resources=resources,
                                    input_files=inputs if len(inputs) > 0 else None,
                                    output_files=outputs if len(outputs) > 0 else None,
                                    always_run=job._always_run,
                                    timeout=job._timeout,
                                    gcsfuse=job._gcsfuse if len(job._gcsfuse) > 0 else None,
                                    env=env,
                                    requester_pays_project=batch.requester_pays_project,
                                    mount_tokens=True,
                                    user_code=user_code)

            n_jobs_submitted += 1

            job_to_client_job_mapping[job] = j
            jobs_to_command[j] = cmd

        if dry_run:
            print("\n\n".join(commands))
            return None

        if delete_scratch_on_exit and used_remote_tmpdir:
            parents = list(jobs_to_command.keys())
            rm_cmd = f'gsutil -m rm -r {batch_remote_tmpdir}'
            cmd = f'''
{bash_flags}
{activate_service_account}
{rm_cmd}
'''
            j = bc_batch.create_job(
                image='gcr.io/google.com/cloudsdktool/cloud-sdk:310.0.0-alpine',
                command=['/bin/bash', '-c', cmd],
                parents=parents,
                attributes={'name': 'remove_tmpdir'},
                always_run=True)
            jobs_to_command[j] = cmd
            n_jobs_submitted += 1

        if verbose:
            print(f'Built DAG with {n_jobs_submitted} jobs in {round(time.time() - build_dag_start, 3)} seconds.')

        submit_batch_start = time.time()
        bc_batch = bc_batch.submit(disable_progress_bar=disable_progress_bar)

        jobs_to_command = {j.id: cmd for j, cmd in jobs_to_command.items()}

        if verbose:
            print(f'Submitted batch {bc_batch.id} with {n_jobs_submitted} jobs in {round(time.time() - submit_batch_start, 3)} seconds:')
            for jid, cmd in jobs_to_command.items():
                print(f'{jid}: {cmd}')
            print('')

        deploy_config = get_deploy_config()
        url = deploy_config.url('batch', f'/batches/{bc_batch.id}')
        print(f'Submitted batch {bc_batch.id}, see {url}')

        if open:
            webbrowser.open(url)
        if wait:
            print(f'Waiting for batch {bc_batch.id}...')
            status = bc_batch.wait()
            print(f'batch {bc_batch.id} complete: {status["state"]}')
        return bc_batch
