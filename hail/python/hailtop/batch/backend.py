import abc
import os
import subprocess as sp
import uuid
import time
import copy
from shlex import quote as shq
import webbrowser
from hailtop.config import get_deploy_config, get_user_config
from hailtop.batch_client.client import BatchClient

from .resource import InputResourceFile, JobResourceFile, ResourceGroup


class Backend:
    """
    Abstract class for backends.
    """

    @abc.abstractmethod
    def _run(self, batch, dry_run, verbose, delete_scratch_on_exit, **backend_kwargs):
        """
        Execute a batch.

        Warning
        -------
        This method should not be called directly. Instead, use :meth:`.Batch.run`.
        """
        return


class LocalBackend(Backend):
    """
    Backend that executes batches on a local computer.

    Examples
    --------

    >>> local_backend = LocalBackend(tmp_dir='/tmp/user/')
    >>> b = Batch(backend=local_backend)

    Parameters
    ----------
    tmp_dir: :obj:`str`, optional
        Temporary directory to use.
    gsa_key_file: :obj:`str`, optional
        Mount a file with a gsa key to `/gsa-key/key.json`. Only used if a
        job specifies a docker image. This option will override the value set by
        the environment variable `HAIL_BATCH_GSA_KEY_FILE`.
    extra_docker_run_flags: :obj:`str`, optional
        Additional flags to pass to `docker run`. Only used if a job specifies
        a docker image. This option will override the value set by the environment
        variable `HAIL_BATCH_EXTRA_DOCKER_RUN_FLAGS`.
    """

    def __init__(self, tmp_dir='/tmp/', gsa_key_file=None, extra_docker_run_flags=None):
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

    def _run(self, batch, dry_run, verbose, delete_scratch_on_exit):  # pylint: disable=R0915
        """
        Execute a batch.

        Warning
        -------
        This method should not be called directly. Instead, use :meth:`.Batch.run`.

        Parameters
        ----------
        batch: :class:`.Batch`
            Batch to execute.
        dry_run: :obj:`bool`
            If `True`, don't execute code.
        verbose: :obj:`bool`
            If `True`, print debugging output.
        delete_scratch_on_exit: :obj:`bool`
            If `True`, delete temporary directories with intermediate files.
        """
        tmpdir = self._get_scratch_dir()

        script = ['#!/bin/bash',
                  'set -e' + 'x' if verbose else '',
                  '\n',
                  '# change cd to tmp directory',
                  f"cd {tmpdir}",
                  '\n']

        copied_input_resource_files = set()
        os.makedirs(tmpdir + '/inputs/', exist_ok=True)

        if batch.requester_pays_project:
            requester_pays_project = f'-u {batch.requester_pays_project}'
        else:
            requester_pays_project = ''

        def copy_input(job, r):
            if isinstance(r, InputResourceFile):
                if r not in copied_input_resource_files:
                    copied_input_resource_files.add(r)

                    if r._input_path.startswith('gs://'):
                        return [f'gsutil {requester_pays_project} cp {shq(r._input_path)} {shq(r._get_path(tmpdir))}']

                    absolute_input_path = os.path.realpath(r._input_path)

                    dest = r._get_path(tmpdir)
                    dir = os.path.dirname(dest)
                    os.makedirs(dir, exist_ok=True)

                    if job._image is not None:  # pylint: disable-msg=W0640
                        return [f'cp {shq(absolute_input_path)} {shq(dest)}']

                    return [f'ln -sf {shq(absolute_input_path)} {shq(dest)}']

                return []

            assert isinstance(r, JobResourceFile)
            return []

        def copy_external_output(r):
            def _cp(dest):
                if not dest.startswith('gs://'):
                    dest = os.path.abspath(dest)
                    directory = os.path.dirname(dest)
                    os.makedirs(directory, exist_ok=True)
                    return 'cp'
                return f'gsutil {requester_pays_project} cp'

            if isinstance(r, InputResourceFile):
                return [f'{_cp(dest)} {shq(r._input_path)} {shq(dest)}'
                        for dest in r._output_paths]

            assert isinstance(r, JobResourceFile)
            return [f'{_cp(dest)} {r._get_path(tmpdir)} {shq(dest)}'
                    for dest in r._output_paths]

        def symlink_input_resource_group(r):
            symlinks = []
            if isinstance(r, ResourceGroup) and r._source is None:
                for name, irf in r._resources.items():
                    src = irf._get_path(tmpdir)
                    dest = f'{r._get_path(tmpdir)}.{name}'
                    symlinks.append(f'ln -sf {shq(src)} {shq(dest)}')
            return symlinks

        write_inputs = [x for r in batch._input_resources for x in copy_external_output(r)]
        if write_inputs:
            script += ["# Write input resources to output destinations"]
            script += write_inputs
            script += ['\n']

        for job in batch._jobs:
            os.makedirs(f'{tmpdir}/{job._job_id}/', exist_ok=True)

            script.append(f"# {job._job_id}: {job.name if job.name else ''}")

            script += [x for r in job._inputs for x in copy_input(job, r)]
            script += [x for r in job._mentioned for x in symlink_input_resource_group(r)]

            resource_defs = [r._declare(tmpdir) for r in job._mentioned]
            env = [f'export {k}={v}' for k, v in job._env.items()]

            if job._image:
                defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
                env = '; '.join(env) + '; ' if env else ''
                cmd = " && ".join(f'{{\n{x}\n}}' for x in job._command)
                memory = f'-m {job._memory}' if job._memory else ''
                cpu = f'--cpus={job._cpu}' if job._cpu else ''

                script += [f"docker run "
                           f"{self._extra_docker_run_flags} "
                           f"-v {tmpdir}:{tmpdir} "
                           f"-w {tmpdir} "
                           f"{memory} "
                           f"{cpu} "
                           f"{job._image} /bin/bash "
                           f"-c {shq(env + defs + cmd)}",
                           '\n']
            else:
                script += env
                script += resource_defs
                script += job._command

            script += [x for r in job._external_outputs for x in copy_external_output(r)]
            script += ['\n']

        script = "\n".join(script)
        if dry_run:
            print(script)
        else:
            try:
                sp.check_call(script, shell=True)
            except sp.CalledProcessError as e:
                print(e)
                print(e.output)
                raise
            finally:
                if delete_scratch_on_exit:
                    sp.run(f'rm -rf {tmpdir}', shell=True)

        print('Batch completed successfully!')

    def _get_scratch_dir(self):
        def _get_random_name():
            dir = f'{self._tmp_dir}/batch/{uuid.uuid4().hex[:6]}'
            if os.path.isdir(dir):
                return _get_random_name()
            os.makedirs(dir, exist_ok=True)
            return dir

        return _get_random_name()


class ServiceBackend(Backend):
    """Backend that executes batches on Hail's Batch Service on Google Cloud.

    Examples
    --------

    >>> service_backend = ServiceBackend('my-billing-account', 'my-bucket') # doctest: +SKIP
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

    Parameters
    ----------
    billing_project: :obj:`str`, optional
        Name of billing project to use.
    bucket: :obj:`str`, optional
        Name of bucket to use.  Should not include the ``gs://``
        prefix.

    """

    def __init__(self, billing_project: str = None, bucket: str = None):
        if billing_project is None:
            billing_project = get_user_config().get('batch', 'billing_project', fallback=None)
        if billing_project is None:
            raise ValueError(
                f'the billing_project parameter of ServiceBackend must be set '
                f'or run `hailctl config set batch/billing_project '
                f'MY_BILLING_PROJECT`')
        self._batch_client = BatchClient(billing_project)

        if bucket is None:
            bucket = get_user_config().get('batch', 'bucket', fallback=None)
        if bucket is None:
            raise ValueError(
                f'the bucket parameter of ServiceBackend must be set '
                f'or run `hailctl config set batch/bucket '
                f'MY_BUCKET`')
        self._bucket_name = bucket

    def close(self):
        """
        Close the connection with the Batch Service.

        Notes
        -----
        This method should be called after executing your batches at the
        end of your script.
        """
        self._batch_client.close()

    def _run(self,
             batch,
             dry_run,
             verbose,
             delete_scratch_on_exit,
             wait=True,
             open=False,
             disable_progress_bar=False,
             callback=None):  # pylint: disable-msg=too-many-statements
        """Execute a batch.

        Warning
        -------
        This method should not be called directly. Instead, use :meth:`.Batch.run`
        and pass :class:`.ServiceBackend` specific arguments as key-word arguments.

        Parameters
        ----------
        batch: :class:`.Batch`
            Batch to execute.
        dry_run: :obj:`bool`
            If `True`, don't execute code.
        verbose: :obj:`bool`
            If `True`, print debugging output.
        delete_scratch_on_exit: :obj:`bool`
            If `True`, delete temporary directories with intermediate files.
        wait: :obj:`bool`, optional
            If `True`, wait for the batch to finish executing before returning.
        open: :obj:`bool`, optional
            If `True`, open the UI page for the batch.
        disable_progress_bar: :obj:`bool`, optional
            If `True`, disable the progress bar.
        callback: :obj:`str`, optional
            If not `None`, a URL that will receive at most one POST request
            after the entire batch completes.
        """
        build_dag_start = time.time()

        token = uuid.uuid4().hex[:6]
        remote_tmpdir = f'gs://{self._bucket_name}/batch/{token}'
        local_tmpdir = f'/io/batch/{token}'

        default_image = 'ubuntu:latest'

        attributes = copy.deepcopy(batch.attributes)
        if batch.name is not None:
            attributes['name'] = batch.name

        bc_batch = self._batch_client.create_batch(attributes=attributes, callback=callback)

        n_jobs_submitted = 0
        used_remote_tmpdir = False

        job_to_client_job_mapping = {}
        jobs_to_command = {}
        commands = []

        bash_flags = 'set -e' + ('x' if verbose else '')

        activate_service_account = 'gcloud -q auth activate-service-account ' \
                                   '--key-file=/gsa-key/key.json'

        def copy_input(r):
            if isinstance(r, InputResourceFile):
                return [(r._input_path, r._get_path(local_tmpdir))]
            assert isinstance(r, JobResourceFile)
            return [(r._get_path(remote_tmpdir), r._get_path(local_tmpdir))]

        def copy_internal_output(r):
            assert isinstance(r, JobResourceFile)
            return [(r._get_path(local_tmpdir), r._get_path(remote_tmpdir))]

        def copy_external_output(r):
            if isinstance(r, InputResourceFile):
                return [(r._input_path, dest) for dest in r._output_paths]
            assert isinstance(r, JobResourceFile)
            return [(r._get_path(local_tmpdir), dest) for dest in r._output_paths]

        def symlink_input_resource_group(r):
            symlinks = []
            if isinstance(r, ResourceGroup) and r._source is None:
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
                j = bc_batch.create_job(image='google/cloud-sdk:237.0.0-alpine',
                                        command=['/bin/bash', '-c', write_cmd],
                                        attributes={'name': 'write_external_inputs'})
                jobs_to_command[j] = write_cmd
                n_jobs_submitted += 1

        for job in batch._jobs:
            inputs = [x for r in job._inputs for x in copy_input(r)]

            outputs = [x for r in job._internal_outputs for x in copy_internal_output(r)]
            if outputs:
                used_remote_tmpdir = True
            outputs += [x for r in job._external_outputs for x in copy_external_output(r)]

            symlinks = [x for r in job._mentioned for x in symlink_input_resource_group(r)]

            env_vars = {
                **job._env,
                **{r._uid: r._get_path(local_tmpdir) for r in job._mentioned}}

            if job._image is None:
                if verbose:
                    print(f"Using image '{default_image}' since no image was specified.")

            make_local_tmpdir = f'mkdir -p {local_tmpdir}/{job._job_id}'
            job_command = [cmd.strip() for cmd in job._command]

            prepared_job_command = (f'{{\n{x}\n}}' for x in job_command)
            cmd = f'''
{bash_flags}
{make_local_tmpdir}
{"; ".join(symlinks)}
{" && ".join(prepared_job_command)}
'''

            if dry_run:
                commands.append(cmd)
                continue

            parents = [job_to_client_job_mapping[j] for j in job._dependencies]

            attributes = copy.deepcopy(job.attributes)
            if job.name:
                attributes['name'] = job.name

            resources = {}
            if job._cpu:
                resources['cpu'] = job._cpu
            if job._memory:
                resources['memory'] = job._memory

            j = bc_batch.create_job(image=job._image if job._image else default_image,
                                    command=['/bin/bash', '-c', cmd],
                                    parents=parents,
                                    attributes=attributes,
                                    resources=resources,
                                    input_files=inputs if len(inputs) > 0 else None,
                                    output_files=outputs if len(outputs) > 0 else None,
                                    pvc_size=job._storage,
                                    always_run=job._always_run,
                                    timeout=job._timeout,
                                    gcsfuse=job._gcsfuse if len(job._gcsfuse) > 0 else None,
                                    env=env_vars,
                                    requester_pays_project=batch.requester_pays_project)

            n_jobs_submitted += 1

            job_to_client_job_mapping[job] = j
            jobs_to_command[j] = cmd

        if dry_run:
            print("\n\n".join(commands))
            return None

        if delete_scratch_on_exit and used_remote_tmpdir:
            parents = list(jobs_to_command.keys())
            rm_cmd = f'gsutil -m rm -r {remote_tmpdir}'
            cmd = f'''
{bash_flags}
{activate_service_account}
{rm_cmd}
'''
            j = bc_batch.create_job(
                image='google/cloud-sdk:237.0.0-alpine',
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
