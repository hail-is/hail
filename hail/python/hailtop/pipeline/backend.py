import abc
import os
import subprocess as sp
import uuid
import secrets
import json
from shlex import quote as shq
from hailtop.batch_client.client import BatchClient, Job
import aiohttp

from .resource import InputResourceFile, TaskResourceFile
from .utils import PipelineException

class Backend:
    @abc.abstractmethod
    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        return


class LocalBackend(Backend):
    """
    Backend that executes pipelines on a local computer.

    Examples
    --------

    >>> local_backend = LocalBackend(tmp_dir='/tmp/user/')
    >>> p = Pipeline(backend=local_backend)

    Parameters
    ----------
    tmp_dir: :obj:`str`, optional
        Temporary directory to use.
    gsa_key_file :obj:`str`, optional
        Mount a file with a gsa key to `/gsa-key/privateKeyData`. Only used if a
        task specifies a docker image. This option will override the value set by
        the environment variable `HAIL_PIPELINE_GSA_KEY_FILE`.
    extra_docker_run_flags :obj:`str`, optional
        Additional flags to pass to `docker run`. Only used if a task specifies
        a docker image. This option will override the value set by the environment
        variable `HAIL_PIPELINE_EXTRA_DOCKER_RUN_FLAGS`.
    """

    def __init__(self, tmp_dir='/tmp/', gsa_key_file=None, extra_docker_run_flags=None):
        self._tmp_dir = tmp_dir

        flags = ''

        if extra_docker_run_flags is not None:
            flags += extra_docker_run_flags
        elif os.environ.get('HAIL_PIPELINE_EXTRA_DOCKER_RUN_FLAGS') is not None:
            flags += os.environ['HAIL_PIPELINE_EXTRA_DOCKER_RUN_FLAGS']

        if gsa_key_file is None:
            gsa_key_file = os.environ.get('HAIL_PIPELINE_GSA_KEY_FILE')
        if gsa_key_file is not None:
            flags += f' -v {gsa_key_file}:/gsa-key/privateKeyData'

        self._extra_docker_run_flags = flags

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):  # pylint: disable=R0915
        tmpdir = self._get_scratch_dir()

        script = ['#!/bin/bash',
                  'set -e' + 'x' if verbose else '',
                  '\n',
                  '# change cd to tmp directory',
                  f"cd {tmpdir}",
                  '\n']

        copied_input_resource_files = set()
        os.makedirs(tmpdir + 'inputs/', exist_ok=True)

        def copy_input(task, r):
            if isinstance(r, InputResourceFile):
                if r not in copied_input_resource_files:
                    copied_input_resource_files.add(r)

                    if r._input_path.startswith('gs://'):
                        return [f'gsutil cp {r._input_path} {r._get_path(tmpdir)}']
                    else:
                        absolute_input_path = shq(os.path.realpath(r._input_path))
                        if task._image is not None:  # pylint: disable-msg=W0640
                            return [f'cp {absolute_input_path} {r._get_path(tmpdir)}']
                        else:
                            return [f'ln -sf {absolute_input_path} {r._get_path(tmpdir)}']
                else:
                    return []
            else:
                assert isinstance(r, TaskResourceFile)
                return []

        def copy_external_output(r):
            def cp(dest):
                if not dest.startswith('gs://'):
                    dest = os.path.abspath(dest)
                    directory = os.path.dirname(dest)
                    os.makedirs(directory, exist_ok=True)
                    return 'cp'
                else:
                    return 'gsutil cp'

            if isinstance(r, InputResourceFile):
                return [f'{cp(dest)} {shq(r._input_path)} {shq(dest)}'
                        for dest in r._output_paths]
            else:
                assert isinstance(r, TaskResourceFile)
                return [f'{cp(dest)} {r._get_path(tmpdir)} {shq(dest)}'
                        for dest in r._output_paths]

        write_inputs = [x for r in pipeline._input_resources for x in copy_external_output(r)]
        if write_inputs:
            script += ["# Write input resources to output destinations"]
            script += write_inputs
            script += ['\n']

        for task in pipeline._tasks:
            os.makedirs(tmpdir + task._uid + '/', exist_ok=True)

            script.append(f"# {task._uid} {task.name if task.name else ''}")

            script += [x for r in task._inputs for x in copy_input(task, r)]

            resource_defs = [r._declare(tmpdir) for r in task._mentioned]

            if task._image:
                defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
                cmd = " && ".join(task._command)
                memory = f'-m {task._memory}' if task._memory else ''
                cpu = f'--cpus={task._cpu}' if task._cpu else ''

                script += [f"docker run "
                           f"{self._extra_docker_run_flags} "
                           f"-v {tmpdir}:{tmpdir} "
                           f"-w {tmpdir} "
                           f"{memory} "
                           f"{cpu} "
                           f"{task._image} /bin/bash "
                           f"-c {shq(defs + cmd)}",
                           '\n']
            else:
                script += resource_defs
                script += task._command

            script += [x for r in task._external_outputs for x in copy_external_output(r)]
            script += ['\n']

        script = "\n".join(script)
        if dry_run:
            print(script)
        else:
            try:
                sp.check_output(script, shell=True)
            except sp.CalledProcessError as e:
                print(e)
                print(e.output)
                raise
            finally:
                if delete_scratch_on_exit:
                    sp.run(f'rm -rf {tmpdir}', shell=True)

        print('Pipeline completed successfully!')

    def _get_scratch_dir(self):
        def _get_random_name():
            directory = self._tmp_dir + '/pipeline-{}/'.format(uuid.uuid4().hex[:12])

            if os.path.isdir(directory):
                return _get_random_name()
            else:
                os.makedirs(directory, exist_ok=True)
                return directory

        return _get_random_name()


class BatchBackend(Backend):
    """
    Backend that executes pipelines on a Kubernetes cluster using `batch`.

    Examples
    --------

    >>> batch_backend = BatchBackend(tmp_dir='http://localhost:5000')
    >>> p = Pipeline(backend=batch_backend)

    Parameters
    ----------
    url: :obj:`str`
        URL to batch server.
    """

    def __init__(self, url):
        session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60))
        self._batch_client = BatchClient(session, url)

    def close(self):
        self._batch_client.close()

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):  # pylint: disable-msg=R0915
        bucket = self._batch_client.bucket
        subdir_name = 'pipeline-{}'.format(uuid.uuid4().hex[:12])

        remote_tmpdir = f'gs://{bucket}/pipeline/{subdir_name}'
        local_tmpdir = f'/io/pipeline/{subdir_name}'

        default_image = 'ubuntu'

        attributes = pipeline.attributes
        if pipeline.name is not None:
            attributes['name'] = pipeline.name

        batch = self._batch_client.create_batch(attributes=attributes)

        n_jobs_submitted = 0
        used_remote_tmpdir = False

        task_to_job_mapping = {}
        jobs_to_command = {}
        commands = []

        bash_flags = 'set -e' + ('x' if verbose else '') + '; '

        activate_service_account = 'gcloud -q auth activate-service-account ' \
                                   '--key-file=/gsa-key/privateKeyData'

        def copy_input(r):
            if isinstance(r, InputResourceFile):
                return [(r._input_path, r._get_path(local_tmpdir))]
            else:
                assert isinstance(r, TaskResourceFile)
                return [(r._get_path(remote_tmpdir), r._get_path(local_tmpdir))]

        def copy_internal_output(r):
            assert isinstance(r, TaskResourceFile)
            return [(r._get_path(local_tmpdir), r._get_path(remote_tmpdir))]

        def copy_external_output(r):
            if isinstance(r, InputResourceFile):
                return [(r._input_path, dest) for dest in r._output_paths]
            else:
                assert isinstance(r, TaskResourceFile)
                return [(r._get_path(local_tmpdir), dest) for dest in r._output_paths]

        write_external_inputs = [x for r in pipeline._input_resources for x in copy_external_output(r)]
        if write_external_inputs:
            def _cp(src, dst):
                return f'gsutil -m cp -R {src} {dst}'

            write_cmd = bash_flags + activate_service_account + ' && ' + \
                        ' && '.join([_cp(*files) for files in write_external_inputs])

            if dry_run:
                commands.append(write_cmd)
            else:
                j = batch.create_job(image='google/cloud-sdk:237.0.0-alpine',
                                     command=['/bin/bash', '-c', write_cmd],
                                     attributes={'name': 'write_external_inputs'})
                jobs_to_command[j] = write_cmd
                n_jobs_submitted += 1

        for task in pipeline._tasks:
            inputs = [x for r in task._inputs for x in copy_input(r)]

            outputs = [x for r in task._internal_outputs for x in copy_internal_output(r)]
            if outputs:
                used_remote_tmpdir = True
            outputs += [x for r in task._external_outputs for x in copy_external_output(r)]

            resource_defs = [r._declare(directory=local_tmpdir) for r in task._mentioned]

            if task._image is None:
                if verbose:
                    print(f"Using image '{default_image}' since no image was specified.")

            make_local_tmpdir = f'mkdir -p {local_tmpdir}/{task._uid}/; '
            defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
            task_command = [cmd.strip() for cmd in task._command]

            cmd = bash_flags + make_local_tmpdir + defs + " && ".join(task_command)
            if dry_run:
                commands.append(cmd)
                continue

            parents = [task_to_job_mapping[t] for t in task._dependencies]

            attributes = {'task_uid': task._uid}
            if task.name:
                attributes['name'] = task.name
            attributes.update(task.attributes)

            resources = {'requests': {}}
            if task._cpu:
                resources['requests']['cpu'] = task._cpu
            if task._memory:
                resources['requests']['memory'] = task._memory

            j = batch.create_job(image=task._image if task._image else default_image,
                                 command=['/bin/bash', '-c', cmd],
                                 parents=parents,
                                 attributes=attributes,
                                 resources=resources,
                                 input_files=inputs if len(inputs) > 0 else None,
                                 output_files=outputs if len(outputs) > 0 else None,
                                 pvc_size=task._storage)
            n_jobs_submitted += 1

            task_to_job_mapping[task] = j
            jobs_to_command[j] = cmd

        if dry_run:
            print("\n\n".join(commands))
            return

        if delete_scratch_on_exit and used_remote_tmpdir:
            parents = list(jobs_to_command.keys())
            rm_cmd = f'gsutil rm -r {remote_tmpdir}'
            cmd = bash_flags + f'{activate_service_account} && {rm_cmd}'
            j = batch.create_job(
                image='google/cloud-sdk:237.0.0-alpine',
                command=['/bin/bash', '-c', cmd],
                parents=parents,
                attributes={'name': 'remove_tmpdir'},
                always_run=True)
            jobs_to_command[j] = cmd
            n_jobs_submitted += 1

        batch = batch.submit()

        jobs_to_command = {j.id: cmd for j, cmd in jobs_to_command.items()}

        if verbose:
            print(f'Submitted batch {batch.id} with {n_jobs_submitted} jobs:')
            for jid, cmd in jobs_to_command.items():
                print(f'{jid}: {cmd}')

        status = batch.wait()

        if status['state'] == 'success':
            print('Pipeline completed successfully!')
            return

        failed_jobs = [((j['batch_id'], j['job_id']), j['exit_code']) for j in status['jobs'] if 'exit_code' in j and any([ec != 0 for _, ec in j['exit_code'].items()])]

        fail_msg = ''
        for jid, ec in failed_jobs:
            ec = Job.exit_code(ec)
            job = self._batch_client.get_job(*jid)
            log = job.log()
            name = job.status()['attributes'].get('name', None)
            fail_msg += (
                f"Job {jid} failed with exit code {ec}:\n"
                f"  Task name:\t{name}\n"
                f"  Command:\t{jobs_to_command[jid]}\n"
                f"  Log:\t{log}\n")

        raise PipelineException(fail_msg)


class HackRunner:
    def __init__(self, pipeline, verbose):
        self.pipeline = pipeline
        self.verbose = verbose

        self.n_complete = 0

        self.ready = []
        self.task_n_deps = {}
        for t in pipeline._tasks:
            n = len(t._dependencies)
            self.task_n_deps[t] = n
            if n == 0:
                self.ready.append(t)

        self.task_children = {t: set() for t in pipeline._tasks}
        for t in pipeline._tasks:
            for d in t._dependencies:
                self.task_children[d].add(t)

        self.task_token = {}

    def gs_input_path(self, r):
        if isinstance(r, InputResourceFile):
            return r._input_path
        else:
            assert isinstance(r, TaskResourceFile)
            assert r._source
            token = self.task_token[r._source]
            return r._get_path(f'gs://hail-cseed/cs-hack/tmp/{token}')

    def gs_output_paths(self, r, token):
        assert isinstance(r, TaskResourceFile)
        output_paths = [r._get_path(f'gs://hail-cseed/cs-hack/tmp/{token}')]
        if r._output_paths:
            for p in r._output_paths:
                output_paths.append(p)
        return output_paths

    def launch(self, t):
        assert t._image

        token = secrets.token_urlsafe(8)

        inputs_cmd = ' && '.join([
            f'gsutil -m cp -r {shq(self.gs_input_path(i))} {shq(i._get_path("/tmp"))}'
            for i in t._inputs
        ]) if t._inputs else None

        bash_flags = 'set -e' + ('x' if self.verbose else '') + '; '
        defs = ''.join([r._declare('/tmp') + '; ' for r in t._mentioned])
        make_local_tmpdir = f'mkdir -p /tmp/{t._uid}'
        cmd = bash_flags + defs + make_local_tmpdir + ' && (' + ' && '.join(t._command) + ')'

        outputs = t._internal_outputs.union(t._external_outputs)
        outputs_cmd = ' && '.join([
            f'gsutil -m cp -r {shq(o._get_path("/tmp"))} {shq(output_path)}'
            for o in outputs for output_path in self.gs_output_paths(o, token)
        ]) if t._outputs else None

        config = {
            'uid': t._uid,
            'name': t.name,
            'token': token,
            'inputs_cmd': inputs_cmd,
            'image': t._image,
            'command': cmd,
            'outputs_cmd': outputs_cmd
        }

        print(f'{json.dumps(config)}')

        self.mark_complete(t, token)

    def mark_complete(self, t, token):
        self.task_token[t] = token

        self.n_complete += 1

        for c in self.task_children[t]:
            n = self.task_n_deps[c]
            assert n > 0
            n -= 1
            self.task_n_deps[c] = n
            if n == 0:
                self.ready.append(c)

    def run(self):
        while self.ready:
            t = self.ready.pop()
            self.launch(t)

        assert self.n_complete == len(self.pipeline._tasks)


class HackBackend(Backend):
    def __init__(self):
        pass

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        runner = HackRunner(pipeline, verbose)
        runner.run()
