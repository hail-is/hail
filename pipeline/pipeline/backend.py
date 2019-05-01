import abc
import os
import subprocess as sp
import uuid
import batch.client

from .resource import InputResourceFile, TaskResourceFile
from .utils import escape_string


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
    """

    def __init__(self, tmp_dir='/tmp/'):
        self._tmp_dir = tmp_dir

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
                        absolute_input_path = escape_string(os.path.realpath(r._input_path))
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
                return [f'{cp(dest)} {escape_string(r._input_path)} {escape_string(dest)}'
                        for dest in r._output_paths]
            else:
                assert isinstance(r, TaskResourceFile)
                return [f'{cp(dest)} {r._get_path(tmpdir)} {escape_string(dest)}'
                        for dest in r._output_paths]

        write_inputs = [x for r in pipeline._input_resources for x in copy_external_output(r)]
        if write_inputs:
            script += ["# Write input resources to output destinations"]
            script += write_inputs
            script += ['\n']

        for task in pipeline._tasks:
            os.makedirs(tmpdir + task._uid + '/', exist_ok=True)

            script.append(f"# {task._uid} {task._label if task._label else ''}")

            script += [x for r in task._inputs for x in copy_input(task, r)]

            resource_defs = [r._declare(tmpdir) for r in task._mentioned]

            if task._image:
                defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
                cmd = " && ".join(task._command)
                memory = f'-m {task._memory}' if task._memory else ''
                cpu = f'--cpus={task._cpu}' if task._cpu else ''

                script += [f"docker run "
                           f"-v {tmpdir}:{tmpdir} "
                           f"-w {tmpdir} "
                           f"{memory} "
                           f"{cpu} "
                           f"{task._image} /bin/bash "
                           f"-c {escape_string(defs + cmd)}",
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
                print(e.output)
                raise e
            finally:
                if delete_scratch_on_exit:
                    sp.run(f'rm -rf {tmpdir}', shell=True)

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
        self._batch_client = batch.client.BatchClient(url)

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):  # pylint: disable-msg=R0915
        if dry_run:
            raise NotImplementedError

        hail_scratch_bucket = 'hail-pipeline-scratch'
        subdir_name = 'pipeline-{}'.format(uuid.uuid4().hex[:12])

        remote_tmpdir = f'gs://{hail_scratch_bucket}/{subdir_name}'
        local_tmpdir = f'/tmp/pipeline/{subdir_name}'

        default_image = 'ubuntu'

        batch = self._batch_client.create_batch()
        n_jobs_submitted = 0

        task_to_job_mapping = {}
        job_id_to_command = {}

        volumes = [{'volume': {'name': 'pipeline-test-0-1--hail-is-service-account-key',
                               'secret': {'optional': False,
                                          'secretName': 'pipeline-test-0-1--hail-is-service-account-key'}},
                    'volume_mount': {'mountPath': '/secrets',
                                     'name': 'pipeline-test-0-1--hail-is-service-account-key',
                                     'readOnly': True}}]

        activate_service_account = 'gcloud auth activate-service-account ' \
                                   'pipeline-test-0-1--hail-is@hail-vdc.iam.gserviceaccount.com ' \
                                   '--key-file /secrets/pipeline-test-0-1--hail-is.key'

        def copy_input(r):
            if isinstance(r, InputResourceFile):
                return [f'gsutil cp {escape_string(r._input_path)} {r._get_path(local_tmpdir)}']
            else:
                assert isinstance(r, TaskResourceFile)
                return [f'gsutil cp {r._get_path(remote_tmpdir)} {r._get_path(local_tmpdir)}']

        def copy_internal_output(r):
            assert isinstance(r, TaskResourceFile)
            return [f'gsutil cp {r._get_path(local_tmpdir)} {r._get_path(remote_tmpdir)}']

        def copy_external_output(r):
            if isinstance(r, InputResourceFile):
                return [f'gsutil cp {escape_string(r._input_path)} {escape_string(dest)}'
                        for dest in r._output_paths]
            else:
                assert isinstance(r, TaskResourceFile)
                return [f'gsutil cp {r._get_path(local_tmpdir)} {escape_string(dest)}'
                        for dest in r._output_paths]

        write_inputs = [x for r in pipeline._input_resources for x in copy_external_output(r)]
        if write_inputs:
            write_cmd = activate_service_account + ' && ' + ' && '.join(write_inputs)
            j = batch.create_job(image='google/cloud-sdk:alpine',
                                 command=['/bin/bash', '-c', write_cmd],
                                 attributes={'label': 'write_inputs'},
                                 volumes=volumes)
            job_id_to_command[j.id] = write_cmd
            n_jobs_submitted += 1
            if verbose:
                print(f"Submitted Job {j.id} with command: {write_cmd}")

        for task in pipeline._tasks:
            copy_task_inputs = [x for r in task._inputs for x in copy_input(r)]
            copy_task_outputs = [x for r in task._internal_outputs for x in copy_internal_output(r)]
            copy_task_outputs += [x for r in task._external_outputs for x in copy_external_output(r)]

            local_dirs_needed = [local_tmpdir,
                                 local_tmpdir + '/inputs/',
                                 local_tmpdir + f'/{task._uid}/']
            local_dirs_needed += [local_tmpdir + f'/{dep._uid}/' for dep in task._dependencies]
            make_local_tmpdir = [f'mkdir -p {dir}' for dir in local_dirs_needed]

            task_inputs = make_local_tmpdir + [activate_service_account] + copy_task_inputs
            task_outputs = [activate_service_account] + copy_task_outputs
            resource_defs = [r._declare(directory=local_tmpdir) for r in task._mentioned]

            if task._image is None:
                if verbose:
                    print(f"Using image '{default_image}' since no image was specified.")

            defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
            task_command = [cmd.strip() for cmd in task._command]

            cmd = " && ".join(task_inputs + task_command + task_outputs)
            parent_ids = [task_to_job_mapping[t].id for t in task._dependencies]

            attributes = {'task_uid': task._uid}
            if task._label:
                attributes['label'] = task._label

            resources = {'requests': {}}
            if task._cpu:
                resources['requests']['cpu'] = task._cpu
            if task._memory:
                resources['requests']['memory'] = task._memory

            j = batch.create_job(image=task._image if task._image else default_image,
                                 command=['/bin/bash', '-c', defs + cmd],
                                 parent_ids=parent_ids,
                                 attributes=attributes,
                                 volumes=volumes,
                                 resources=resources)
            n_jobs_submitted += 1

            task_to_job_mapping[task] = j
            job_id_to_command[j.id] = defs + cmd
            if verbose:
                print(f"Submitted Job {j.id} with command: {defs + cmd}")

        if delete_scratch_on_exit:
            parent_ids = list(job_id_to_command.keys())
            rm_cmd = f'gsutil rm -r {remote_tmpdir}'
            cmd = f'{activate_service_account} && {rm_cmd}'
            j = batch.create_job(
                image='google/cloud-sdk:237.0.0-alpine',
                command=['/bin/bash', '-c', cmd],
                parent_ids=parent_ids,
                volumes=volumes,
                attributes={'label': 'remove_tmpdir'},
                always_run=True)
            job_id_to_command[j.id] = cmd
            n_jobs_submitted += 1

        batch.close()
        status = batch.wait()

        failed_jobs = [(j['id'], j['exit_code']) for j in status['jobs'] if 'exit_code' in j and j['exit_code'] > 0]

        fail_msg = ''
        for jid, ec in failed_jobs:
            job = self._batch_client.get_job(jid)
            log = job.log()
            label = job.status()['attributes'].get('label', None)
            fail_msg += (
                f"Job {jid} failed with exit code {ec}:\n"
                f"  Task label:\t{label}\n"
                f"  Command:\t{job_id_to_command[jid]}\n"
                f"  Log:\t{log}\n")

        n_complete = sum([j['state'] == 'Complete' for j in status['jobs']])
        if failed_jobs or n_complete != n_jobs_submitted:
            raise Exception(fail_msg)

        print("Pipeline completed successfully!")
