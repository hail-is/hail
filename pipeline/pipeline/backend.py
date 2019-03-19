import abc
import os
import subprocess as sp
import uuid
import batch.client

from .resource import ResourceFile, ResourceGroup, InputResourceFile, TaskResourceFile
from .utils import escape_string, flatten


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

        for task in pipeline._tasks:
            os.makedirs(tmpdir + task._uid + '/', exist_ok=True)

            script.append(f"# {task._uid} {task._label if task._label else ''}")

            def copy_input(r):
                if isinstance(r, InputResourceFile):
                    if r not in copied_input_resource_files:
                        copied_input_resource_files.add(r)
                        absolute_input_path = os.path.realpath(r._input_path)
                        if task._image is not None:  # pylint: disable-msg=W0640
                            return [f'cp {absolute_input_path} {r._get_path(tmpdir)}']
                        else:
                            return [f'ln -sf {absolute_input_path} {r._get_path(tmpdir)}']
                    else:
                        return []
                elif isinstance(r, ResourceGroup):
                    return [x for _, rf in r._resources.items() for x in copy_input(rf)]
                else:
                    assert isinstance(r, TaskResourceFile)
                    return []

            script += [x for r in task._inputs for x in copy_input(r)]

            resource_defs = [r._declare(tmpdir) for r in task._inputs.union(task._mentioned)]

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
                script += task._command + ['\n']

        def write_pipeline_outputs(r, dest):
            dest = os.path.abspath(dest)
            directory = os.path.dirname(dest)
            os.makedirs(directory, exist_ok=True)

            if isinstance(r, InputResourceFile):
                return [f'cp {r._input_path} {dest}']
            elif isinstance(r, TaskResourceFile):
                return [f'cp {r._get_path(tmpdir)} {dest}']
            else:
                assert isinstance(r, ResourceGroup)
                return [write_pipeline_outputs(rf, dest + '.' + ext) for ext, rf in r._resources.items()]

        outputs = [x for _, r in pipeline._resource_map.items()
                   for dest in r._output_paths
                   for x in write_pipeline_outputs(r, dest)]

        script += ["# Write resources to output destinations"]
        script += outputs

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
                    sp.run(f'rm -r {tmpdir}', shell=True)

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

    def __init__(self, url, headers):
        self._batch_client = batch.client.BatchClient(url, headers=headers)

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):  # pylint: disable-msg=R0915
        if dry_run:
            raise NotImplementedError

        hail_scratch_bucket = 'hail-pipeline-scratch'
        subdir_name = 'pipeline-{}'.format(uuid.uuid4().hex[:12])

        remote_tmpdir = f'gs://{hail_scratch_bucket}/{subdir_name}'
        local_tmpdir = f'/io'

        default_image = 'ubuntu'

        batch = self._batch_client.create_batch()
        n_jobs_submitted = 0

        task_to_job_mapping = {}
        job_id_to_command = {}

        for task in pipeline._tasks:
            def copy_input(r):
                if isinstance(r, ResourceFile):
                    if isinstance(r, InputResourceFile):
                        return [(r._input_path, r._get_path(local_tmpdir))]
                    else:
                        assert isinstance(r, TaskResourceFile)
                        return [(r._get_path(remote_tmpdir), r._get_path(local_tmpdir))]
                else:
                    assert isinstance(r, ResourceGroup)
                    return [x for _, rf in r._resources.items() for x in copy_input(rf)]

            def copy_output(r):
                assert r._source == task  # pylint: disable-msg=W0640
                if isinstance(r, TaskResourceFile):
                    return [
                        (r._get_path(local_tmpdir), r._get_path(remote_tmpdir)),
                        *[(r._get_path(local_tmpdir), output_path) for output_path in r._output_paths]]
                else:
                    assert isinstance(r, ResourceGroup)
                    return [x for _, rf in r._resources.items() for x in copy_output(rf)]

            copy_task_inputs = [x for r in task._inputs for x in copy_input(r)]
            copy_task_outputs = [x for r in task._outputs for x in copy_output(r)]

            resource_defs = [r._declare(directory=local_tmpdir) for r in task._inputs.union(task._mentioned)]

            if task._image is None:
                if verbose:
                    print(f"Using image '{default_image}' since no image was specified.")

            defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
            task_command = [cmd.strip() for cmd in task._command]
            cmd = defs + ' && ' + task_command

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
                                 command=['/bin/bash', '-c', cmd],
                                 parent_ids=parent_ids,
                                 attributes=attributes,
                                 resources=resources,
                                 input_files=copy_task_inputs,
                                 output_files=copy_task_outputs)
            n_jobs_submitted += 1

            task_to_job_mapping[task] = j
            job_id_to_command[j.id] = defs + cmd
            if verbose:
                print(f"Submitted Job {j.id} with command: {defs + cmd}")

        status = batch.wait()

        if delete_scratch_on_exit:
            j = self._batch_client.create_job(
                image='google/cloud-sdk:237.0.0-alpine',
                command=['/bin/bash', '-c',
                         f'gcloud -q auth activate-service-account --key-file=/gcp-sa-key/key.json && '
                         f'gsutil rm -r {remote_tmpdir}'],
                attributes={'label': 'remove_tmpdir'})

        failed_jobs = [(int(jid), ec) for jid, ec in status['exit_codes'].items() if ec is not None and ec > 0]

        fail_msg = ''
        for jid, ec in failed_jobs:
            jstatus = self._batch_client.get_job(jid).status()
            log = jstatus['log']
            label = jstatus['attributes'].get('label', None)
            fail_msg += (
                f"Job {jid} failed with exit code {ec}:\n"
                f"  Task label:\t{label}\n"
                f"  Command:\t{job_id_to_command[jid]}\n"
                f"  Log:\t{log}\n")

        if failed_jobs or status['jobs']['Complete'] != n_jobs_submitted:
            raise Exception(fail_msg)

        print("Pipeline completed successfully!")
