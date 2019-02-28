import abc
import os
import subprocess as sp
import uuid
import batch.client

from .resource import ResourceFile, ResourceGroup, InputResourceFile, TaskResourceFile
from .utils import escape_string, flatten


class Backend:
    @abc.abstractmethod
    def run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        return


class LocalBackend(Backend):
    def __init__(self, tmp_dir='/tmp/'):
        self._tmp_dir = tmp_dir

    def run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        tmpdir = self.tmp_dir()

        script = ['#!/bin/bash',
                  'set -e' + 'x' if verbose else '',
                  '\n',
                  '# change cd to tmp directory',
                  f"cd {tmpdir}",
                  '\n']

        for task in pipeline._tasks:
            script.append(f"# {task._uid} {task._label if task._label else ''}")

            def copy_input(r):
                if isinstance(r, InputResourceFile):
                    absolute_input_path = os.path.realpath(r._input_path)
                    if task._image is not None:  # pylint: disable-msg=W0640
                        return [f'cp {absolute_input_path} {tmpdir}/{r.path}']
                    else:
                        return [f'ln -sf {absolute_input_path} {tmpdir}/{r.path}']
                elif isinstance(r, ResourceGroup):
                    return [x for _, rf in r._resources.items() for x in copy_input(rf)]
                else:
                    assert isinstance(r, TaskResourceFile)
                    return []

            script += [x for r in task._inputs for x in copy_input(r)]

            resource_defs = [r._declare() for r in task._inputs.union(task._mentioned)]

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
                return [f'cp {r.path} {dest}']
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

    def tmp_dir(self):
        def _get_random_name():
            directory = self._tmp_dir + '/pipeline.{}/'.format(uuid.uuid4().hex[:8])

            if os.path.isdir(directory):
                return _get_random_name()
            else:
                os.makedirs(directory, exist_ok=True)
                return directory

        return _get_random_name()


class BatchBackend(Backend):
    def __init__(self, url):
        self._batch_client = batch.client.BatchClient(url)

    def run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):  # pylint: disable-msg=R0915
        if dry_run:
            raise NotImplementedError

        hail_scratch_bucket = 'hail-pipeline-scratch'
        subdir_name = 'pipeline-{}'.format(uuid.uuid4().hex[:12])

        remote_tmpdir = f'gs://{hail_scratch_bucket}/{subdir_name}/'
        local_tmpdir = '/tmp/pipeline/'

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

        for task in pipeline._tasks:
            def copy_input(r):
                if isinstance(r, ResourceFile):
                    if isinstance(r, InputResourceFile):
                        return [f'gsutil cp {r._input_path} {local_tmpdir}/{r.path}']
                    else:
                        assert isinstance(r, TaskResourceFile)
                        return [f'gsutil cp {remote_tmpdir}/{r.path} {local_tmpdir}/{r.path}']
                else:
                    assert isinstance(r, ResourceGroup)
                    return [x for _, rf in r._resources.items() for x in copy_input(rf)]

            def copy_output(r):
                assert r._source == task  # pylint: disable-msg=W0640
                if isinstance(r, TaskResourceFile):
                    return [f'gsutil cp {local_tmpdir}/{r.path} {remote_tmpdir}/{r.path}']
                else:
                    assert isinstance(r, ResourceGroup)
                    return [x for _, rf in r._resources.items() for x in copy_output(rf)]

            copy_task_inputs = [x for r in task._inputs for x in copy_input(r)]
            copy_task_outputs = [x for r in task._outputs for x in copy_output(r)]

            make_local_tmpdir = f'mkdir -p {local_tmpdir}'

            task_inputs = [make_local_tmpdir, activate_service_account] + copy_task_inputs
            task_outputs = [activate_service_account] + copy_task_outputs
            resource_defs = [r._declare(directory=local_tmpdir) for r in task._inputs.union(task._mentioned)]

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

        def write_pipeline_outputs(r, dest):
            if isinstance(r, InputResourceFile):
                copy_output = activate_service_account + ' && ' + f'gsutil cp {r._input_path} {dest}'
                return [(task_to_job_mapping[r._source].id, copy_output)]
            elif isinstance(r, TaskResourceFile):
                copy_output = activate_service_account + ' && ' + f'gsutil cp {remote_tmpdir}/{r.path} {dest}'
                return [(task_to_job_mapping[r._source].id, copy_output)]
            else:
                assert isinstance(r, ResourceGroup)
                return [write_pipeline_outputs(rf, dest + '.' + ext) for ext, rf in r._resources.items()]

        pipeline_outputs = list(flatten([write_pipeline_outputs(r, dest)
                                         for _, r in pipeline._resource_map.items()
                                         for dest in r._output_paths]))

        for parent_id, write_cmd in pipeline_outputs:
            j = batch.create_job(image='google/cloud-sdk:alpine',
                                 command=['/bin/bash', '-c', write_cmd],
                                 parent_ids=[parent_id],
                                 attributes={'label': 'write_output'},
                                 volumes=volumes)
            job_id_to_command[j.id] = write_cmd
            n_jobs_submitted += 1
            if verbose:
                print(f"Submitted Job {j.id} with command: {write_cmd}")

        status = batch.wait()

        if delete_scratch_on_exit:
            rm_cmd = f'gsutil rm -r {remote_tmpdir}'
            j = self._batch_client.create_job(image='google/cloud-sdk:alpine',
                                              command=['/bin/bash', '-c', activate_service_account + '&&' + rm_cmd],
                                              volumes=volumes,
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
        else:
            print("Pipeline completed successfully!")
