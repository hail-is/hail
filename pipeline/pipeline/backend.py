import abc
import os
import subprocess as sp
import batch.client

from google.cloud import storage

from .resource import ResourceFile, ResourceGroup, InputResourceFile, TaskResourceFile
from .utils import get_sha, escape_string, flatten


def get_path(r):
    if isinstance(r, ResourceFile):
        assert r._value is not None
        return r._value
    else:
        assert isinstance(r, ResourceGroup)
        return r._root


class Backend:
    @abc.abstractmethod
    def tmp_dir(self):
        return

    @abc.abstractmethod
    def run(self, pipeline, dry_run, verbose, bg, delete_on_exit):
        return


class LocalBackend(Backend):
    def __init__(self, tmp_dir='/tmp/'):
        self._tmp_dir = tmp_dir

    def run(self, pipeline, dry_run, verbose, bg, delete_on_exit):
        tmpdir = self.tmp_dir()

        script = ['#!/bin/bash',
                  'set -e' + 'x' if verbose else '',
                  '\n',
                  '# change cd to tmp directory',
                  f"cd {tmpdir}",
                  '\n']

        def define_resource(r):
            r = pipeline._get_resource(r)
            return f"{r._uid}={escape_string(get_path(r))}"

        for task in pipeline._tasks:
            script.append(f"# {task._uid} {task._label if task._label else ''}")

            # Make copies of input resource files if they don't already exist
            def copy_task_inputs(r):
                r = pipeline._get_resource(r)
                if isinstance(r, InputResourceFile):
                    absolute_input_path = os.path.realpath(r._input_path)
                    if task._docker:
                        return [f'cp {absolute_input_path} {tmpdir}/{get_path(r)}']
                    else:
                        return [f'ln -sf {absolute_input_path} {tmpdir}/{get_path(r)}']
                elif isinstance(r, ResourceGroup):
                    return [copy_task_inputs(rf) for _, rf in r._resources.items()]
                else:
                    assert isinstance(r, TaskResourceFile)
                    pass

            script += list(flatten([copy_task_inputs(r) for r in task._inputs]))

            resource_defs = [define_resource(r) for r in task._inputs.union(task._outputs)]

            if task._docker:
                defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
                cmd = " && ".join(task._command)
                image = task._docker
                script += [f"docker run "
                           f"-v {tmpdir}:{tmpdir} "
                           f"-w {tmpdir} "
                           f"{image} /bin/bash "
                           f"-c {escape_string(defs + cmd)}",
                           '\n']
            else:
                script += resource_defs
                script += task._command + ['\n']

        def write_pipeline_outputs(r, dest):
            r = pipeline._get_resource(r)
            if isinstance(r, InputResourceFile):
                return [f'cp {r._input_path} {dest}']
            elif isinstance(r, TaskResourceFile):
                return [f'cp {get_path(r)} {dest}']
            else:
                assert isinstance(r, ResourceGroup)
                return [write_pipeline_outputs(rf, dest + '.' + ext) for ext, rf in r._resources.items()]

        outputs = list(flatten([write_pipeline_outputs(r, dest)
                                for _, r in pipeline._resource_map.items()
                                for dest in r._output_paths]))
        script += ["# Write resources to output destinations"]
        script += outputs

        script = "\n".join(script)
        print(script)
        if dry_run:
            print(script)
        else:
            try:
                sp.check_output(script, shell=True)  # FIXME: implement non-blocking (bg = True)
            except sp.CalledProcessError as e:
                print(e.output)
                raise e
            finally:
                if delete_on_exit:
                    sp.run(f'rm -r {tmpdir}', shell=True)

    def tmp_dir(self):
        def _get_random_name():
            directory = self._tmp_dir + '/pipeline.{}/'.format(get_sha(8))

            if os.path.isdir(directory):
                return _get_random_name()
            else:
                os.makedirs(directory, exist_ok=True)
                return directory

        return _get_random_name()


class BatchBackend(Backend):
    def __init__(self, url='http://localhost:5000'):
        self._batch_client = batch.client.BatchClient(url)
        self._gcs_client = storage.Client()

    def run(self, pipeline, dry_run, verbose, bg, delete_on_exit):  # pylint: disable-msg=R0915
        remote_tmpdir = self.tmp_dir()
        local_tmpdir = '/tmp/pipeline/'

        batch = self._batch_client.create_batch()
        n_jobs_submitted = 0

        def define_resource(r):
            r = pipeline._get_resource(r)
            return f"{r._uid}={escape_string(local_tmpdir + '/' + get_path(r))}"

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
            def copy_task_inputs(r):
                r = pipeline._get_resource(r)
                if isinstance(r, ResourceFile):
                    if isinstance(r, InputResourceFile):
                        return [f'gsutil cp {r._input_path} {local_tmpdir}/{get_path(r)}']  # FIXME: This is serial!
                    else:
                        assert isinstance(r, TaskResourceFile)
                        return [f'gsutil cp {remote_tmpdir}/{get_path(r)} {local_tmpdir}/{get_path(r)}']
                else:
                    assert isinstance(r, ResourceGroup)
                    return [copy_task_inputs(rf) for _, rf in r._resources.items()]

            def copy_task_outputs(r):
                r = pipeline._get_resource(r)
                assert r._source == task  # pylint: disable-msg=W0640
                if isinstance(r, TaskResourceFile):
                    return [f'gsutil cp {local_tmpdir}/{get_path(r)} {remote_tmpdir}/{get_path(r)}']
                else:
                    assert isinstance(r, ResourceGroup)
                    return [copy_task_outputs(rf) for _, rf in r._resources.items()]

            make_local_tmpdir = f'mkdir -p {local_tmpdir}'

            task_inputs = [make_local_tmpdir, activate_service_account] + \
                list(flatten([copy_task_inputs(r) for r in task._inputs]))
            task_outputs = [activate_service_account] + \
                list(flatten([copy_task_outputs(r) for r in task._outputs]))
            resource_defs = [define_resource(r) for r in task._inputs.union(task._outputs)]

            image = 'google/cloud-sdk:alpine'  # task._docker if task._docker else pipeline._default_image
            defs = '; '.join(resource_defs) + '; ' if resource_defs else ''
            cmd = " && ".join(task_inputs + task._command + task_outputs)
            parent_ids = [task_to_job_mapping[t] for t in task._dependencies]
            attributes = {'task_uid': task._uid}
            if task._label:
                attributes['label'] = task._label

            # FIXME: add memory, cpu, etc.
            j = batch.create_job(image=image,
                                 command=['/bin/bash', '-c', defs + cmd],
                                 parent_ids=parent_ids,
                                 attributes=attributes,
                                 volumes=volumes)
            n_jobs_submitted += 1

            task_to_job_mapping[task] = j
            job_id_to_command[j.id] = defs + cmd

        def write_pipeline_outputs(r, dest):
            r = pipeline._get_resource(r)
            if isinstance(r, InputResourceFile):
                return [(task_to_job_mapping[r._source].id, activate_service_account +
                         ' && ' + f'gsutil cp {r._input_path} {dest}')]
            elif isinstance(r, TaskResourceFile):
                return [(task_to_job_mapping[r._source].id, activate_service_account +
                         ' && ' + f'gsutil cp {remote_tmpdir}/{get_path(r)} {dest}')]
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

        status = batch.wait()  # FIXME: add background mode

        # cleanup scratch directory
        if delete_on_exit:
            rm_cmd = f'gsutil rm -r {remote_tmpdir}'
            j = self._batch_client.create_job(image='google/cloud-sdk:alpine',
                                              command=['/bin/bash', '-c', activate_service_account + '&&' + rm_cmd],
                                              volumes=volumes,
                                              attributes={'label': 'remove_tmpdir'})

        failed_jobs = [(jid, ec) for jid, ec in status['exit_codes'].items() if ec > 0]

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

    def tmp_dir(self):
        hail_scratch_bucket = 'hail-pipeline-scratch'

        def create_bucket(n_attempts):
            subdir_name = 'pipeline-{}'.format(get_sha(16).lower())
            bucket = self._gcs_client.bucket(hail_scratch_bucket)  # FIXME: Want to create separate bucket per pipeline
            f = bucket.blob(subdir_name)
            if not f.exists(self._gcs_client):
                return f'gs://{hail_scratch_bucket}/{subdir_name}/'
            else:
                if n_attempts > 3:
                    raise Exception("Too many attempts to create a bucket!")
                return create_bucket(n_attempts + 1)

        return create_bucket(0)
