import json
import logging
import os
import threading
import uuid
import traceback
from shlex import quote as shq
import asyncio
import requests
import kubernetes as kube

from .globals import states, complete_states, valid_state_transitions, tasks
from .batch_configuration import INSTANCE_ID
from .log_store import LogStore

log = logging.getLogger('batch')


def copy(files):
    if files is None:
        return 'true'

    authenticate = 'set -ex; gcloud -q auth activate-service-account --key-file=/gsa-key/privateKeyData'

    def copy_command(src, dst):
        if not dst.startswith('gs://'):
            mkdirs = f'mkdir -p {shq(os.path.dirname(dst))};'
        else:
            mkdirs = ""
        return f'{mkdirs} gsutil -m cp -R {shq(src)} {shq(dst)}'

    copies = ' && '.join([copy_command(src, dst) for (src, dst) in files])
    return f'{authenticate} && {copies}'


class JobStateWriteFailure(Exception):
    pass


class Job:
    @staticmethod
    def _copy_container(name, files):
        sh_expression = copy(files)
        return kube.client.V1Container(
            image='google/cloud-sdk:237.0.0-alpine',
            name=name,
            command=['/bin/sh', '-c', sh_expression],
            resources=kube.client.V1ResourceRequirements(
                requests={'cpu': '500m' if files else '100m'}),
            volume_mounts=[kube.client.V1VolumeMount(
                mount_path='/batch-gsa-key',
                name='batch-gsa-key')])

    async def _create_pod(self):
        assert self.userdata is not None
        assert self._state in states
        assert self._state == 'Running'

        input_container = Job._copy_container('setup', self.input_files)
        output_container = Job._copy_container('cleanup', self.output_files)

        volumes = [
            kube.client.V1Volume(
                secret=kube.client.V1SecretVolumeSource(
                    secret_name=self.userdata['gsa_key_secret_name']),
                name='gsa-key'),
            kube.client.V1Volume(
                secret=kube.client.V1SecretVolumeSource(
                    secret_name='batch-gsa-key'),
                name='batch-gsa-key')]

        volume_mounts = [
            kube.client.V1VolumeMount(
                mount_path='/gsa-key',
                name='gsa-key')]  # FIXME: this shouldn't be mounted to every container

        if self._pvc_name is not None:
            volumes.append(kube.client.V1Volume(
                empty_dir=kube.client.V1EmptyDirVolumeSource(
                    size_limit=self._pvc_size),
                name=self._pvc_name))
            volume_mounts.append(kube.client.V1VolumeMount(
                mount_path='/io',
                name=self._pvc_name))

        pod_spec = self.app['k8s_client'].api_client._ApiClient__deserialize(self._pod_spec, kube.client.V1PodSpec)
        pod_spec.containers = [input_container, pod_spec.containers[0], output_container]

        if pod_spec.volumes is None:
            pod_spec.volumes = []
        pod_spec.volumes.extend(volumes)

        for container in pod_spec.containers:
            if container.volume_mounts is None:
                container.volume_mounts = []
            container.volume_mounts.extend(volume_mounts)

        pod_template = kube.client.V1Pod(
            metadata=kube.client.V1ObjectMeta(
                name=self._pod_name,
                labels={'app': 'batch-job',
                        'hail.is/batch-instance': INSTANCE_ID,
                        'batch_id': str(self.batch_id),
                        'job_id': str(self.job_id),
                        'user': self.user
                        }),
            spec=pod_spec)

        err = await self.app['driver'].create_pod(spec=pod_template.to_dict(),
                                                  output_directory=self.directory)
        if err is not None:
            if err.status == 409:
                log.info(f'pod already exists for job {self.id}')
                return
            traceback.print_tb(err.__traceback__)
            log.info(f'pod creation failed for job {self.id} '
                     f'with the following error: {err}')

    async def _delete_pod(self):
        err = await self.app['driver'].delete_pod(name=self._pod_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'ignoring pod deletion failure for job {self.id} due to {err}')

    async def _read_logs(self):
        if self._state in ('Pending', 'Cancelled'):
            return None

        async def _read_log_from_gcs(task_name):
            pod_log, err = await self.app['log_store'].read_gs_file(LogStore.container_log_path(self.directory, task_name))
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read log for {self.id}, {task_name} '
                         f'due to {err}')
            return task_name, pod_log

        async def _read_log_from_worker(task_name):
            pod_log, err = await self.app['driver'].read_pod_log(self._pod_name, container=task_name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read log for {self.id}, {task_name} '
                         f'due to {err}; will still try to load other containers')
            return task_name, pod_log

        if self._state == 'Running':
            future_logs = asyncio.gather(*[_read_log_from_worker(task) for task in tasks])
            return {k: v for k, v in await future_logs}
        assert self._state in ('Error', 'Failed', 'Success')
        future_logs = asyncio.gather(*[_read_log_from_gcs(task) for task in tasks])
        return {k: v for k, v in await future_logs}

    async def _read_pod_status(self):
        if self._state in ('Pending', 'Cancelled'):
            return None

        async def _read_status_from_gcs(task_name):
            status, err = await self.app['log_store'].read_gs_file(LogStore.container_status_path(self.directory, task_name))
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read container status for {self.id} '
                         f'due to {err}')
            return task_name, status

        async def _read_status_from_worker(task_name):
            status, err = await self.app['driver'].read_container_status(self._pod_name, container=task_name)
            log.info(f'status {status} err {err}')
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read container status for {self.id} '
                         f'due to {err}; will still try to load other containers')
            return task_name, status

        if self._state == 'Running':
            future_statuses = asyncio.gather(*[_read_status_from_worker(task) for task in tasks])
            return {k: v for k, v in await future_statuses}
        assert self._state in ('Error', 'Failed', 'Success')
        future_statuses = asyncio.gather(*[_read_status_from_gcs(task) for task in tasks])
        return {k: v for k, v in await future_statuses}

    async def _delete_gs_files(self):
        errs = await self.app['log_store'].delete_gs_files(self.directory)
        for file, err in errs:
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'could not delete {self.directory}/{file} for job {self.id} due to {err}')

    @staticmethod
    def from_record(app, record):
        if record is not None:
            attributes = json.loads(record['attributes'])
            userdata = json.loads(record['userdata'])
            pod_spec = json.loads(record['pod_spec'])
            input_files = json.loads(record['input_files'])
            output_files = json.loads(record['output_files'])
            exit_codes = json.loads(record['exit_codes'])
            durations = json.loads(record['durations'])
            messages = json.loads(record['messages'])

            return Job(app, batch_id=record['batch_id'], job_id=record['job_id'], attributes=attributes,
                       callback=record['callback'], userdata=userdata, user=record['user'],
                       always_run=record['always_run'], exit_codes=exit_codes, messages=messages,
                       durations=durations, state=record['state'], pvc_size=record['pvc_size'],
                       cancelled=record['cancelled'], directory=record['directory'],
                       token=record['token'], pod_spec=pod_spec, input_files=input_files,
                       output_files=output_files)

        return None

    @staticmethod
    async def from_pod(app, pod):
        if pod.metadata.labels is None:
            return None
        if not {'batch_id', 'job_id', 'user'}.issubset(set(pod.metadata.labels)):
            return None
        batch_id = pod.metadata.labels['batch_id']
        job_id = pod.metadata.labels['job_id']
        user = pod.metadata.labels['user']
        return await Job.from_db(app, batch_id, job_id, user)

    @staticmethod
    async def from_db(app, batch_id, job_id, user):
        jobs = await Job.from_db_multiple(app, batch_id, job_id, user)
        if len(jobs) == 1:
            return jobs[0]
        return None

    @staticmethod
    async def from_db_multiple(app, batch_id, job_ids, user):
        records = await app['db'].jobs.get_undeleted_records(batch_id, job_ids, user)
        jobs = [Job.from_record(app, record) for record in records]
        return jobs

    @staticmethod
    def create_job(app, jobs_builder, pod_spec, batch_id, job_id, attributes, callback,
                   parent_ids, input_files, output_files, userdata, always_run,
                   pvc_size, state):
        cancelled = False
        user = userdata['username']
        token = uuid.uuid4().hex[:6]

        exit_codes = [None for _ in tasks]
        durations = [None for _ in tasks]
        messages = [None for _ in tasks]
        directory = app['log_store'].gs_job_output_directory(batch_id, job_id, token)
        pod_spec = app['k8s_client'].api_client.sanitize_for_serialization(pod_spec)

        jobs_builder.create_job(
            batch_id=batch_id,
            job_id=job_id,
            state=state,
            pvc_size=pvc_size,
            callback=callback,
            attributes=json.dumps(attributes),
            always_run=always_run,
            token=token,
            pod_spec=json.dumps(pod_spec),
            input_files=json.dumps(input_files),
            output_files=json.dumps(output_files),
            directory=directory,
            exit_codes=json.dumps(exit_codes),
            durations=json.dumps(durations),
            messages=json.dumps(messages))

        for parent in parent_ids:
            jobs_builder.create_job_parent(
                batch_id=batch_id,
                job_id=job_id,
                parent_id=parent)

        job = Job(app, batch_id=batch_id, job_id=job_id, attributes=attributes, callback=callback,
                  userdata=userdata, user=user, always_run=always_run, exit_codes=exit_codes,
                  messages=messages, durations=durations, state=state, pvc_size=pvc_size,
                  cancelled=cancelled, directory=directory, token=token,
                  pod_spec=pod_spec, input_files=input_files, output_files=output_files)

        return job

    def __init__(self, app, batch_id, job_id, attributes, callback, userdata, user, always_run,
                 exit_codes, messages, durations, state, pvc_size, cancelled, directory,
                 token, pod_spec, input_files, output_files):
        self.app = app
        self.batch_id = batch_id
        self.job_id = job_id
        self.id = (batch_id, job_id)

        self.attributes = attributes
        self.callback = callback
        self.always_run = always_run
        self.userdata = userdata
        self.user = user
        self.exit_codes = exit_codes
        self.messages = messages
        self.directory = directory
        self.durations = durations
        self.token = token
        self.input_files = input_files
        self.output_files = output_files

        name = f'batch-{batch_id}-job-{job_id}-{token}'
        self._pod_name = name
        self._pvc_name = name if pvc_size else None
        self._pvc_size = pvc_size
        self._state = state
        self._cancelled = cancelled
        self._pod_spec = pod_spec

    async def refresh_parents_and_maybe_create(self):
        for record in await self.app['db'].jobs.get_parents(*self.id):
            parent_job = Job.from_record(self.app, record)
            assert parent_job.batch_id == self.batch_id
            await self.parent_new_state(parent_job._state, *parent_job.id)

    async def set_state(self, new_state):
        assert new_state in valid_state_transitions[self._state], f'{self._state} -> {new_state}'
        if self._state != new_state:
            n_updated = await self.app['db'].jobs.update_record(*self.id, compare_items={'state': self._state}, state=new_state)
            if n_updated == 0:
                log.warning(f'changing the state from {self._state} -> {new_state} '
                            f'for job {self.id} failed due to the expected state not in db')
                raise JobStateWriteFailure()

            log.info('job {} changed state: {} -> {}'.format(
                self.id,
                self._state,
                new_state))
            self._state = new_state
            await self.notify_children(new_state)

    async def notify_children(self, new_state):
        if new_state not in complete_states:
            return

        children = [Job.from_record(self.app, record) for record in await self.app['db'].jobs.get_children(*self.id)]
        for child in children:
            await child.parent_new_state(new_state, *self.id)

    async def parent_new_state(self, new_state, parent_batch_id, parent_job_id):
        del parent_job_id
        assert parent_batch_id == self.batch_id
        if new_state in complete_states:
            await self.create_if_ready()

    async def create_if_ready(self):
        incomplete_parent_ids = await self.app['db'].jobs.get_incomplete_parents(*self.id)
        if self._state == 'Pending' and not incomplete_parent_ids:
            await self.set_state('Running')
            parents = [Job.from_record(self.app, record) for record in await self.app['db'].jobs.get_parents(*self.id)]
            if (self.always_run or
                    (not self._cancelled and all(p.is_successful() for p in parents))):
                log.info(f'all parents complete for {self.id},'
                         f' creating pod')
                await self._create_pod()
            else:
                log.info(f'parents deleted, cancelled, or failed: cancelling {self.id}')
                await self.set_state('Cancelled')

    async def cancel(self):
        self._cancelled = True

        if not self.always_run and self._state == 'Running':
            await self.set_state('Cancelled')  # must call before deleting resources to prevent race conditions
            await self._delete_pod()

    def is_complete(self):
        return self._state in complete_states

    def is_successful(self):
        return self._state == 'Success'

    async def mark_unscheduled(self):
        updated_job = await Job.from_db(self.app, *self.id, self.user)
        if updated_job.is_complete():
            log.info(f'job is already completed in db, not rescheduling pod')
            return

        await self._delete_pod()
        if self._state == 'Running' and (not self._cancelled or self.always_run):
            await self._create_pod()

    async def mark_complete(self, pod):  # pylint: disable=R0915
        def process_container(status):
            state = status.state
            ec = None
            duration = None
            message = None

            if state.terminated:
                ec = state.terminated.exit_code
                if state.terminated.started_at and state.terminated.finished_at:
                    duration = max(0, (state.terminated.finished_at - state.terminated.started_at).total_seconds())
                message = state.terminated.message
            else:
                assert state.waiting, state
                if state.waiting.message:
                    message = state.waiting.message

            return ec, duration, message

        exit_codes, durations, messages = zip(*[process_container(status)
                                                for status in pod.status.container_statuses])  # FIXME: use gear unzip?
        exit_codes = list(exit_codes)
        durations = list(durations)
        messages = list(messages)

        if pod.status.phase == 'Succeeded':
            new_state = 'Success'
        elif any(messages):
            new_state = 'Error'
        else:
            new_state = 'Failed'

        n_updated = await self.app['db'].jobs.update_record(*self.id,
                                                            compare_items={'state': self._state},
                                                            durations=json.dumps(durations),
                                                            exit_codes=json.dumps(exit_codes),
                                                            messages=json.dumps(messages),
                                                            state=new_state)

        if n_updated == 0:
            log.info(f'could not update job {self.id} due to db not matching expected state')
            raise JobStateWriteFailure()

        self.exit_codes = exit_codes
        self.durations = durations

        if self._state != new_state:
            log.info('job {} changed state: {} -> {}'.format(
                self.id,
                self._state,
                new_state))

        self._state = new_state

        await self._delete_pod()
        await self.notify_children(new_state)

        log.info('job {} complete with state {}, exit_codes {}'.format(self.id, self._state, self.exit_codes))

        if self.callback:
            def handler(id, callback, json):
                try:
                    requests.post(callback, json=json, timeout=120)
                except requests.exceptions.RequestException as exc:
                    log.warning(
                        f'callback for job {id} failed due to an error, I will not retry. '
                        f'Error: {exc}')

            threading.Thread(target=handler, args=(self.id, self.callback, self.to_dict())).start()

        if self.batch_id:
            batch = await Batch.from_db(self.app, self.batch_id, self.user)
            if batch is not None:
                await batch.mark_job_complete(self)

    def to_dict(self):
        result = {
            'batch_id': self.batch_id,
            'job_id': self.job_id,
            'state': self._state
        }
        if self.is_complete():
            result['exit_code'] = {k: v for k, v in zip(tasks, self.exit_codes)}
            result['duration'] = {k: v for k, v in zip(tasks, self.durations)}
            result['message'] = {k: v for k, v in zip(tasks, self.messages)}

        if self.attributes:
            result['attributes'] = self.attributes
        return result


class Batch:
    @staticmethod
    def from_record(app, record, deleted=False):
        if record is not None:
            if not deleted:
                assert not record['deleted']
            attributes = json.loads(record['attributes'])
            userdata = json.loads(record['userdata'])

            if record['n_failed'] > 0:
                state = 'failure'
            elif record['n_cancelled'] > 0:
                state = 'cancelled'
            elif record['closed'] and record['n_succeeded'] == record['n_jobs']:
                state = 'success'
            else:
                state = 'running'

            complete = record['closed'] and record['n_completed'] == record['n_jobs']

            return Batch(app,
                         id=record['id'],
                         attributes=attributes,
                         callback=record['callback'],
                         userdata=userdata,
                         user=record['user'],
                         state=state,
                         complete=complete,
                         deleted=record['deleted'],
                         cancelled=record['cancelled'],
                         closed=record['closed'])
        return None

    @staticmethod
    async def from_db(app, ids, user):
        batches = await Batch.from_db_multiple(app, ids, user)
        if len(batches) == 1:
            return batches[0]
        return None

    @staticmethod
    async def from_db_multiple(app, ids, user):
        records = await app['db'].batch.get_undeleted_records(ids, user)
        batches = [Batch.from_record(app, record) for record in records]
        return batches

    @staticmethod
    async def create_batch(app, attributes, callback, userdata, n_jobs):
        user = userdata['username']

        id = await app['db'].batch.new_record(
            attributes=json.dumps(attributes),
            callback=callback,
            userdata=json.dumps(userdata),
            user=user,
            deleted=False,
            cancelled=False,
            closed=False,
            n_jobs=n_jobs)

        batch = Batch(app, id=id, attributes=attributes, callback=callback,
                      userdata=userdata, user=user, state='running',
                      complete=False, deleted=False, cancelled=False,
                      closed=False)

        if attributes is not None:
            items = [{'batch_id': id, 'key': k, 'value': v} for k, v in attributes.items()]
            success = await app['db'].batch_attributes.new_records(items)
            if not success:
                await batch.delete()
                return

        return batch

    def __init__(self, app, id, attributes, callback, userdata, user,
                 state, complete, deleted, cancelled, closed):
        self.app = app
        self.id = id
        self.attributes = attributes
        self.callback = callback
        self.userdata = userdata
        self.user = user
        self.state = state
        self.complete = complete
        self.deleted = deleted
        self.cancelled = cancelled
        self.closed = closed

    async def get_jobs(self, limit=None, offset=None):
        jobs = await self.app['db'].jobs.get_records_by_batch(self.id, limit, offset)
        return [Job.from_record(self.app, record) for record in jobs]

    # called by driver
    async def _cancel_jobs(self):
        await asyncio.gather(*[j.cancel() for j in await self.get_jobs()])

    # called by front end
    async def cancel(self):
        await self.app['db'].batch.update_record(self.id, cancelled=True, closed=True)
        self.cancelled = True
        self.closed = True
        log.info(f'batch {self.id} cancelled')

    # called by driver
    async def _close_jobs(self):
        await asyncio.gather(*[j._create_pod() for j in await self.get_jobs()
                               if j._state == 'Running'])

    # called by front end
    async def close(self):
        await self.app['db'].batch.update_record(self.id, closed=True)
        self.closed = True
        log.info(f'batch {self.id} closed')

    # called by driver
    # FIXME make called by front end
    async def mark_deleted(self):
        await self.cancel()
        await self.app['db'].batch.update_record(self.id,
                                                 deleted=True)
        self.deleted = True
        self.closed = True
        log.info(f'batch {self.id} marked for deletion')

    async def delete(self):
        # Job deleted from database when batch is deleted with delete cascade
        await asyncio.gather(*[j._delete_gs_files() for j in await self.get_jobs()])
        await self.app['db'].batch.delete_record(self.id)
        log.info(f'batch {self.id} deleted')

    async def mark_job_complete(self, job):
        if self.callback:
            def handler(id, job_id, callback, json):
                try:
                    requests.post(callback, json=json, timeout=120)
                except requests.exceptions.RequestException as exc:
                    log.warning(
                        f'callback for batch {id}, job {job_id} failed due to an error, I will not retry. '
                        f'Error: {exc}')

            threading.Thread(
                target=handler,
                args=(self.id, job.id, self.callback, job.to_dict())
            ).start()

    def is_complete(self):
        return self.complete

    def is_successful(self):
        return self.state == 'success'

    async def to_dict(self, include_jobs=False, limit=None, offset=None):
        result = {
            'id': self.id,
            'state': self.state,
            'complete': self.complete,
            'closed': self.closed
        }
        if self.attributes:
            result['attributes'] = self.attributes
        if include_jobs:
            jobs = await self.get_jobs(limit, offset)
            result['jobs'] = sorted([j.to_dict() for j in jobs], key=lambda j: j['job_id'])
        return result
