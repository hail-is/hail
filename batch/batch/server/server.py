import asyncio
import concurrent
import time
import logging
import os
import threading
import traceback
import json
import uuid
from shlex import quote as shq

import jinja2
import aiohttp_jinja2
from aiohttp import web
import cerberus
import kubernetes as kube
import requests
import uvloop

from gear.auth import rest_authenticated_users_only, web_authenticated_users_only, \
    new_csrf_token, check_csrf_token

from .blocking_to_async import blocking_to_async
from .log_store import LogStore
from .database import BatchDatabase, BatchBuilder
from .k8s import K8s
from ..globals import complete_states

from .. import schemas


def make_logger():
    fmt = logging.Formatter(
        # NB: no space after levename because WARNING is so long
        '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| %(funcName)s:%(lineno)d | '
        '%(message)s')

    file_handler = logging.FileHandler('batch.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    log = logging.getLogger('batch')
    log.setLevel(logging.INFO)

    logging.basicConfig(handlers=[file_handler, stream_handler], level=logging.INFO)

    return log


log = make_logger()


uvloop.install()


KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
HAIL_POD_NAMESPACE = os.environ.get('HAIL_POD_NAMESPACE', 'batch-pods')
POD_VOLUME_SIZE = os.environ.get('POD_VOLUME_SIZE', '10Mi')
INSTANCE_ID = os.environ.get('HAIL_INSTANCE_ID', uuid.uuid4().hex)

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'HAIL_POD_NAMESPACE {HAIL_POD_NAMESPACE}')
log.info(f'POD_VOLUME_SIZE {POD_VOLUME_SIZE}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')

STORAGE_CLASS_NAME = 'batch'

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()
v1 = kube.client.CoreV1Api()

app = web.Application(client_max_size=None)
routes = web.RouteTableDef()

db = BatchDatabase.create_synchronous('/batch-user-secret/sql-config.json')


def abort(code, reason=None):
    if code == 400:
        raise web.HTTPBadRequest(reason=reason)
    if code == 404:
        raise web.HTTPNotFound(reason=reason)
    raise web.HTTPException(reason=reason)


def jsonify(data):
    return web.json_response(data)


class JobTask:  # pylint: disable=R0903
    @staticmethod
    def from_dict(d):
        name = d['name']
        pod_spec_dict = d['pod_spec_dict']
        return JobTask(name, pod_spec_dict)

    @staticmethod
    def copy_task(task_name, files):
        if files is not None:
            authenticate = 'set -ex; gcloud -q auth activate-service-account --key-file=/gsa-key/privateKeyData'

            def copy_command(src, dst):
                if not dst.startswith('gs://'):
                    mkdirs = f'mkdir -p {shq(os.path.dirname(dst))};'
                else:
                    mkdirs = ""
                return f'{mkdirs} gsutil -m cp -R {shq(src)} {shq(dst)}'

            copies = ' && '.join([copy_command(src, dst) for (src, dst) in files])
            sh_expression = f'{authenticate} && {copies}'
            container = kube.client.V1Container(
                image='google/cloud-sdk:237.0.0-alpine',
                name=task_name,
                command=['/bin/sh', '-c', sh_expression])
            spec = kube.client.V1PodSpec(
                containers=[container],
                restart_policy='Never',
                tolerations=[
                    kube.client.V1Toleration(key='preemptible', value='true')
                ])
            return JobTask.from_spec(task_name, spec)
        return None

    @staticmethod
    def from_spec(name, pod_spec):
        assert pod_spec is not None
        return JobTask(name, v1.api_client.sanitize_for_serialization(pod_spec))

    def __init__(self, name, pod_spec_dict):
        self.pod_spec_dict = pod_spec_dict
        self.name = name

    def to_dict(self):
        return {'name': self.name,
                'pod_spec_dict': self.pod_spec_dict}


class Job:
    async def _create_pvc(self):
        pvc, err = await app['k8s'].create_pvc(
            body=kube.client.V1PersistentVolumeClaim(
                metadata=kube.client.V1ObjectMeta(
                    generate_name=f'batch-{self.batch_id}-job-{self.job_id}-',
                    labels={'app': 'batch-job',
                            'hail.is/batch-instance': INSTANCE_ID}),
                spec=kube.client.V1PersistentVolumeClaimSpec(
                    access_modes=['ReadWriteOnce'],
                    volume_mode='Filesystem',
                    resources=kube.client.V1ResourceRequirements(
                        requests={'storage': self._pvc_size}),
                    storage_class_name=STORAGE_CLASS_NAME)))
        if err is not None:
            log.info(f'persistent volume claim cannot be created for job {self.id} with the following error: {err}')
            if err.status == 403:
                await self.mark_complete(None, failed=True, failure_reason=str(err))
            return None
        pvc_name = pvc.metadata.name
        await db.jobs.update_record(*self.id, pvc_name=pvc_name)
        log.info(f'created pvc name: {pvc_name} for job {self.id}')
        return pvc_name

    # may be called twice with the same _current_task
    async def _create_pod(self):
        assert self._pod_name is None
        assert self._current_task is not None
        assert self.userdata is not None

        volumes = [
            kube.client.V1Volume(
                secret=kube.client.V1SecretVolumeSource(
                    secret_name=self.userdata['gsa_key_secret_name']),
                name='gsa-key')]
        volume_mounts = [
            kube.client.V1VolumeMount(
                mount_path='/gsa-key',
                name='gsa-key')]

        if len(self._tasks) > 1:
            if self._pvc_name is None:
                self._pvc_name = await self._create_pvc()
                if self._pvc_name is None:
                    log.info(f'could not create pod for job {self.id} due to pvc creation failure')
                    return
            volumes.append(kube.client.V1Volume(
                persistent_volume_claim=kube.client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=self._pvc_name),
                name=self._pvc_name))
            volume_mounts.append(kube.client.V1VolumeMount(
                mount_path='/io',
                name=self._pvc_name))

        pod_spec = v1.api_client._ApiClient__deserialize(self._current_task.pod_spec_dict, kube.client.V1PodSpec)
        if pod_spec.volumes is None:
            pod_spec.volumes = []
        pod_spec.volumes.extend(volumes)
        for container in pod_spec.containers:
            if container.volume_mounts is None:
                container.volume_mounts = []
            container.volume_mounts.extend(volume_mounts)

        pod_template = kube.client.V1Pod(
            metadata=kube.client.V1ObjectMeta(
                generate_name='batch-{}-job-{}-{}-'.format(self.batch_id, self.job_id, self._current_task.name),
                labels={'app': 'batch-job',
                        'hail.is/batch-instance': INSTANCE_ID,
                        'uuid': uuid.uuid4().hex
                        }),
            spec=pod_spec)

        pod, err = await app['k8s'].create_pod(body=pod_template)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'pod creation failed for job {self.id} with the following error: {err}')
            return
        self._pod_name = pod.metadata.name
        await db.jobs.update_record(*self.id,
                                    pod_name=self._pod_name)
        log.info('created pod name: {} for job {}, task {}'.format(self._pod_name,
                                                                   self.id,
                                                                   self._current_task.name))

    async def _delete_pvc(self):
        if self._pvc_name is None:
            return

        log.info(f'deleting persistent volume claim {self._pvc_name}')
        err = await app['k8s'].delete_pvc(self._pvc_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'ignoring: could not delete {self._pvc_name} due to {err}')
        await db.jobs.update_record(*self.id, pvc_name=None)
        self._pvc_name = None

    async def _delete_k8s_resources(self):
        await self._delete_pvc()
        if self._pod_name is not None:
            await db.jobs.update_record(*self.id, pod_name=None)
            err = await app['k8s'].delete_pod(name=self._pod_name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring pod deletion failure for job {self.id} due to {err}')
            self._pod_name = None

    async def _read_logs(self):
        async def _read_log(jt):
            log_uri = await db.jobs.get_log_uri(*self.id, jt.name)
            if log_uri is None:
                return None
            log, err = await app['log_store'].read_gs_log_file(log_uri)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read log for {self.id} '
                         f'{jt.name} due to {err}; will still try to load '
                         f'other tasks')
            return jt.name, log

        future_logs = asyncio.gather(*[_read_log(jt) for idx, jt in enumerate(self._tasks)
                                       if idx < self._task_idx])
        logs = {k: v for k, v in await future_logs}

        if self._state == 'Ready':
            if self._pod_name:
                pod_log, err = await app['k8s'].read_pod_log(self._pod_name)
                if err is None:
                    logs[self._current_task.name] = pod_log
                else:
                    traceback.print_tb(err.__traceback__)
                    log.info(f'ignoring: could not read log for {self.id} '
                             f'{self._current_task.name}; will still try to load other tasks')
            return logs
        if self.is_complete():
            return logs
        assert self._state == 'Cancelled' or self._state == 'Pending'
        return None

    async def _mark_job_task_complete(self, task_name, log, exit_code, new_state):
        self.exit_codes[self._task_idx] = exit_code

        self._task_idx += 1
        self._current_task = self._tasks[self._task_idx] if self._task_idx < len(self._tasks) else None

        uri = None
        if log is not None:
            uri, err = await app['log_store'].write_gs_log_file(*self.id, task_name, log)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'job {self.id} task {task_name} will have a missing log due to {err}')

        await db.jobs.update_with_log_ec(*self.id, task_name, uri, exit_code,
                                         task_idx=self._task_idx,
                                         pod_name=None,
                                         duration=self.duration,
                                         state=new_state)

        if self._pod_name is None:
            return

        err = await app['k8s'].delete_pod(self._pod_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'ignoring pod deletion failure for job {self.id} due to {err}')
        self._pod_name = None

    async def _delete_logs(self):
        for idx, jt in enumerate(self._tasks):
            if idx < self._task_idx:
                err = await app['log_store'].delete_gs_log_file(*self.id, jt.name)
                if err is not None:
                    traceback.print_tb(err.__traceback__)
                    log.info(f'could not delete log {idx} due to {err}')

    @staticmethod
    def from_record(record):
        if record is not None:
            tasks = [JobTask.from_dict(item) for item in json.loads(record['tasks'])]
            attributes = json.loads(record['attributes'])
            userdata = json.loads(record['userdata'])

            exit_codes = [record[db.jobs.exit_code_field(t.name)] for t in tasks]
            ec_indices = [idx for idx, ec in enumerate(exit_codes) if ec is not None]
            assert record['task_idx'] == 1 + (ec_indices[-1] if len(ec_indices) else -1)

            return Job(batch_id=record['batch_id'], job_id=record['job_id'], attributes=attributes,
                       callback=record['callback'], userdata=userdata, user=record['user'],
                       always_run=record['always_run'], pvc_name=record['pvc_name'], pod_name=record['pod_name'],
                       exit_codes=exit_codes, duration=record['duration'], tasks=tasks,
                       task_idx=record['task_idx'], state=record['state'], pvc_size=record['pvc_size'],
                       cancelled=record['cancelled'])
        return None

    @staticmethod
    async def from_db(batch_id, job_id, user):
        jobs = await Job.from_db_multiple(batch_id, job_id, user)
        if len(jobs) == 1:
            return jobs[0]
        return None

    @staticmethod
    async def from_db_multiple(batch_id, job_ids, user):
        records = await db.jobs.get_undeleted_records(batch_id, job_ids, user)
        jobs = [Job.from_record(record) for record in records]
        return jobs

    @staticmethod
    def create_job(batch_builder, pod_spec, batch_id, job_id, attributes, callback,
                   parent_ids, input_files, output_files, userdata, always_run, pvc_size):
        pvc_name = None
        pod_name = None
        duration = 0
        task_idx = 0
        state = 'Pending'
        cancelled = False
        user = userdata['username']

        tasks = [JobTask.copy_task('input', input_files),
                 JobTask.from_spec('main', pod_spec),
                 JobTask.copy_task('output', output_files)]

        tasks = [t for t in tasks if t is not None]
        exit_codes = [None for _ in tasks]

        batch_builder.create_job(
            batch_id=batch_id,
            job_id=job_id,
            state=state,
            pod_name=pod_name,
            pvc_name=pvc_name,
            pvc_size=pvc_size,
            callback=callback,
            attributes=json.dumps(attributes),
            tasks=json.dumps([jt.to_dict() for jt in tasks]),
            task_idx=task_idx,
            always_run=always_run,
            duration=duration)

        for parent in parent_ids:
            batch_builder.create_job_parent(
                batch_id=batch_id,
                job_id=job_id,
                parent_id=parent)

        job = Job(batch_id=batch_id, job_id=job_id, attributes=attributes, callback=callback,
                  userdata=userdata, user=user, always_run=always_run, pvc_name=pvc_name,
                  pod_name=pod_name, exit_codes=exit_codes, duration=duration, tasks=tasks,
                  task_idx=task_idx, state=state, pvc_size=pvc_size, cancelled=cancelled)

        return job

    def __init__(self, batch_id, job_id, attributes, callback, userdata, user, always_run,
                 pvc_name, pod_name, exit_codes, duration, tasks, task_idx, state,
                 pvc_size, cancelled):
        self.batch_id = batch_id
        self.job_id = job_id
        self.id = (batch_id, job_id)

        self.attributes = attributes
        self.callback = callback
        self.always_run = always_run
        self.userdata = userdata
        self.user = user
        self.exit_codes = exit_codes
        self.duration = duration

        self._pvc_name = pvc_name
        self._pvc_size = pvc_size
        self._pod_name = pod_name
        self._tasks = tasks
        self._task_idx = task_idx
        self._current_task = tasks[task_idx] if task_idx < len(tasks) else None
        self._state = state
        self._cancelled = cancelled

    async def run(self):
        parents = await db.jobs.get_parents(*self.id)
        if not parents:
            await self.set_state('Ready')
            await self._create_pod()
        else:
            await self.refresh_parents_and_maybe_create()

    async def refresh_parents_and_maybe_create(self):
        for record in await db.jobs.get_parents(*self.id):
            parent_job = Job.from_record(record)
            assert parent_job.batch_id == self.batch_id
            await self.parent_new_state(parent_job._state, *parent_job.id)

    async def set_state(self, new_state):
        if self._state != new_state:
            await db.jobs.update_record(*self.id, state=new_state)
            log.info('job {} changed state: {} -> {}'.format(
                self.id,
                self._state,
                new_state))
            self._state = new_state
            await self.notify_children(new_state)

    async def notify_children(self, new_state):
        children = [Job.from_record(record) for record in await db.jobs.get_children(*self.id)]
        for child in children:
            if child:
                await child.parent_new_state(new_state, *self.id)
            else:
                log.info(f'missing child: {child.id}')

    async def parent_new_state(self, new_state, parent_batch_id, parent_job_id):
        assert parent_batch_id == self.batch_id
        assert await db.jobs_parents.has_record(*self.id, parent_job_id)
        if new_state in complete_states:
            await self.create_if_ready()

    async def create_if_ready(self):
        incomplete_parent_ids = await db.jobs.get_incomplete_parents(*self.id)
        if self._state == 'Pending' and not incomplete_parent_ids:
            parents = [Job.from_record(record) for record in await db.jobs.get_parents(*self.id)]
            if (self.always_run or
                (not self._cancelled and all(p.is_successful() for p in parents))):
                log.info(f'all parents complete for {self.id},'
                         f' creating pod')
                await self.set_state('Ready')
                await self._create_pod()
            else:
                log.info(f'parents deleted, cancelled, or failed: cancelling {self.id}')
                await self.set_state('Cancelled')

    async def cancel(self):
        self._cancelled = True
        if self.is_complete() or self._state == 'Pending':
            return
        else:
            assert self._state == 'Ready', self._state
            if not self.always_run:
                await self.set_state('Cancelled')  # must call before deleting resources to prevent race conditions
                await self._delete_k8s_resources()

    def is_complete(self):
        return self._state in complete_states

    def is_successful(self):
        return self._state == 'Success'

    async def mark_unscheduled(self):
        if self._pod_name:
            await db.jobs.update_record(*self.id, pod_name=None)
            err = await app['k8s'].delete_pod(self._pod_name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring pod deletion failure for job {self.id} due to {err}')
            self._pod_name = None
        await self.set_state('Ready')
        await self._create_pod()

    async def mark_complete(self, pod, failed=False, failure_reason=None):
        if pod is not None:
            assert pod.metadata.name == self._pod_name

        task_name = self._current_task.name

        if failed:
            exit_code = 999  # FIXME hack
            pod_log = failure_reason
        else:
            terminated = pod.status.container_statuses[0].state.terminated
            exit_code = terminated.exit_code
            if terminated.finished_at is not None and terminated.started_at is not None:
                duration = (terminated.finished_at - terminated.started_at).total_seconds()
                if self.duration is not None:
                    self.duration += duration
            else:
                log.warning(f'job {self.id} has pod {pod.metadata.name} which is '
                            f'terminated but has no timing information. {pod}')
                self.duration = None
            pod_log, err = await app['k8s'].read_pod_log(pod.metadata.name)
            if err:
                traceback.print_tb(err.__traceback__)
                log.info(f'no logs for {pod.metadata.name} due to previous error, rescheduling pod')
                await self.mark_unscheduled()
                return

        has_next_task = self._task_idx + 1 < len(self._tasks)

        if exit_code == 0:
            if has_next_task:
                new_state = 'Ready'
            else:
                new_state = 'Success'
        elif exit_code == 999:
            new_state = 'Error'
        else:
            new_state = 'Failed'

        await self._mark_job_task_complete(task_name, pod_log, exit_code, new_state)
        
        if exit_code == 0:
            if has_next_task:
                await self._create_pod()
                return

        await self._delete_pvc()
        await self.set_state(new_state)

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
            batch = await Batch.from_db(self.batch_id, self.user)
            if batch is not None:
                await batch.mark_job_complete(self)

    def to_dict(self):
        state = self._state
        if state == 'Ready' and self._pod_name is not None:
            state = 'Running'

        result = {
            'batch_id': self.batch_id,
            'job_id': self.job_id,
            'state': state
        }
        if self.is_complete():
            result['exit_code'] = {t.name: ec for ec, t in zip(self.exit_codes, self._tasks)}
            result['duration'] = self.duration

        if self.attributes:
            result['attributes'] = self.attributes
        return result


async def create_job(batch_builder, batch_id, userdata, parameters):  # pylint: disable=R0912
    user = userdata['username']

    pod_spec = v1.api_client._ApiClient__deserialize(
        parameters['spec'], kube.client.V1PodSpec)

    job_id = parameters.get('job_id')
    has_record = await db.jobs.has_record(batch_id, job_id)
    if has_record:
        abort(400, f'invalid request: batch {batch_id} already has a job_id={job_id}')

    parent_ids = parameters.get('parent_ids', [])
    input_files = parameters.get('input_files')
    output_files = parameters.get('output_files')
    pvc_size = parameters.get('pvc_size', POD_VOLUME_SIZE)
    always_run = parameters.get('always_run', False)

    if len(pod_spec.containers) != 1:
        abort(400, f'only one container allowed in pod_spec {pod_spec}')

    if pod_spec.containers[0].name != 'main':
        abort(400, f'container name must be "main" was {pod_spec.containers[0].name}')

    if not pod_spec.containers[0].resources:
        pod_spec.containers[0].resources = kube.client.V1ResourceRequirements()
    if not pod_spec.containers[0].resources.requests:
        pod_spec.containers[0].resources.requests = {}
    if 'cpu' not in pod_spec.containers[0].resources.requests:
        pod_spec.containers[0].resources.requests['cpu'] = '100m'
    if 'memory' not in pod_spec.containers[0].resources.requests:
        pod_spec.containers[0].resources.requests['memory'] = '500M'

    if not pod_spec.tolerations:
        pod_spec.tolerations = []
    pod_spec.tolerations.append(kube.client.V1Toleration(key='preemptible', value='true'))

    job = Job.create_job(
        batch_builder,
        batch_id=batch_id,
        job_id=job_id,
        pod_spec=pod_spec,
        attributes=parameters.get('attributes'),
        callback=parameters.get('callback'),
        parent_ids=parent_ids,
        input_files=input_files,
        output_files=output_files,
        userdata=userdata,
        always_run=always_run,
        pvc_size=pvc_size)
    return job


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return jsonify({})


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
@rest_authenticated_users_only
async def get_job(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']

    job = await Job.from_db(batch_id, job_id, user)
    if not job:
        abort(404)
    return jsonify(job.to_dict())


async def _get_job_log(batch_id, job_id, user):
    job = await Job.from_db(batch_id, job_id, user)
    if not job:
        abort(404)

    job_log = await job._read_logs()
    if job_log:
        return job_log
    abort(404)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
@rest_authenticated_users_only
async def get_job_log(request, userdata):  # pylint: disable=R1710
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    job_log = await _get_job_log(batch_id, job_id, user)
    return jsonify(job_log)


class Batch:
    @staticmethod
    def from_record(record, deleted=False):
        if record is not None:
            if not deleted:
                assert not record['deleted']
            attributes = json.loads(record['attributes'])
            userdata = json.loads(record['userdata'])

            if record['n_failed'] > 0:
                state = 'failure'
            elif record['n_cancelled'] > 0:
                state = 'cancelled'
            elif record['n_succeeded'] == record['n_jobs']:
                state = 'success'
            else:
                state = 'running'

            complete = record['n_completed'] == record['n_jobs']

            return Batch(id=record['id'],
                         attributes=attributes,
                         callback=record['callback'],
                         userdata=userdata,
                         user=record['user'],
                         state=state,
                         complete=complete,
                         deleted=record['deleted'],
                         cancelled=record['cancelled'])
        return None

    @staticmethod
    async def from_db(ids, user):
        batches = await Batch.from_db_multiple(ids, user)
        if len(batches) == 1:
            return batches[0]
        return None

    @staticmethod
    async def from_db_multiple(ids, user):
        records = await db.batch.get_undeleted_records(ids, user)
        batches = [Batch.from_record(record) for record in records]
        return batches

    @staticmethod
    async def create_batch(batch_builder, attributes, callback, userdata):
        user = userdata['username']

        id = await batch_builder.create_batch(
            attributes=json.dumps(attributes),
            callback=callback,
            userdata=json.dumps(userdata),
            user=user,
            deleted=False,
            cancelled=False)

        batch = Batch(id=id, attributes=attributes, callback=callback,
                      userdata=userdata, user=user, state='running',
                      complete=False, deleted=False, cancelled=False)
        return batch

    def __init__(self, id, attributes, callback, userdata, user,
                 state, complete, deleted, cancelled):
        self.id = id
        self.attributes = attributes
        self.callback = callback
        self.userdata = userdata
        self.user = user
        self.state = state
        self.complete = complete
        self.deleted = deleted
        self.cancelled = cancelled

    async def get_jobs(self):
        return [Job.from_record(record) for record in await db.jobs.get_records_by_batch(self.id)]

    async def cancel(self):
        await db.batch.update_record(self.id, cancelled=True)
        jobs = await self.get_jobs()
        for j in jobs:
            await j.cancel()
        log.info(f'batch {self.id} cancelled')

    async def mark_deleted(self):
        await self.cancel()
        await db.batch.update_record(self.id,
                                     deleted=True)
        self.deleted = True
        log.info(f'batch {self.id} marked for deletion')

    async def delete(self):
        for j in await self.get_jobs():
            # Job deleted from database when batch is deleted with delete cascade
            await j._delete_logs()
        await db.batch.delete_record(self.id)
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

    async def to_dict(self, include_jobs=False):
        result = {
            'id': self.id,
            'state': self.state,
            'complete': self.complete
        }
        if self.attributes:
            result['attributes'] = self.attributes
        if include_jobs:
            jobs = await self.get_jobs()
            result['jobs'] = sorted([j.to_dict() for j in jobs], key=lambda j: j['job_id'])
        return result


async def _get_batches_list(params, user):
    batches = [Batch.from_record(record)
               for record in await db.batch.get_records_where({'user': user, 'deleted': False})]

    for name, value in params.items():
        if name == 'complete':
            if value not in ('0', '1'):
                abort(400, f'invalid complete value, expected 0 or 1, got {value}')
            c = value == '1'
            batches = [batch for batch in batches if batch.is_complete() == c]
        elif name == 'success':
            if value not in ('0', '1'):
                abort(400, f'invalid success value, expected 0 or 1, got {value}')
            s = value == '1'
            batches = [batch for batch in batches if batch.is_successful() == s]
        else:
            if not name.startswith('a:'):
                abort(400, f'unknown query parameter {name}')
            k = name[2:]
            batches = [batch for batch in batches
                       if batch.attributes and k in batch.attributes and batch.attributes[k] == value]

    return [await batch.to_dict(include_jobs=False) for batch in batches]


@routes.get('/api/v1alpha/batches')
@rest_authenticated_users_only
async def get_batches_list(request, userdata):
    params = request.query
    user = userdata['username']
    return jsonify(await _get_batches_list(params, user))


@routes.post('/api/v1alpha/batches/create')
@rest_authenticated_users_only
async def create_batch(request, userdata):
    reader = await request.multipart()

    part = await reader.next()
    batch_parameters = await part.json()

    validator = cerberus.Validator(schemas.batch_schema)
    if not validator.validate(batch_parameters):
        abort(400, 'invalid request: {}'.format(validator.errors))

    n_jobs = batch_parameters['n_jobs']
    start_time = time.time()
    batch_builder = BatchBuilder(db, n_jobs, log)

    try:
        batch = await Batch.create_batch(
            batch_builder,
            attributes=batch_parameters.get('attributes'),
            callback=batch_parameters.get('callback'),
            userdata=userdata)

        jobs = []
        while True:
            part = await reader.next()
            if part is None:
                break

            jobs_part_data = await part.json()

            validator = cerberus.Validator(schemas.job_array_schema)
            if not validator.validate(jobs_part_data):
                abort(400, 'invalid request: {}'.format(validator.errors))

            for job_params in jobs_part_data['jobs']:
                job = await create_job(batch_builder, batch.id, userdata, job_params)
                jobs.append(job)

        success = await batch_builder.commit()
        if not success:
            abort(400, f'batch creation failed')
    finally:
        await batch_builder.close()

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)
    log.info(f'created batch {batch.id} with {n_jobs} jobs in {elapsed_time} seconds')

    for j in jobs:
        await j.run()

    log.info(f'started all jobs in batch {batch.id}')

    return jsonify(await batch.to_dict(include_jobs=False))


async def _get_batch(batch_id, user):
    batch = await Batch.from_db(batch_id, user)
    if not batch:
        abort(404)
    return await batch.to_dict(include_jobs=True)


async def _cancel_batch(batch_id, user):
    batch = await Batch.from_db(batch_id, user)
    if not batch:
        abort(404)
    await batch.cancel()


@routes.get('/api/v1alpha/batches/{batch_id}')
@rest_authenticated_users_only
async def get_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    return jsonify(await _get_batch(batch_id, user))


@routes.patch('/api/v1alpha/batches/{batch_id}/cancel')
@rest_authenticated_users_only
async def cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(batch_id, user)
    return jsonify({})


@routes.delete('/api/v1alpha/batches/{batch_id}')
@rest_authenticated_users_only
async def delete_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']

    batch = await Batch.from_db(batch_id, user)
    if not batch:
        abort(404)
    await batch.mark_deleted()
    return jsonify({})


@routes.get('/batches/{batch_id}')
@aiohttp_jinja2.template('batch.html')
@web_authenticated_users_only
async def ui_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await _get_batch(batch_id, user)
    return {'batch': batch}


@routes.post('/batches/{batch_id}/cancel')
@aiohttp_jinja2.template('batches.html')
@check_csrf_token
@web_authenticated_users_only
async def ui_cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(batch_id, user)
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


@routes.get('/batches', name='batches')
@web_authenticated_users_only
async def ui_batches(request, userdata):
    params = request.query
    user = userdata['username']
    batches = await _get_batches_list(params, user)
    token = new_csrf_token()
    context = {'batch_list': batches, 'token': token}

    response = aiohttp_jinja2.render_template('batches.html',
                                              request,
                                              context)
    response.set_cookie('_csrf', token, secure=True, httponly=True)
    return response


@routes.get('/batches/{batch_id}/jobs/{job_id}/log')
@aiohttp_jinja2.template('job_log.html')
@web_authenticated_users_only
async def ui_get_job_log(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    job_log = await _get_job_log(batch_id, job_id, user)
    return {'batch_id': batch_id, 'job_id': job_id, 'job_log': job_log}


@routes.get('/')
@web_authenticated_users_only
async def batch_id(request, userdata):
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def update_job_with_pod(job, pod):
    log.info(f'update job {job.id} with pod {pod.metadata.name if pod else "None"}')
    if not pod or (pod.status and pod.status.reason == 'Evicted'):
        log.info(f'job {job.id} mark unscheduled')
        await job.mark_unscheduled()
    elif (pod
          and pod.status
          and pod.status.container_statuses):
        assert len(pod.status.container_statuses) == 1
        container_status = pod.status.container_statuses[0]
        assert container_status.name in ['input', 'main', 'output']

        if container_status.state:
            if container_status.state.terminated:
                log.info(f'job {job.id} mark complete')
                await job.mark_complete(pod)
            elif (container_status.state.waiting
                  and container_status.state.waiting.reason == 'ImagePullBackOff'):
                log.info(f'job {job.id} mark failed: ImagePullBackOff')
                await job.mark_complete(pod, failed=True, failure_reason=container_status.state.waiting.reason)


class DeblockedIterator:
    def __init__(self, it):
        self.it = it

    def __aiter__(self):
        return self

    def __anext__(self):
        return blocking_to_async(app['blocking_pool'], self.it.__next__)


async def pod_changed(pod):
    job = Job.from_record(await db.jobs.get_record_by_pod(pod.metadata.name))

    if job and not job.is_complete():
        await update_job_with_pod(job, pod)


async def kube_event_loop():
    while True:
        try:
            stream = kube.watch.Watch().stream(
                v1.list_namespaced_pod,
                HAIL_POD_NAMESPACE,
                label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}')
            async for event in DeblockedIterator(stream):
                await pod_changed(event['object'])
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'k8s event stream failed due to: {exc}')
        await asyncio.sleep(5)


async def refresh_k8s_pods():
    # if we do this after we get pods, we will pick up jobs created
    # while listing pods and unnecessarily restart them
    pod_jobs = [Job.from_record(record) for record in await db.jobs.get_records_where({'pod_name': 'NOT NULL'})]

    pods, err = await app['k8s'].list_pods(
        label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}')
    if err is not None:
        traceback.print_tb(err.__traceback__)
        log.info('could not refresh pods due to {err}, will try again later')
        return

    log.info(f'k8s had {len(pods.items)} pods')

    seen_pods = set()
    for pod in pods.items:
        pod_name = pod.metadata.name
        seen_pods.add(pod_name)

        job = Job.from_record(await db.jobs.get_record_by_pod(pod_name))
        if job and not job.is_complete():
            await update_job_with_pod(job, pod)

    log.info('starting pods not seen in k8s')

    for job in pod_jobs:
        pod_name = job._pod_name
        if pod_name not in seen_pods:
            log.info(f'restarting job {job.id}')
            await update_job_with_pod(job, None)


async def refresh_k8s_pvc():
    pvcs, err = await app['k8s'].list_pvcs(
        label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}')
    if err is not None:
        traceback.print_tb(err.__traceback__)
        log.info('could not refresh pvcs due to {err}, will try again later')
        return

    log.info(f'k8s had {len(pvcs.items)} pvcs')

    seen_pvcs = set()
    for record in await db.jobs.get_records_where({'pvc_name': 'NOT NULL'}):
        job = Job.from_record(record)
        assert job._pvc_name
        seen_pvcs.add(job._pvc_name)

    for pvc in pvcs.items:
        if pvc.metadata.name not in seen_pvcs:
            log.info(f'deleting orphaned pvc {pvc.metadata.name}')
            err = await app['k8s'].delete_pvc(pvc.metadata.name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info('could not delete {pvc.metadata.name} due to {err}')


async def create_pods_if_ready():
    await asyncio.sleep(30)
    while True:
        for record in await db.jobs.get_records_where({'state': 'Ready',
                                                       'pod_name': None}):
            job = Job.from_record(record)
            try:
                await job._create_pod()
            except Exception as exc:  # pylint: disable=W0703
                log.exception(f'Could not create pod for job {job.id} due to exception: {exc}')
        await asyncio.sleep(30)


async def refresh_k8s_state():  # pylint: disable=W0613
    log.info('started k8s state refresh')
    await refresh_k8s_pods()
    await refresh_k8s_pvc()
    log.info('k8s state refresh complete')


async def polling_event_loop():
    await asyncio.sleep(1)
    while True:
        try:
            await refresh_k8s_state()
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'Could not poll due to exception: {exc}')
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


async def db_cleanup_event_loop():
    await asyncio.sleep(60)
    while True:
        try:
            for record in await db.batch.get_finished_deleted_records():
                batch = Batch.from_record(record, deleted=True)
                await batch.delete()
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'Could not delete batches due to exception: {exc}')
        await asyncio.sleep(60)


def serve(port=5000):
    batch_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(os.path.join(batch_root, 'templates')))
    routes.static('/static', os.path.join(batch_root, 'static'))
    app.add_routes(routes)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        app['blocking_pool'] = pool
        app['k8s'] = K8s(pool, KUBERNETES_TIMEOUT_IN_SECONDS, HAIL_POD_NAMESPACE, v1, log)
        app['log_store'] = LogStore(pool, INSTANCE_ID, log)
        asyncio.ensure_future(polling_event_loop())
        asyncio.ensure_future(kube_event_loop())
        asyncio.ensure_future(db_cleanup_event_loop())
        asyncio.ensure_future(create_pods_if_ready())
        web.run_app(app, host='0.0.0.0', port=port)
