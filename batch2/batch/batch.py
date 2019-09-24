import asyncio
import concurrent
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
import prometheus_client as pc
from prometheus_async.aio import time as prom_async_time
from prometheus_async.aio.web import server_stats

from hailtop.utils import unzip, blocking_to_async
from hailtop.auth import async_get_userinfo
from hailtop.config import get_deploy_config
from gear import setup_aiohttp_session, \
    rest_authenticated_users_only, web_authenticated_users_only, \
    new_csrf_token, check_csrf_token
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, base_context

import uvloop
uvloop.install()

from .blocking_to_async import blocking_to_async
from .log_store import LogStore
from .database import JobsBuilder
from .datetime_json import JSON_ENCODER
from .globals import states, complete_states, valid_state_transitions, tasks, get_db
from .batch_configuration import KUBERNETES_TIMEOUT_IN_SECONDS, REFRESH_INTERVAL_IN_SECONDS, \
    POD_VOLUME_SIZE, INSTANCE_ID, BATCH_IMAGE, BATCH_NAMESPACE
from .driver import Driver
from .k8s import K8s

from . import schemas

log = logging.getLogger('batch')

REQUEST_TIME = pc.Summary('batch_request_latency_seconds', 'Batch request latency in seconds', ['endpoint', 'verb'])
REQUEST_TIME_GET_JOB = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id', verb="GET")
REQUEST_TIME_GET_JOB_LOG = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id/log', verb="GET")
REQUEST_TIME_GET_POD_STATUS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/job_id/pod_status', verb="GET")
REQUEST_TIME_GET_BATCHES = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches', verb="GET")
REQUEST_TIME_POST_CREATE_JOBS = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/jobs/create', verb="POST")
REQUEST_TIME_POST_CREATE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/create', verb='POST')
REQUEST_TIME_POST_GET_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id', verb='GET')
REQUEST_TIME_PATCH_CANCEL_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/cancel', verb="PATCH")
REQUEST_TIME_PATCH_CLOSE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id/close', verb="PATCH")
REQUEST_TIME_DELETE_BATCH = REQUEST_TIME.labels(endpoint='/api/v1alpha/batches/batch_id', verb="DELETE")
REQUEST_TIME_GET_BATCH_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id', verb='GET')
REQUEST_TIME_POST_CANCEL_BATCH_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/cancel', verb='POST')
REQUEST_TIME_GET_BATCHES_UI = REQUEST_TIME.labels(endpoint='/batches', verb='GET')
REQUEST_TIME_GET_LOGS_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/jobs/job_id/log', verb="GET")
REQUEST_TIME_GET_POD_STATUS_UI = REQUEST_TIME.labels(endpoint='/batches/batch_id/jobs/job_id/pod_status', verb="GET")

POD_EVICTIONS = pc.Counter('batch_pod_evictions', 'Count of batch pod evictions')
READ_POD_LOG_FAILURES = pc.Counter('batch_read_pod_log_failures', 'Count of batch read_pod_log failures')

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'POD_VOLUME_SIZE {POD_VOLUME_SIZE}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')
log.info(f'BATCH_IMAGE = {BATCH_IMAGE}')

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()
v1 = kube.client.CoreV1Api()

app = web.Application(client_max_size=None)
setup_aiohttp_session(app)

routes = web.RouteTableDef()

db = get_db()
deploy_config = get_deploy_config()


def abort(code, reason=None):
    if code == 400:
        raise web.HTTPBadRequest(reason=reason)
    if code == 404:
        raise web.HTTPNotFound(reason=reason)
    raise web.HTTPException(reason=reason)


def jsonify(data):
    return web.json_response(data)


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

        pod_spec = v1.api_client._ApiClient__deserialize(self._pod_spec, kube.client.V1PodSpec)
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

        err = await app['driver'].create_pod(spec=pod_template.to_dict(),
                                             output_directory=self.directory)
        if err is not None:
            if err.status == 409:  # FIXME: Error from driver is not 409 right now
                log.info(f'pod already exists for job {self.id}')
                return
            # traceback.print_tb(err.__traceback__)
            log.info(f'pod creation failed for job {self.id} '
                     f'with the following error: {err}')

    async def _delete_pod(self):
        err = await app['driver'].delete_pod(name=self._pod_name)
        if err is not None:
            # traceback.print_tb(err.__traceback__)
            log.info(f'ignoring pod deletion failure for job {self.id} due to {err}')

    async def _read_logs(self):
        if self._state in ('Pending', 'Cancelled'):
            return None

        async def _read_log_from_gcs(task_name):
            pod_log, err = await app['log_store'].read_gs_file(LogStore.container_log_path(self.directory, task_name))
            if err is not None:
                # traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read log for {self.id}, {task_name} '
                         f'due to {err}')
            return task_name, pod_log

        async def _read_log_from_worker(task_name):
            pod_log, err = await app['driver'].read_pod_log(self._pod_name, container=task_name)
            if err is not None:
                # traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read log for {self.id}, {task_name} '
                         f'due to {err}; will still try to load other containers')
            return task_name, pod_log

        if self._state == 'Running':
            future_logs = asyncio.gather(*[_read_log_from_worker(task) for task in tasks])
            return {k: v for k, v in await future_logs}
        else:
            assert self._state in ('Error', 'Failed', 'Success')
            future_logs = asyncio.gather(*[_read_log_from_gcs(task) for task in tasks])
            return {k: v for k, v in await future_logs}

    async def _read_pod_status(self):
        if self._state in ('Pending', 'Cancelled'):
            return None

        async def _read_status_from_gcs(task_name):
            status, err = await app['log_store'].read_gs_file(LogStore.container_status_path(self.directory, task_name))
            if err is not None:
                # traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read container status for {self.id} '
                         f'due to {err}')
                return None
            return task_name, status

        async def _read_status_from_worker(task_name):
            status, err = await app['driver'].read_container_status(self._pod_name, container=task_name)
            if err is not None:
                # traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read container status for {self.id} '
                         f'due to {err}; will still try to load other containers')
            return task_name, status

        if self._state == 'Running':
            future_statuses = asyncio.gather(*[_read_status_from_worker(task) for task in tasks])
            return {k: v for k, v in await future_statuses}
        else:
            assert self._state in ('Error', 'Failed', 'Success')
            future_statuses = asyncio.gather(*[_read_status_from_gcs(task) for task in tasks])
            return {k: v for k, v in await future_statuses}

    async def _delete_gs_files(self):
        errs = await app['log_store'].delete_gs_files(self.directory)
        for file, err in errs:
            if err is not None:
                # traceback.print_tb(err.__traceback__)
                log.info(f'could not delete {self.directory}/{file} for job {self.id} due to {err}')

    @staticmethod
    def from_record(record):
        if record is not None:
            attributes = json.loads(record['attributes'])
            userdata = json.loads(record['userdata'])
            pod_spec = json.loads(record['pod_spec'])
            input_files = json.loads(record['input_files'])
            output_files = json.loads(record['output_files'])
            exit_codes = json.loads(record['exit_codes'])
            durations = json.loads(record['durations'])
            messages = json.loads(record['messages'])

            return Job(batch_id=record['batch_id'], job_id=record['job_id'], attributes=attributes,
                       callback=record['callback'], userdata=userdata, user=record['user'],
                       always_run=record['always_run'], exit_codes=exit_codes, messages=messages,
                       durations=durations, state=record['state'], pvc_size=record['pvc_size'],
                       cancelled=record['cancelled'], directory=record['directory'],
                       token=record['token'], pod_spec=pod_spec, input_files=input_files,
                       output_files=output_files)

        return None

    @staticmethod
    async def from_pod(pod):
        if pod.metadata.labels is None:
            return None
        if not {'batch_id', 'job_id', 'user'}.issubset(set(pod.metadata.labels)):
            return None
        batch_id = pod.metadata.labels['batch_id']
        job_id = pod.metadata.labels['job_id']
        user = pod.metadata.labels['user']
        return await Job.from_db(batch_id, job_id, user)

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
    def create_job(jobs_builder, pod_spec, batch_id, job_id, attributes, callback,
                   parent_ids, input_files, output_files, userdata, always_run,
                   pvc_size, state):
        cancelled = False
        user = userdata['username']
        token = uuid.uuid4().hex[:6]

        exit_codes = [None for _ in tasks]
        durations = [None for _ in tasks]
        messages = [None for _ in tasks]
        directory = app['log_store'].gs_job_output_directory(batch_id, job_id, token)
        pod_spec = v1.api_client.sanitize_for_serialization(pod_spec)

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

        job = Job(batch_id=batch_id, job_id=job_id, attributes=attributes, callback=callback,
                  userdata=userdata, user=user, always_run=always_run, exit_codes=exit_codes,
                  messages=messages, durations=durations, state=state, pvc_size=pvc_size,
                  cancelled=cancelled, directory=directory, token=token,
                  pod_spec=pod_spec, input_files=input_files, output_files=output_files)

        return job

    def __init__(self, batch_id, job_id, attributes, callback, userdata, user, always_run,
                 exit_codes, messages, durations, state, pvc_size, cancelled, directory,
                 token, pod_spec, input_files, output_files):
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
        for record in await db.jobs.get_parents(*self.id):
            parent_job = Job.from_record(record)
            assert parent_job.batch_id == self.batch_id
            await self.parent_new_state(parent_job._state, *parent_job.id)

    async def set_state(self, new_state):
        assert new_state in valid_state_transitions[self._state], f'{self._state} -> {new_state}'
        if self._state != new_state:
            n_updated = await db.jobs.update_record(*self.id, compare_items={'state': self._state}, state=new_state)
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

        children = [Job.from_record(record) for record in await db.jobs.get_children(*self.id)]
        for child in children:
            await child.parent_new_state(new_state, *self.id)

    async def parent_new_state(self, new_state, parent_batch_id, parent_job_id):
        del parent_job_id
        assert parent_batch_id == self.batch_id
        if new_state in complete_states:
            await self.create_if_ready()

    async def create_if_ready(self):
        incomplete_parent_ids = await db.jobs.get_incomplete_parents(*self.id)
        if self._state == 'Pending' and not incomplete_parent_ids:
            await self.set_state('Running')
            parents = [Job.from_record(record) for record in await db.jobs.get_parents(*self.id)]
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
        updated_job = await Job.from_db(*self.id, self.user)
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

        n_updated = await db.jobs.update_record(*self.id,
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
            batch = await Batch.from_db(self.batch_id, self.user)
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


def create_job(jobs_builder, batch_id, userdata, parameters):  # pylint: disable=R0912
    pod_spec = v1.api_client._ApiClient__deserialize(
        parameters['spec'], kube.client.V1PodSpec)

    job_id = parameters.get('job_id')
    parent_ids = parameters.get('parent_ids', [])
    input_files = parameters.get('input_files')
    output_files = parameters.get('output_files')
    pvc_size = parameters.get('pvc_size')
    if pvc_size is None and (input_files or output_files):
        pvc_size = POD_VOLUME_SIZE
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

    state = 'Running' if len(parent_ids) == 0 else 'Pending'

    job = Job.create_job(
        jobs_builder,
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
        pvc_size=pvc_size,
        state=state)
    return job


@routes.get('/healthcheck')
async def get_healthcheck(request):  # pylint: disable=W0613
    return web.Response()


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
@prom_async_time(REQUEST_TIME_GET_JOB)
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


async def _get_pod_status(batch_id, job_id, user):
    job = await Job.from_db(batch_id, job_id, user)
    if not job:
        abort(404)

    pod_statuses = await job._read_pod_status()
    if pod_statuses:
        return JSON_ENCODER.encode(pod_statuses)
    abort(404)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
@prom_async_time(REQUEST_TIME_GET_JOB_LOG)
@rest_authenticated_users_only
async def get_job_log(request, userdata):  # pylint: disable=R1710
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    job_log = await _get_job_log(batch_id, job_id, user)
    return jsonify(job_log)


@routes.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/pod_status')
@prom_async_time(REQUEST_TIME_GET_POD_STATUS)
@rest_authenticated_users_only
async def get_pod_status(request, userdata):  # pylint: disable=R1710
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    pod_spec = await _get_pod_status(batch_id, job_id, user)
    return jsonify(pod_spec)


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
            elif record['closed'] and record['n_succeeded'] == record['n_jobs']:
                state = 'success'
            else:
                state = 'running'

            complete = record['closed'] and record['n_completed'] == record['n_jobs']

            return Batch(id=record['id'],
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
    async def create_batch(attributes, callback, userdata):
        user = userdata['username']

        id = await db.batch.new_record(
            attributes=json.dumps(attributes),
            callback=callback,
            userdata=json.dumps(userdata),
            user=user,
            deleted=False,
            cancelled=False,
            closed=False)

        batch = Batch(id=id, attributes=attributes, callback=callback,
                      userdata=userdata, user=user, state='running',
                      complete=False, deleted=False, cancelled=False,
                      closed=False)

        if attributes is not None:
            items = [{'batch_id': id, 'key': k, 'value': v} for k, v in attributes.items()]
            success = await db.batch_attributes.new_records(items)
            if not success:
                await batch.delete()
                return

        return batch

    def __init__(self, id, attributes, callback, userdata, user,
                 state, complete, deleted, cancelled, closed):
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
        return [Job.from_record(record) for record in await db.jobs.get_records_by_batch(self.id, limit, offset)]

    async def cancel(self):
        await db.batch.update_record(self.id, cancelled=True, closed=True)
        self.cancelled = True
        self.closed = True
        for j in await self.get_jobs():
            await j.cancel()
        log.info(f'batch {self.id} cancelled')

    async def _close_jobs(self):
        for j in await self.get_jobs():
            if j._state == 'Running':
                await j._create_pod()

    async def close(self):
        await db.batch.update_record(self.id, closed=True)
        self.closed = True
        asyncio.ensure_future(self._close_jobs())

    async def mark_deleted(self):
        await self.cancel()
        await db.batch.update_record(self.id,
                                     deleted=True)
        self.deleted = True
        self.closed = True
        log.info(f'batch {self.id} marked for deletion')

    async def delete(self):
        for j in await self.get_jobs():
            # Job deleted from database when batch is deleted with delete cascade
            await j._delete_gs_files()
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


async def _get_batches_list(params, user):
    complete = params.get('complete')
    if complete:
        complete = complete == '1'
    success = params.get('success')
    if success:
        success = success == '1'
    attributes = {}
    for k, v in params.items():
        if k in ('complete', 'success'):  # params does not support deletion
            continue
        if not k.startswith('a:'):
            abort(400, f'unknown query parameter {k}')
        attributes[k[2:]] = v

    records = await db.batch.find_records(user=user,
                                          complete=complete,
                                          success=success,
                                          deleted=False,
                                          attributes=attributes)

    return [await Batch.from_record(batch).to_dict(include_jobs=False)
            for batch in records]


@routes.get('/api/v1alpha/batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES)
@rest_authenticated_users_only
async def get_batches_list(request, userdata):
    params = request.query
    user = userdata['username']
    return jsonify(await _get_batches_list(params, user))


@routes.post('/api/v1alpha/batches/{batch_id}/jobs/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_JOBS)
@rest_authenticated_users_only
async def create_jobs(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await Batch.from_db(batch_id, user)
    if not batch:
        abort(404)
    if batch.closed:
        abort(400, f'batch {batch_id} is already closed')

    jobs_parameters = await request.json()

    validator = cerberus.Validator(schemas.job_array_schema)
    if not validator.validate(jobs_parameters):
        abort(400, 'invalid request: {}'.format(validator.errors))

    jobs_builder = JobsBuilder(db)
    try:
        for job_params in jobs_parameters['jobs']:
            create_job(jobs_builder, batch.id, userdata, job_params)

        success = await jobs_builder.commit()
        if not success:
            abort(400, f'insertion of jobs in db failed')

        log.info(f"created {len(jobs_parameters['jobs'])} jobs for batch {batch_id}")
    finally:
        await jobs_builder.close()

    return jsonify({})


@routes.post('/api/v1alpha/batches/create')
@prom_async_time(REQUEST_TIME_POST_CREATE_BATCH)
@rest_authenticated_users_only
async def create_batch(request, userdata):
    parameters = await request.json()

    validator = cerberus.Validator(schemas.batch_schema)
    if not validator.validate(parameters):
        abort(400, 'invalid request: {}'.format(validator.errors))

    batch = await Batch.create_batch(
        attributes=parameters.get('attributes'),
        callback=parameters.get('callback'),
        userdata=userdata)
    if batch is None:
        abort(400, f'creation of batch in db failed')

    return jsonify(await batch.to_dict(include_jobs=False))


async def _get_batch(batch_id, user, limit=None, offset=None):
    batch = await Batch.from_db(batch_id, user)
    if not batch:
        abort(404)
    return await batch.to_dict(include_jobs=True, limit=limit, offset=offset)


async def _cancel_batch(batch_id, user):
    batch = await Batch.from_db(batch_id, user)
    if not batch:
        abort(404)
    asyncio.ensure_future(batch.cancel())


async def _get_job(batch_id, job_id, user):
    job = await Job.from_db(batch_id, job_id, user)
    if not job:
        abort(404)
    return job.to_dict()


@routes.get('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_POST_GET_BATCH)
@rest_authenticated_users_only
async def get_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    params = request.query
    limit = params.get('limit')
    offset = params.get('offset')
    return jsonify(await _get_batch(batch_id, user, limit=limit, offset=offset))


@routes.patch('/api/v1alpha/batches/{batch_id}/cancel')
@prom_async_time(REQUEST_TIME_PATCH_CANCEL_BATCH)
@rest_authenticated_users_only
async def cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(batch_id, user)
    return jsonify({})


@routes.patch('/api/v1alpha/batches/{batch_id}/close')
@prom_async_time(REQUEST_TIME_PATCH_CLOSE_BATCH)
@rest_authenticated_users_only
async def close_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await Batch.from_db(batch_id, user)
    if not batch:
        abort(404)
    await batch.close()
    return jsonify({})


@routes.delete('/api/v1alpha/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_DELETE_BATCH)
@rest_authenticated_users_only
async def delete_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await Batch.from_db(batch_id, user)
    if not batch:
        abort(404)
    asyncio.ensure_future(batch.mark_deleted())
    return jsonify({})


@routes.get('/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_GET_BATCH_UI)
@aiohttp_jinja2.template('batch.html')
@web_authenticated_users_only()
async def ui_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    params = request.query
    limit = params.get('limit')
    offset = params.get('offset')
    context = base_context(deploy_config, userdata, 'batch')
    context['batch'] = await _get_batch(batch_id, user, limit=limit, offset=offset)
    return context


@routes.post('/batches/{batch_id}/cancel')
@prom_async_time(REQUEST_TIME_POST_CANCEL_BATCH_UI)
@aiohttp_jinja2.template('batches.html')
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def ui_cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(batch_id, user)
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


@routes.get('/batches', name='batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES_UI)
@web_authenticated_users_only()
async def ui_batches(request, userdata):
    params = request.query
    user = userdata['username']
    batches = await _get_batches_list(params, user)
    token = new_csrf_token()
    context = base_context(deploy_config, userdata, 'batch')
    context['batch_list'] = batches[::-1]
    context['token'] = token
    response = aiohttp_jinja2.render_template('batches.html',
                                              request,
                                              context)
    response.set_cookie('_csrf', token, secure=True, httponly=True)
    return response


@routes.get('/batches/{batch_id}/jobs/{job_id}/log')
@prom_async_time(REQUEST_TIME_GET_LOGS_UI)
@aiohttp_jinja2.template('job_log.html')
@web_authenticated_users_only()
async def ui_get_job_log(request, userdata):
    context = base_context(deploy_config, userdata, 'batch')
    batch_id = int(request.match_info['batch_id'])
    context['batch_id'] = batch_id
    job_id = int(request.match_info['job_id'])
    context['job_id'] = job_id
    user = userdata['username']
    context['job_log'] = await _get_job_log(batch_id, job_id, user)
    return context


@routes.get('/batches/{batch_id}/jobs/{job_id}/pod_status')
@prom_async_time(REQUEST_TIME_GET_POD_STATUS_UI)
@aiohttp_jinja2.template('pod_status.html')
@web_authenticated_users_only()
async def ui_get_pod_status(request, userdata):
    context = base_context(deploy_config, userdata, 'batch')
    batch_id = int(request.match_info['batch_id'])
    context['batch_id'] = batch_id
    job_id = int(request.match_info['job_id'])
    context['job_id'] = job_id
    user = userdata['username']
    context['pod_status'] = json.dumps(
        json.loads(await _get_pod_status(batch_id, job_id, user)), indent=2)
    return context


@routes.get('')
@routes.get('/')
@web_authenticated_users_only()
async def index(request, userdata):
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def update_job_with_pod(job, pod):  # pylint: disable=R0911
    log.info(f'update job {job.id if job else "None"} with pod {pod.metadata.name if pod else "None"}')
    if job and job._state == 'Pending':
        if pod:
            log.error(f'job {job.id} has pod {pod.metadata.name}, ignoring')
        return

    if pod and (not job or job.is_complete()):
        err = await app['driver'].delete_pod(name=pod.metadata.name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'failed to delete pod {pod.metadata.name} for job {job.id if job else "None"} due to {err}, ignoring')
        return

    if job and job._cancelled and not job.always_run and job._state == 'Running':
        await job.set_state('Cancelled')
        await job._delete_pod()
        return

    if pod and pod.status and pod.status.phase == 'Pending':
        all_container_statuses = pod.status.container_statuses or []
        for container_status in all_container_statuses:
            if (container_status.state and
                    container_status.state.waiting and
                    container_status.state.waiting.reason):
                await job.mark_complete(pod)
                return

    if not pod:
        log.info(f'job {job.id} no pod found, rescheduling')
        await job.mark_unscheduled()
        return

    if pod and pod.status and pod.status.reason == 'Evicted':
        POD_EVICTIONS.inc()
        log.info(f'job {job.id} mark unscheduled -- pod was evicted')
        await job.mark_unscheduled()
        return

    if pod and pod.status and pod.status.phase in ('Succeeded', 'Failed'):
        log.info(f'job {job.id} mark complete')
        await job.mark_complete(pod)
        return

    if pod and pod.status and pod.status.phase == 'Unknown':
        log.info(f'job {job.id} mark unscheduled -- pod phase is unknown')
        await job.mark_unscheduled()
        return


class DeblockedIterator:
    def __init__(self, it):
        self.it = it

    def __aiter__(self):
        return self

    def __anext__(self):
        return blocking_to_async(app['blocking_pool'], self.it.__next__)


async def pod_changed(pod):
    job = await Job.from_pod(pod)
    await update_job_with_pod(job, pod)


async def refresh_pods():
    log.info(f'refreshing pods')

    # if we do this after we get pods, we will pick up jobs created
    # while listing pods and unnecessarily restart them
    pod_jobs = [Job.from_record(record) for record in await db.jobs.get_records_where({'state': 'Running'})]

    pods = app['driver'].list_pods()
    log.info(f'batch had {len(pods)} pods')

    seen_pods = set()
    for pod_dict in pods:
        pod = v1.api_client._ApiClient__deserialize(pod_dict, kube.client.V1Pod)
        pod_name = pod.metadata.name
        seen_pods.add(pod_name)
        asyncio.ensure_future(pod_changed(pod))

    if len(seen_pods) != len(pod_jobs):
        log.info('restarting running jobs with pods not seen in batch')

    async def restart_job(job):
        log.info(f'restarting job {job.id}')
        await update_job_with_pod(job, None)
    asyncio.gather(*[restart_job(job)
                     for job in pod_jobs
                     if job._pod_name not in seen_pods])


async def driver_event_loop():
    await asyncio.sleep(1)
    while True:
        try:
            object = await app['driver'].complete_queue.get()
            pod = v1.api_client._ApiClient__deserialize(object, kube.client.V1Pod)
            log.info(f'received complete status for pod {pod.metadata.name}')
            await pod_changed(pod)
        except Exception as exc:
            log.exception(f'driver event loop failed due to exception: {exc}')


async def db_cleanup_event_loop():
    await asyncio.sleep(1)
    while True:
        try:
            for record in await db.batch.get_finished_deleted_records():
                batch = Batch.from_record(record, deleted=True)
                await batch.delete()
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'Could not delete batches due to exception: {exc}')
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


@routes.post('/api/v1alpha/instances/activate')
# @rest_authenticated_users_only
async def activate_worker(request):
    return await asyncio.shield(app['driver'].activate_worker(request))


@routes.post('/api/v1alpha/instances/deactivate')
# @rest_authenticated_users_only
async def deactivate_worker(request):
    return await asyncio.shield(app['driver'].deactivate_worker(request))


@routes.post('/api/v1alpha/instances/pod_complete')
# @rest_authenticated_users_only
async def pod_complete(request):
    return await asyncio.shield(app['driver'].pod_complete(request))


setup_aiohttp_jinja2(app, 'batch')

setup_common_static_routes(routes)

app.add_routes(routes)

app.router.add_get("/metrics", server_stats)


async def on_startup(app):
    pool = concurrent.futures.ThreadPoolExecutor()
    k8s = K8s(pool, KUBERNETES_TIMEOUT_IN_SECONDS, BATCH_NAMESPACE, v1)

    userinfo = await async_get_userinfo()
    bucket_name = userinfo['bucket_name']

    driver = Driver(k8s, bucket_name)

    app['blocking_pool'] = pool
    app['driver'] = driver
    app['log_store'] = LogStore(pool, INSTANCE_ID, bucket_name)

    await driver.initialize()
    await refresh_pods()

    asyncio.ensure_future(driver.run())
    asyncio.ensure_future(driver_event_loop())
    asyncio.ensure_future(db_cleanup_event_loop())


app.on_startup.append(on_startup)


async def on_cleanup(app):
    blocking_pool = app['blocking_pool']
    blocking_pool.shutdown()


app.on_cleanup.append(on_cleanup)
