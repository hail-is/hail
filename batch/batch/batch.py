import asyncio
import concurrent
import logging
import os
import traceback
import json
import uuid
from shlex import quote as shq

import aiohttp
from aiohttp import web
import aiohttp_session
import cerberus
import kubernetes as kube
import uvloop
import prometheus_client as pc
from prometheus_async.aio import time as prom_async_time
from prometheus_async.aio.web import server_stats
from hailtop import batch_client
from hailtop.utils import unzip, blocking_to_async
from hailtop.config import get_deploy_config
from hailtop.auth import async_get_userinfo
from gear import setup_aiohttp_session, \
    rest_authenticated_users_only, web_authenticated_users_only, \
    check_csrf_token
# sass_compile,
from web_common import setup_aiohttp_jinja2, setup_common_static_routes, render_template, \
    set_message

from .log_store import LogStore
from .database import BatchDatabase, JobsBuilder
from .datetime_json import JSON_ENCODER
from .k8s import K8s
from .globals import states, complete_states, valid_state_transitions
from .batch_configuration import KUBERNETES_TIMEOUT_IN_SECONDS, REFRESH_INTERVAL_IN_SECONDS, \
    HAIL_POD_NAMESPACE, POD_VOLUME_SIZE, INSTANCE_ID, BATCH_IMAGE, QUEUE_SIZE, MAX_PODS
from .throttler import PodThrottler

from . import schemas


async def scale_queue_consumers(queue, f, n=1):
    for _ in range(n):
        asyncio.ensure_future(f(queue))


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
PVC_CREATION_FAILURES = pc.Counter('batch_pvc_creation_failures', 'Count of batch pvc creation failures')
READ_POD_LOG_FAILURES = pc.Counter('batch_read_pod_log_failures', 'Count of batch read_pod_log failures')

uvloop.install()

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'HAIL_POD_NAMESPACE {HAIL_POD_NAMESPACE}')
log.info(f'POD_VOLUME_SIZE {POD_VOLUME_SIZE}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')
log.info(f'BATCH_IMAGE = {BATCH_IMAGE}')
log.info(f'MAX_PODS = {MAX_PODS}')
log.info(f'QUEUE_SIZE = {QUEUE_SIZE}')

deploy_config = get_deploy_config()

STORAGE_CLASS_NAME = 'batch'

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()
v1 = kube.client.CoreV1Api()

app = web.Application(client_max_size=None)
setup_aiohttp_session(app)

routes = web.RouteTableDef()

db = BatchDatabase.create_synchronous('/batch-user-secret/sql-config.json')

tasks = ('setup', 'main', 'cleanup')


def abort(code, reason=None):
    if code == 400:
        raise web.HTTPBadRequest(reason=reason)
    if code == 404:
        raise web.HTTPNotFound(reason=reason)
    raise web.HTTPException(reason=reason)


def jsonify(data):
    return web.json_response(data)


def resiliently_authenticate(key_file):
    gcloud_auth = f'gcloud -q auth activate-service-account --key-file={key_file}'
    return f"""({gcloud_auth} || (sleep $(( 5 + (RANDOM % 5) )); {gcloud_auth}))"""


def copy(files):
    if files is None:
        return 'true'

    def copy_command(src, dst):
        if not dst.startswith('gs://'):
            mkdirs = f'mkdir -p {shq(os.path.dirname(dst))};'
        else:
            mkdirs = ""
        return f'{mkdirs} gsutil -m cp -R {shq(src)} {shq(dst)}'

    copies = ' && '.join([copy_command(f['from'], f['to']) for f in files])
    return f'set -ex; {resiliently_authenticate("/gsa-key/privateKeyData")} && {copies}'


class JobStateWriteFailure(Exception):
    pass


class Job:
    def _log(self, fun, message, *args, **kwargs):
        fun(f'{self.id} {self._state} {self._pod_name}: ' + message, *args, **kwargs)

    def log_info(self, message, *args, **kwargs):
        self._log(log.info, message, *args, **kwargs)

    def log_warning(self, message, *args, **kwargs):
        self._log(log.warning, message, *args, **kwargs)

    def log_error(self, message, *args, **kwargs):
        self._log(log.error, message, *args, **kwargs)

    async def _create_pvc(self):
        _, err = await app['k8s'].create_pvc(
            body=kube.client.V1PersistentVolumeClaim(
                metadata=kube.client.V1ObjectMeta(
                    name=self._pvc_name,
                    labels={'app': 'batch-job',
                            'hail.is/batch-instance': INSTANCE_ID,
                            'batch_id': str(self.batch_id),
                            'job_id': str(self.job_id),
                            'user': self.user}),
                spec=kube.client.V1PersistentVolumeClaimSpec(
                    access_modes=['ReadWriteOnce'],
                    volume_mode='Filesystem',
                    resources=kube.client.V1ResourceRequirements(
                        requests={'storage': self._pvc_size}),
                    storage_class_name=STORAGE_CLASS_NAME)))
        if err is not None:
            if err.status == 409:
                return True
            self.log_info(f'pvc cannot be created for job with the following error: {err}')
            PVC_CREATION_FAILURES.inc()
            if err.status == 403:
                await self.mark_creation_failed(failure_reason=str(err))
            return False

        self.log_info(f'created pvc name: {self._pvc_name}')
        return True

    def _setup_container(self):
        success_file = f'{self.directory}{LogStore.log_file_name}'

        sh_expression = f"""
        set -ex
        {resiliently_authenticate("/batch-gsa-key/privateKeyData")}
        gsutil -q stat {success_file} && exit 1
        rm -rf /io/*
        {copy(self.input_files)}
         """

        return kube.client.V1Container(
            image='google/cloud-sdk:237.0.0-alpine',
            name='setup',
            command=['/bin/sh', '-c', sh_expression],
            resources=kube.client.V1ResourceRequirements(
                requests={'cpu': '500m'}),
            volume_mounts=[kube.client.V1VolumeMount(
                mount_path='/batch-gsa-key',
                name='batch-gsa-key')])

    def _cleanup_container(self):
        sh_expression = f"""
        set -ex
        python3 -m batch.cleanup_sidecar
        """

        env = [kube.client.V1EnvVar(name='COPY_OUTPUT_CMD',
                                    value=copy(self.output_files))]

        return kube.client.V1Container(
            image=BATCH_IMAGE,
            name='cleanup',
            command=['/bin/sh', '-c', sh_expression],
            env=env,
            resources=kube.client.V1ResourceRequirements(
                requests={'cpu': '500m'}),
            volume_mounts=[
                kube.client.V1VolumeMount(
                    mount_path='/batch-gsa-key',
                    name='batch-gsa-key')],
            ports=[kube.client.V1ContainerPort(container_port=5000)])

    def _keep_alive_container(self):  # pylint: disable=R0201
        sh_expression = f"""
        set -ex
        python3 -m batch.keep_alive_sidecar
        """

        return kube.client.V1Container(
            image=BATCH_IMAGE,
            name='keep-alive',
            command=['/bin/sh', '-c', sh_expression],
            resources=kube.client.V1ResourceRequirements(
                requests={'cpu': '1m'}),
            ports=[kube.client.V1ContainerPort(container_port=5001)])

    async def _create_pod(self):
        assert self.userdata is not None
        assert self._state in states
        assert self._state == 'Running'

        setup_container = self._setup_container()
        cleanup_container = self._cleanup_container()
        keep_alive_container = self._keep_alive_container()

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
                name='gsa-key')]

        if self._pvc_name is not None:
            pvc_created = await self._create_pvc()
            if not pvc_created:
                self.log_info(f'could not create pod due to pvc creation failure')
                return
            volumes.append(kube.client.V1Volume(
                persistent_volume_claim=kube.client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=self._pvc_name),
                name=self._pvc_name))
            volume_mounts.append(kube.client.V1VolumeMount(
                mount_path='/io',
                name=self._pvc_name))

        pod_spec = v1.api_client._ApiClient__deserialize(self._pod_spec, kube.client.V1PodSpec)
        pod_spec.containers.extend([cleanup_container, keep_alive_container])
        pod_spec.init_containers = [setup_container]

        if pod_spec.volumes is None:
            pod_spec.volumes = []
        pod_spec.volumes.extend(volumes)
        for container_set in [pod_spec.containers, pod_spec.init_containers]:
            for container in container_set:
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
                        'user': self.user,
                        'uuid': uuid.uuid4().hex
                        }),
            spec=pod_spec)

        _, err = await app['k8s'].create_pod(body=pod_template)
        if err is not None:
            if err.status == 409:
                self.log_info(f'pod already exists')
                return
            traceback.print_tb(err.__traceback__)
            self.log_info(f'pod creation failed with the following error: {err}')
            return

    async def _delete_pvc(self):
        if self._pvc_name is None:
            return

        self.log_info(f'deleting persistent volume claim {self._pvc_name}')
        err = await app['k8s'].delete_pvc(self._pvc_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            self.log_info(f'ignoring: could not delete {self._pvc_name} due to {err}')

    async def _delete_pod(self):
        err = await app['k8s'].delete_pod(name=self._pod_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            self.log_info(f'ignoring pod deletion failure due to {err}')

    async def _delete_k8s_resources(self):
        await self._delete_pvc()
        await app['pod_throttler'].delete_pod(self)

    async def _read_logs(self):
        if self._state in ('Pending', 'Cancelled'):
            return None

        async def _read_log_from_gcs():
            pod_logs, err = await app['log_store'].read_gs_file(self.directory,
                                                                LogStore.log_file_name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                self.log_info(f'ignoring: could not read log due to {err}')
                return None
            return json.loads(pod_logs)

        async def _read_log_from_k8s(task_name):
            pod_log, err = await app['k8s'].read_pod_log(self._pod_name, container=task_name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                self.log_info(
                    f'ignoring: could not read log due to {err}; will still '
                    f'try to load other tasks')
            return task_name, pod_log

        if self._state == 'Running':
            future_logs = await asyncio.gather(*[_read_log_from_k8s(task) for task in tasks])
            return {k: v for k, v in future_logs}
        if self._state in ('Error', 'Failed', 'Success'):
            return await _read_log_from_gcs()

    async def _read_pod_statuses(self):
        if self._state in ('Pending', 'Cancelled'):
            return None
        if self._state == 'Running':
            pod_status, err = await app['k8s'].read_pod_status(self._pod_name, pretty=True)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                self.log_info(f'ignoring: could not get pod status due to {err}')
            pod_status = pod_status.to_dict()
            return pod_status
        assert self._state in ('Error', 'Failed', 'Success')
        pod_status, err = await app['log_store'].read_gs_file(self.directory,
                                                              LogStore.pod_status_file_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            self.log_info(f'ignoring: could not read pod status due to {err}')
            return None
        return json.loads(pod_status)

    async def _delete_gs_files(self):
        errs = await app['log_store'].delete_gs_files(self.directory)
        for file, err in errs:
            if err is not None:
                traceback.print_tb(err.__traceback__)
                self.log_info(f'could not delete {self.directory}/{file} due to {err}')

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

            return Job(batch_id=record['batch_id'], job_id=record['job_id'], attributes=attributes,
                       userdata=userdata, user=record['user'],
                       always_run=record['always_run'], exit_codes=exit_codes, durations=durations,
                       state=record['state'], pvc_size=record['pvc_size'], cancelled=record['cancelled'],
                       directory=record['directory'], token=record['token'], pod_spec=pod_spec,
                       input_files=input_files, output_files=output_files)

        return None

    @staticmethod
    async def from_k8s_labels(pod):
        if pod.metadata.labels is None:
            return None
        if not set(['batch_id', 'job_id', 'user']).issubset(set(pod.metadata.labels)):
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
    def create_job(jobs_builder, pod_spec, batch_id, job_id, attributes,
                   parent_ids, input_files, output_files, userdata, always_run,
                   pvc_size, state):
        cancelled = False
        user = userdata['username']
        token = uuid.uuid4().hex[:6]

        exit_codes = [None for _ in tasks]
        durations = [None for _ in tasks]
        directory = app['log_store'].gs_job_output_directory(batch_id, job_id, token)
        pod_spec = v1.api_client.sanitize_for_serialization(pod_spec)

        jobs_builder.create_job(
            batch_id=batch_id,
            job_id=job_id,
            state=state,
            pvc_size=pvc_size,
            callback=None,  # legacy
            attributes=json.dumps(attributes),
            always_run=always_run,
            token=token,
            pod_spec=json.dumps(pod_spec),
            input_files=json.dumps(input_files),
            output_files=json.dumps(output_files),
            directory=directory,
            exit_codes=json.dumps(exit_codes),
            durations=json.dumps(durations))

        for parent in parent_ids:
            jobs_builder.create_job_parent(
                batch_id=batch_id,
                job_id=job_id,
                parent_id=parent)

        job = Job(batch_id=batch_id, job_id=job_id, attributes=attributes,
                  userdata=userdata, user=user, always_run=always_run,
                  exit_codes=exit_codes, durations=durations, state=state, pvc_size=pvc_size,
                  cancelled=cancelled, directory=directory, token=token,
                  pod_spec=pod_spec, input_files=input_files, output_files=output_files)

        return job

    def __init__(self, batch_id, job_id, attributes, userdata, user, always_run,
                 exit_codes, durations, state, pvc_size, cancelled, directory,
                 token, pod_spec, input_files, output_files):
        self.batch_id = batch_id
        self.job_id = job_id
        self.id = (batch_id, job_id)

        self.attributes = attributes
        self.always_run = always_run
        self.userdata = userdata
        self.user = user
        self.exit_codes = exit_codes
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

    async def set_state(self, new_state, durations=None, exit_codes=None):
        assert new_state in valid_state_transitions[self._state], f'{self._state} -> {new_state}'
        if self._state != new_state:
            column_updates = {'state': new_state}
            if durations is not None:
                column_updates['durations'] = json.dumps(durations)
            if exit_codes is not None:
                column_updates['exit_codes'] = json.dumps(exit_codes)
            n_updated = await db.jobs.update_record(
                *self.id,
                compare_items={'state': self._state},
                **column_updates)
            if n_updated == 0:
                self.log_warning(
                    f'changing the state from {self._state} -> {new_state} '
                    f'failed due to the expected state not in db')
                raise JobStateWriteFailure()

            self.log_info(f'changed state: {self._state} -> {new_state}')
            self._state = new_state
            if durations is not None:
                self.durations = durations
            if exit_codes is not None:
                self.exit_codes = exit_codes
            await self.notify_children(new_state)

    async def notify_children(self, new_state):
        if new_state not in complete_states:
            self.log_info(f'{new_state} not complete, will not notify children')
            return

        children = [Job.from_record(record) for record in await db.jobs.get_children(*self.id)]
        self.log_info(f'children: {j.id for j in children}')
        for child in children:
            await child.create_if_ready()

    async def create_if_ready(self):
        incomplete_parent_ids = await db.jobs.get_incomplete_parents(*self.id)
        if self._state == 'Pending' and not incomplete_parent_ids:
            await self.set_state('Running')
            parents = [Job.from_record(record) for record in await db.jobs.get_parents(*self.id)]
            if (self.always_run or
                    (not self._cancelled and all(p.is_successful() for p in parents))):
                self.log_info(f'all parents complete creating pod')
                app['pod_throttler'].create_pod(self)
            else:
                self.log_info(f'parents deleted, cancelled, or failed: cancelling')
                await self.set_state('Cancelled')

    async def cancel(self):
        self._cancelled = True

        if not self.always_run and self._state == 'Running':
            await self.set_state('Cancelled')  # must call before deleting resources to prevent race conditions
            await self._delete_k8s_resources()

    def is_complete(self):
        return self._state in complete_states

    def is_successful(self):
        return self._state == 'Success'

    async def mark_unscheduled(self):
        updated_job = await Job.from_db(*self.id, self.user)
        if updated_job.is_complete():
            self.log_info(f'job is already completed in db, not rescheduling pod')
            return

        await app['pod_throttler'].delete_pod(self)
        if self._state == 'Running' and (not self._cancelled or self.always_run):
            app['pod_throttler'].create_pod(self)

    async def _store_status(self, pod):
        pod_status = JSON_ENCODER.encode(pod.status.to_dict())
        err = await app['log_store'].write_gs_file(self.directory,
                                                   LogStore.pod_status_file_name,
                                                   pod_status)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            self.log_info(f'will have a missing pod status due to {err}')

    async def _upload_logs(self, container_logs):
        err = await app['log_store'].write_gs_file(self.directory,
                                                   LogStore.log_file_name,
                                                   json.dumps(container_logs))
        if err is not None:
            traceback.print_tb(err.__traceback__)
            self.log_info(f'will have a missing log due to {err}')

    async def _terminate_keep_alive_pod(self, pod):
        try:
            async with app['client_session'].post(f'http://{pod.status.pod_ip}:5001/') as resp:
                assert resp.status == 200
        except Exception as e:  # pylint: disable=W0703
            self.log_info(f'could not connect to keep-alive pod, but '
                          f'pod will be deleted shortly anyway {e}')

    async def _terminate_cleanup_pod(self, pod):
        try:
            async with app['client_session'].post(f'http://{pod.status.pod_ip}:5000/') as resp:
                assert resp.status == 200
            return None
        except Exception as err:  # pylint: disable=W0703
            return err

    async def mark_creation_failed(self, failure_reason):
        await self._upload_logs({'setup': failure_reason})
        await self._reap_job('Error')

    async def mark_setup_failed(self, pod):
        container_log, err = await app['k8s'].read_pod_log(pod.metadata.name, container='setup')
        if err is not None:
            container_log = err
        container_logs = {'setup': container_log}
        await self._upload_logs(container_logs)
        await self._store_status(pod)
        assert len(pod.status.init_container_statuses) == 1
        startup_status = pod.status.init_container_statuses[0]
        exit_codes = [
            startup_status.state.terminated.exit_code,
            None,
            None]
        finished_at = startup_status.state.terminated.finished_at
        started_at = startup_status.state.terminated.started_at
        durations = [
            finished_at and started_at and ((finished_at - started_at).total_seconds()),
            None,
            None]

        await self._reap_job(
            'Failed',
            exit_codes=exit_codes,
            durations=durations)

    async def mark_terminated(self, pod):
        container_logs, errs = unzip(
            await asyncio.gather(*(
                app['k8s'].read_pod_log(pod.metadata.name, container=container)
                for container in tasks)))
        if any(err is not None for err in errs):
            self.log_info(f'failed to read log {pod.metadata.name} due to {errs} rescheduling')
            await self.mark_unscheduled()
            return
        container_logs = {k: v for k, v in zip(tasks, container_logs)}
        await self._upload_logs(container_logs)
        await self._terminate_keep_alive_pod(pod)
        await self._store_status(pod)
        assert len(pod.status.init_container_statuses) == 1
        startup_status = pod.status.init_container_statuses[0]
        main_status = None
        cleanup_status = None
        for x in pod.status.container_statuses:
            if x.name == 'main':
                assert main_status is None
                main_status = x
            elif x.name == 'cleanup':
                assert cleanup_status is None
                cleanup_status = x
        container_statuses = [startup_status, main_status, cleanup_status]
        exit_codes = [
            status.state.terminated.exit_code
            for status in container_statuses]
        durations = [
            status.state.terminated.finished_at and status.state.terminated.started_at and (
                (status.state.terminated.finished_at - status.state.terminated.started_at).total_seconds())
            for status in container_statuses]

        await self._reap_job(
            'Success' if all(ec == 0 for ec in exit_codes) else 'Failed',
            exit_codes=exit_codes,
            durations=durations)

    async def _reap_job(self, new_state, exit_codes=None, durations=None):
        if exit_codes is None:
            exit_codes = [None for _ in tasks]
        if durations is None:
            durations = [None for _ in tasks]
        await self.set_state(new_state, durations, exit_codes)
        await self._delete_k8s_resources()
        self.log_info(f'complete with state {self._state}, exit_codes {self.exit_codes}')
        if self.batch_id:
            batch = await Batch.from_db(self.batch_id, self.user)
            if batch is not None:
                await batch.mark_job_complete()

    def to_dict(self):
        result = {
            'batch_id': self.batch_id,
            'job_id': self.job_id,
            'state': self._state
        }
        if self.is_complete():
            result['exit_code'] = {
                k: v for k, v in zip(['setup', 'main', 'cleanup'], self.exit_codes)}
            result['duration'] = {
                k: v for k, v in zip(['setup', 'main', 'cleanup'], self.durations)}

        if self.attributes:
            result['attributes'] = self.attributes
        return result


BATCH_JOB_DEFAULT_CPU = os.environ.get('HAIL_BATCH_JOB_DEFAULT_CPU', '1')
BATCH_JOB_DEFAULT_MEMORY = os.environ.get('HAIL_BATCH_JOB_DEFAULT_MEMORY', '3.75G')


def job_spec_to_k8s_pod_spec(job_spec):
    volumes = []
    volume_mounts = []

    if job_spec.get('mount_docker_socket', False):
        volumes.append({
            'name': 'docker-sock-volume',
            'hostPath': {
                'path': '/var/run/docker.sock',
                'type': 'File'
            }
        })
        volume_mounts.append({
            'mountPath': '/var/run/docker.sock',
            'name': 'docker-sock-volume'
        })

    if 'secrets' in job_spec:
        secrets = job_spec['secrets']
        for secret in secrets:
            volumes.append({
                'name': secret['name'],
                'secret': {
                    'secretName': secret['name']
                }
            })
            volume_mounts.append({
                'mountPath': secret['mount_path'],
                'name': secret['name'],
                'readOnly': True
            })

    container = {
        'command': job_spec['command'],
        'image': job_spec['image'],
        'name': 'main',
        'volumeMounts': volume_mounts
    }
    if 'env' in job_spec:
        container['env'] = job_spec['env']

    # defaults
    cpu = BATCH_JOB_DEFAULT_CPU
    memory = BATCH_JOB_DEFAULT_MEMORY
    if 'resources' in job_spec:
        resources = job_spec['resources']
        if 'memory' in resources:
            memory = resources['memory']
        if 'cpu' in resources:
            cpu = resources['cpu']
    container['resources'] = {
        'requests': {
            'cpu': cpu,
            'memory': memory
        },
        'limits': {
            'cpu': cpu,
            'memory': memory
        }
    }
    pod_spec = {
        'containers': [container],
        'restartPolicy': 'Never',
        'tolerations': [{
            'key': 'preemptible',
            'value': 'true'
        }],
        'volumes': volumes
    }
    if 'service_account_name' in job_spec:
        pod_spec['serviceAccountName'] = job_spec['service_account_name']
    return pod_spec


def create_job(jobs_builder, batch_id, userdata, job_spec):  # pylint: disable=R0912
    job_id = job_spec['job_id']
    parent_ids = job_spec.get('parent_ids', [])
    input_files = job_spec.get('input_files')
    output_files = job_spec.get('output_files')
    pvc_size = job_spec.get('pvc_size')
    if pvc_size is None and (input_files or output_files):
        pvc_size = POD_VOLUME_SIZE
    always_run = job_spec.get('always_run', False)

    pod_spec = job_spec_to_k8s_pod_spec(job_spec)

    state = 'Running' if len(parent_ids) == 0 else 'Pending'

    job = Job.create_job(
        jobs_builder,
        batch_id=batch_id,
        job_id=job_id,
        pod_spec=pod_spec,
        attributes=job_spec.get('attributes'),
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
    return jsonify({})


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

    pod_statuses = await job._read_pod_statuses()
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
    async def create_batch(attributes, callback, userdata, n_jobs):
        user = userdata['username']

        id = await db.batch.new_record(
            attributes=json.dumps(attributes),
            callback=callback,
            userdata=json.dumps(userdata),
            user=user,
            deleted=False,
            cancelled=False,
            closed=False,
            n_jobs=n_jobs)

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
                app['pod_throttler'].create_pod(j)

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

    async def mark_job_complete(self):
        if self.complete and self.callback:
            log.info(f'making callback for batch {self.id}: {self.callback}')
            try:
                await app['client_session'].post(self.callback, json=await self.to_dict(include_jobs=False))
                log.info(f'callback for batch {self.id} successful')
            except Exception:  # pylint: disable=broad-except
                log.exception(f'callback for batch {self.id} failed, will not retry.')

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

    jobs = await request.json()
    try:
        batch_client.validate.validate_jobs(jobs)
    except batch_client.validate.ValidationError as e:
        abort(400, e.reason)

    jobs_builder = JobsBuilder(db)
    try:
        for job in jobs:
            create_job(jobs_builder, batch.id, userdata, job)

        success = await jobs_builder.commit()
        if not success:
            abort(400, f'insertion of jobs in db failed')

        log.info(f"created {len(jobs)} jobs for batch {batch_id}")
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
        userdata=userdata,
        n_jobs=parameters['n_jobs'])
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
@web_authenticated_users_only()
async def ui_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    params = request.query
    limit = params.get('limit')
    offset = params.get('offset')
    page_context = {
        'batch': await _get_batch(batch_id, user, limit=limit, offset=offset)
    }
    return await render_template('batch', request, userdata, 'batch.html', page_context)


@routes.post('/batches/{batch_id}/cancel')
@prom_async_time(REQUEST_TIME_POST_CANCEL_BATCH_UI)
@check_csrf_token
@web_authenticated_users_only(redirect=False)
async def ui_cancel_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    await _cancel_batch(batch_id, user)
    session = await aiohttp_session.get_session(request)
    set_message(session, 'Batch {batch_id} cancelled.', 'info')
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


@routes.get('/batches', name='batches')
@prom_async_time(REQUEST_TIME_GET_BATCHES_UI)
@web_authenticated_users_only()
async def ui_batches(request, userdata):
    params = request.query
    user = userdata['username']
    batches = await _get_batches_list(params, user)
    page_context = {
        'batch_list': batches[::-1],
    }
    return await render_template('batch', request, userdata, 'batches.html', page_context)


@routes.get('/batches/{batch_id}/jobs/{job_id}/log')
@prom_async_time(REQUEST_TIME_GET_LOGS_UI)
@web_authenticated_users_only()
async def ui_get_job_log(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'job_log': await _get_job_log(batch_id, job_id, user)
    }
    return await render_template('batch', request, userdata, 'job_log.html', page_context)


@routes.get('/batches/{batch_id}/jobs/{job_id}/pod_status')
@prom_async_time(REQUEST_TIME_GET_POD_STATUS_UI)
@web_authenticated_users_only()
async def ui_get_pod_status(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    page_context = {
        'batch_id': batch_id,
        'job_id': job_id,
        'pod_status': json.dumps(
            json.loads(await _get_pod_status(batch_id, job_id, user)), indent=2)
    }
    return await render_template('batch', request, userdata, 'pod_status.html', page_context)


@routes.get('')
@routes.get('/')
@web_authenticated_users_only()
async def index(request, userdata):
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def update_job_with_pod(job, pod):  # pylint: disable=R0911,R0915
    log.info(f'update job {job.id if job else "None"} with pod {pod.metadata.name if pod else "None"}')
    if job and job._state == 'Pending':
        if pod:
            log.error('job {job.id} has pod {pod.metadata.name}, ignoring')
        return

    if pod and (not job or job.is_complete()):
        err = await app['k8s'].delete_pod(name=pod.metadata.name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'failed to delete pod {pod.metadata.name} for job {job.id if job else "None"} due to {err}, ignoring')

        err = await app['k8s'].delete_pvc(name=pod.metadata.name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'failed to delete pvc {pod.metadata.name} for job {job.id if job else "None"} due to {err}, ignoring')
        return

    if job and job._cancelled and not job.always_run and job._state == 'Running':
        await job.set_state('Cancelled')
        await job._delete_k8s_resources()
        return

    if pod and pod.status and pod.status.phase == 'Pending':
        def image_pull_back_off_reason(container_status):
            if (container_status.state and
                    container_status.state.waiting and
                    container_status.state.waiting.reason == 'ImagePullBackOff'):
                return (container_status.state.waiting.reason +
                        ': ' +
                        container_status.state.waiting.message)
            return None

        all_container_statuses = []
        all_container_statuses.extend(pod.status.init_container_statuses or [])
        all_container_statuses.extend(pod.status.container_statuses or [])

        image_pull_back_off_reasons = []
        for container_status in all_container_statuses:
            maybe_reason = image_pull_back_off_reason(container_status)
            if maybe_reason:
                image_pull_back_off_reasons.append(maybe_reason)
        if image_pull_back_off_reasons:
            await job.mark_creation_failed("\n".join(image_pull_back_off_reasons))
            return

    if not pod and not app['pod_throttler'].is_queued(job):
        log.info(f'job {job.id} no pod found, rescheduling')
        await job.mark_unscheduled()
        return

    if pod and pod.status and pod.status.reason == 'Evicted':
        POD_EVICTIONS.inc()
        log.info(f'job {job.id} mark unscheduled -- pod was evicted')
        await job.mark_unscheduled()
        return

    if pod and pod.status:  # pylint: disable=R1702
        if pod.status.init_container_statuses:
            assert len(pod.status.init_container_statuses) == 1, pod
            init_status = pod.status.init_container_statuses[0]
            if init_status.state and init_status.state.terminated and init_status.state.terminated.exit_code != 0:
                log.info(f'job {job.id} failed -- setup container exited {init_status.state.terminated.exit_code}')
                await job.mark_setup_failed(pod)
                return
        if pod.status.container_statuses:
            main_status = [x for x in pod.status.container_statuses
                           if x.name == 'main']
            if len(main_status) != 0:
                main_status = main_status[0]
                if main_status.state and main_status.state.terminated:
                    cleanup_status = [x for x in pod.status.container_statuses
                                      if x.name == 'cleanup']
                    if len(cleanup_status) != 0:
                        cleanup_status = cleanup_status[0]
                        if cleanup_status.state and cleanup_status.state.terminated:
                            log.info(f'job {job.id} mark complete')
                            await job.mark_terminated(pod=pod)
                            return
                        err = await job._terminate_cleanup_pod(pod)
                        if err:
                            log.info(f'could not connect to cleanup pod, we will '
                                     f'try again in next refresh loop {job.id} {pod} {err}')
                            return
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
    job = await Job.from_k8s_labels(pod)
    await update_job_with_pod(job, pod)


async def kube_event_loop():
    while True:
        try:
            stream = kube.watch.Watch().stream(
                v1.list_namespaced_pod,
                HAIL_POD_NAMESPACE,
                label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}')
            async for event in DeblockedIterator(stream):
                type = event['type']
                object = event['object']
                name = object.metadata.name
                log.info(f'received event {type} {name}')
                if type == 'ERROR':
                    log.info(f'kubernetes sent an ERROR event: {event}')
                    continue
                await pod_changed(object)
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'k8s event stream failed due to: {exc}')
        await asyncio.sleep(5)


async def refresh_k8s_pods():
    log.info(f'refreshing k8s pods')

    # if we do this after we get pods, we will pick up jobs created
    # while listing pods and unnecessarily restart them
    pod_jobs = [Job.from_record(record) for record in await db.jobs.get_records_where({'state': 'Running'})]

    pods, err = await app['k8s'].list_pods(
        label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}')
    if err is not None:
        traceback.print_tb(err.__traceback__)
        log.info(f'could not refresh pods due to {err}, will try again later')
        return

    log.info(f'k8s had {len(pods.items)} pods')

    seen_pods = set()

    async def see_pod(pod):
        pod_name = pod.metadata.name
        seen_pods.add(pod_name)
        await pod_changed(pod)
    await asyncio.gather(*[see_pod(pod) for pod in pods.items])

    if app['pod_throttler'].full():
        log.info(f'pod creation queue is full; skipping restarting jobs not seen in k8s')
        return

    log.info('restarting running jobs with pods not seen in k8s')

    async def restart_job(job):
        log.info(f'restarting job {job.id}')
        await update_job_with_pod(job, None)
    await asyncio.gather(*[restart_job(job)
                           for job in pod_jobs
                           if job._pod_name not in seen_pods])


async def refresh_k8s_pvc():
    pvcs, err = await app['k8s'].list_pvcs(
        label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}')
    if err is not None:
        traceback.print_tb(err.__traceback__)
        log.info(f'could not refresh pvcs due to {err}, will try again later')
        return

    log.info(f'k8s had {len(pvcs.items)} pvcs')

    for pvc in pvcs.items:
        job = await Job.from_k8s_labels(pvc)
        if job is None or job.is_complete():
            log.info(f'deleting orphaned pvc {pvc.metadata.name}')
            err = await app['k8s'].delete_pvc(pvc.metadata.name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'could not delete {pvc.metadata.name} due to {err}')


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
    await asyncio.sleep(1)
    while True:
        try:
            for record in await db.batch.get_finished_deleted_records():
                batch = Batch.from_record(record, deleted=True)
                await batch.delete()
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'Could not delete batches due to exception: {exc}')
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


setup_aiohttp_jinja2(app, 'batch')

setup_common_static_routes(routes)

app.add_routes(routes)

app.router.add_get("/metrics", server_stats)


async def on_startup(app):
    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool
    app['k8s'] = K8s(pool, KUBERNETES_TIMEOUT_IN_SECONDS, HAIL_POD_NAMESPACE, v1)

    userinfo = await async_get_userinfo()

    app['log_store'] = LogStore(pool, INSTANCE_ID, userinfo['bucket_name'])
    app['pod_throttler'] = PodThrottler(QUEUE_SIZE, MAX_PODS, parallelism=16)
    app['client_session'] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(10))

    asyncio.ensure_future(polling_event_loop())
    asyncio.ensure_future(kube_event_loop())
    asyncio.ensure_future(db_cleanup_event_loop())


app.on_startup.append(on_startup)


async def on_cleanup(app):
    app['blocking_pool'].shutdown()
    app['client_session'].close()


app.on_cleanup.append(on_cleanup)
