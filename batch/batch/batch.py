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
import prometheus_client as pc
from prometheus_async.aio import time as prom_async_time
from prometheus_async.aio.web import server_stats

from hailtop import gear
from hailtop.gear.auth import rest_authenticated_users_only, web_authenticated_users_only, \
    new_csrf_token, check_csrf_token

from .blocking_to_async import blocking_to_async
from .log_store import LogStore
from .database import BatchDatabase, JobsBuilder, JobsTable
from .k8s import K8s
from .globals import complete_states
from .queue import scale_queue_consumers

from . import schemas

gear.configure_logging()
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

KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
HAIL_POD_NAMESPACE = os.environ.get('HAIL_POD_NAMESPACE', 'batch-pods')
POD_VOLUME_SIZE = os.environ.get('POD_VOLUME_SIZE', '10Mi')
INSTANCE_ID = os.environ.get('HAIL_INSTANCE_ID', uuid.uuid4().hex)
BATCH_IMAGE = os.environ.get('BATCH_IMAGE', 'gcr.io/hail-vdc/batch:latest')

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'HAIL_POD_NAMESPACE {HAIL_POD_NAMESPACE}')
log.info(f'POD_VOLUME_SIZE {POD_VOLUME_SIZE}')
log.info(f'INSTANCE_ID = {INSTANCE_ID}')
log.info(f'BATCH_IMAGE = {BATCH_IMAGE}')

STORAGE_CLASS_NAME = 'batch'

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()
v1 = kube.client.CoreV1Api()

app = web.Application(client_max_size=None)
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


class Job:
    async def _create_pvc(self):
        pvc, err = await app['k8s'].create_pvc(
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
            log.info(f'pvc cannot be created for job {self.id} with the following error: {err}')
            PVC_CREATION_FAILURES.inc()
            if err.status == 403:
                await self.mark_complete(None, failed=True, failure_reason=str(err))
            return False

        log.info(f'created pvc name: {self._pvc_name} for job {self.id}')
        return True

    def _setup_container(self, success_file):
        sh_expression = f"""
        set -ex
        if [ -e {success_file} ]; then
            exit 1
        fi
        rm -rf /io/*
        {copy(self.input_files)}
         """

        setup_container = kube.client.V1Container(
            image='google/cloud-sdk:237.0.0-alpine',
            name='setup',
            command=['/bin/sh', '-c', sh_expression],
            resources=kube.client.V1ResourceRequirements(
                requests={'cpu': '500m'}))

        return setup_container

    def _cleanup_container(self, success_file):
        sh_expression = f"""
        set -ex
        python3 sidecar.py
        touch {success_file}
        """

        env = {'INSTANCE_ID': INSTANCE_ID,
               'OUTPUT_DIRECTORY': self.directory,
               'COPY_OUTPUT_CMD': copy(self.output_files),
               'BATCH_USE_KUBE_CONFIG': os.environ.get('BATCH_USE_KUBE_CONFIG')}
        env = [kube.client.V1EnvVar(name=name, value=value) for name, value in env.items()]
        env.append(kube.client.V1EnvVar(name='POD_NAME', value_from='metadata.name'))

        cleanup_container = kube.client.V1Container(
            image=BATCH_IMAGE,
            name='cleanup',
            command=['/bin/sh', '-c', sh_expression],
            env=env,
            resources=kube.client.V1ResourceRequirements(
                requests={'cpu': '500m'}),
            volume_mounts=[kube.client.V1VolumeMount(
                mount_path='/batch-gsa-key',
                name='batch-gsa-key')])

        return cleanup_container

    async def _create_pod(self):
        assert self.userdata is not None

        success_file = '/io/__BATCH_SUCCESS__'  # FIXME: /io doesn't always exist
        setup_container = self._setup_container(success_file)
        cleanup_container = self._cleanup_container(success_file)

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
                log.info(f'could not create pod for job {self.id} due to pvc creation failure')
                return
            volumes.append(kube.client.V1Volume(
                persistent_volume_claim=kube.client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=self._pvc_name),
                name=self._pvc_name))
            volume_mounts.append(kube.client.V1VolumeMount(
                mount_path='/io',
                name=self._pvc_name))

        pod_spec = v1.api_client._ApiClient__deserialize(self._pod_spec, kube.client.V1PodSpec)
        pod_spec.containers.extend(cleanup_container)
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

        pod, err = await app['k8s'].create_pod(body=pod_template)
        if err is not None:
            if err.status == 409:
                log.info(f'pod already exists for job {self.id}')
                n_updated = await db.jobs.update_record(*self.id, compare_items={'state': self._state}, state='Running')
                if n_updated == 0:
                    log.warning(f'changing the state for job {self.id} failed due to the expected state {self._state} not in db')
                return
            traceback.print_tb(err.__traceback__)
            log.info(f'pod creation failed for job {self.id} '
                     f'with the following error: {err}')
            return

        n_updated = await db.jobs.update_record(*self.id, compare_items={'state': self._state}, state='Running')
        if n_updated == 0:
            log.warning(f'changing the state for job {self.id} failed due to the expected state {self._state} not in db')

    async def _delete_pvc(self):
        if self._pvc_name is None:
            return

        log.info(f'deleting persistent volume claim {self._pvc_name}')
        err = await app['k8s'].delete_pvc(self._pvc_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'ignoring: could not delete {self._pvc_name} due to {err}')

    async def _delete_pod(self):
        err = await app['k8s'].delete_pod(name=self._pod_name)
        if err is not None:
            traceback.print_tb(err.__traceback__)
            log.info(f'ignoring pod deletion failure for job {self.id} due to {err}')

    async def _delete_k8s_resources(self):
        await self._delete_pvc()
        await self._delete_pod()

    async def _read_logs(self):
        async def _read_log_from_gcs():
            pod_logs, err = await app['log_store'].read_gs_file(self.directory,
                                                                LogStore.log_file_name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read log for {self.id} '
                         f'due to {err}')
                return None
            return json.loads(pod_logs)

        async def _read_log_from_k8s(task_name):
            pod_log, err = await app['k8s'].read_pod_log(self._pod_name, container=task_name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read log for {self.id} '
                         f'due to {err}; will still try to load other tasks')
            return task_name, pod_log

        if self._state == 'Running':
            future_logs = asyncio.gather(*[_read_log_from_k8s(task) for task in tasks])
            return {k: v for k, v in await future_logs}
        elif self._state in ('Error', 'Failed', 'Success'):
            return await _read_log_from_gcs()
        else:
            assert self._state in ('Ready', 'Pending', 'Cancelled')
            return None

    async def _read_pod_statuses(self):
        async def _read_pod_status_from_gcs():
            pod_status, err = await app['log_store'].read_gs_file(self.directory,
                                                                  LogStore.pod_status_file_name)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not read pod status for {self.id} '
                         f'due to {err}')
                return None
            return json.loads(pod_status)

        if self._state == 'Running':
            pod_status, err = await app['k8s'].read_pod_status(self._pod_name, pretty=True)
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'ignoring: could not get pod status for {self.id} '
                         f'due to {err}')
            return pod_status
        elif self._state in ('Error', 'Failed', 'Success'):
            return await _read_pod_status_from_gcs()
        else:
            assert self._state in ('Pending', 'Ready')
            return None

    async def _delete_gs_files(self):
        errs = await app['log_store'].delete_gs_files(self.directory)
        for file, err in errs:
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'could not delete {self.directory}/{file} for job {self.id} due to {err}')

    @staticmethod
    def from_record(record):
        if record is not None:
            attributes = json.loads(record['attributes'])
            userdata = json.loads(record['userdata'])
            pod_spec = v1.api_client.sanitize_for_serialization(record['pod_spec'])
            input_files = json.loads(record['input_files'])
            output_files = json.loads(record['output_files'])
            exit_codes = json.loads(record['exit_codes'])
            durations = json.loads(record['durations'])

            return Job(batch_id=record['batch_id'], job_id=record['job_id'], attributes=attributes,
                       callback=record['callback'], userdata=userdata, user=record['user'],
                       always_run=record['always_run'], exit_codes=exit_codes, durations=durations,
                       state=record['state'], pvc_size=record['pvc_size'], cancelled=record['cancelled'],
                       directory=record['directory'], token=record['token'], pod_spec=pod_spec,
                       input_files=input_files, output_files=output_files)

        return None

    @staticmethod
    async def from_k8s_labels(pod):
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
            durations=json.dumps(durations))

        for parent in parent_ids:
            jobs_builder.create_job_parent(
                batch_id=batch_id,
                job_id=job_id,
                parent_id=parent)

        job = Job(batch_id=batch_id, job_id=job_id, attributes=attributes, callback=callback,
                  userdata=userdata, user=user, always_run=always_run,
                  exit_codes=exit_codes, durations=durations, state=state, pvc_size=pvc_size,
                  cancelled=cancelled, directory=directory, token=token,
                  pod_spec=pod_spec, input_files=input_files, output_files=output_files)

        return job

    def __init__(self, batch_id, job_id, attributes, callback, userdata, user, always_run,
                 exit_codes, durations, state, pvc_size, cancelled, directory,
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
        self.directory = directory
        self.durations = durations
        self.token = token
        self.input_files = input_files
        self.output_files = output_files

        name = f'batch-{batch_id}-job-{job_id}-{token}'
        self._pod_name = name
        self._pvc_name = name if self.input_files or self.output_files or pvc_size else None
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
        if self._state != new_state:
            n_updated = await db.jobs.update_record(*self.id, compare_items={'state': self._state}, state=new_state)
            if n_updated == 0:
                log.warning(f'changing the state from {self._state} -> {new_state} '
                            f'for job {self.id} failed due to the expected state not in db')
                return

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
        assert parent_batch_id == self.batch_id
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

        if not self.always_run:
            await self.set_state('Cancelled')  # must call before deleting resources to prevent race conditions
            await self._delete_k8s_resources()

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
        if not self._cancelled or self.always_run:
            await self.set_state('Ready')
            await self._create_pod()

    async def mark_complete(self, pod, failed=False, failure_reason=None):
        if pod is not None:
            assert pod.metadata.name == self._pod_name
            if self.is_complete():
                log.info(f'ignoring because job {self.id} is already complete')
                return

        new_state = None
        exit_codes = [None for _ in tasks]
        durations = [None for _ in tasks]
        container_logs = {t: None for t in tasks}

        if failed:
            if failure_reason is not None:
                container_logs['setup'] = failure_reason
                uri, err = await app['log_store'].write_gs_file(self.directory,
                                                                LogStore.log_file_name,
                                                                json.dumps(container_logs))
                if err is not None:
                    traceback.print_tb(err.__traceback__)
                    log.info(f'job {self.id} will have a missing log due to {err}')

            if pod is not None:
                pod_status = pod.status.to_str()
                err = await app['log_store'].write_gs_file(self.directory,
                                                           LogStore.pod_status_file_name,
                                                           pod_status)
                if err is not None:
                    traceback.print_tb(err.__traceback__)
                    log.info(f'job {self.id} will have a missing pod status due to {err}')

            new_state = 'Error'
        else:
            pod_outputs = await app['log_store'].read_gs_file(self.directory, LogStore.results_file_name)

            if pod_outputs is None:
                setup_container = pod.status.init_container_statuses[0]
                assert setup_container.name == 'setup'
                setup_terminated = setup_container.state.terminated

                if setup_terminated.exit_code != 0:
                    if setup_terminated.finished_at is not None and setup_terminated.started_at is not None:
                        durations[0] = (setup_terminated.finished_at - setup_terminated.started_at).total_seconds()
                    else:
                        log.warning(f'setup container terminated but has no timing information. {setup_container.to_str()}')

                    container_log, err = await app['k8s'].read_pod_log(pod.metadata.name, container='setup')
                    if err:
                        await self.mark_unscheduled()

                    container_logs['setup'] = container_log
                    exit_codes[0] = setup_terminated.exit_code

                    await app['log_store'].write_gs_file(self.directory,
                                                         LogStore.log_file_name,
                                                         json.dumps(container_logs))

                    pod_status, err = await app['k8s'].read_pod_status(pod.metadata.name)
                    if err:
                        await self.mark_unscheduled()
                    else:
                        assert pod_status is not None
                        await app['log_store'].write_gs_file(self.directory,
                                                             LogStore.pod_status_file_name,
                                                             pod_status)

                    new_state = 'Failed'
                else:
                    # cleanup sidecar container must have failed (error in copying outputs doesn't cause a non-zero exit code)
                    cleanup_container = [status for status in pod.status.container_statuses if status.name == 'cleanup'][0]
                    assert cleanup_container.state.terminated.exit_code != 0
                    log.info(f'rescheduling job {self.id} -- cleanup container failed')
                    await self.mark_unscheduled()
            else:
                pod_outputs = json.loads(pod_outputs)

                READ_POD_LOG_FAILURES.inc()
                exit_codes = pod_outputs['exit_codes']
                durations = pod_outputs['durations']

                if all([ec == 0 for ec in self.exit_codes]):
                    new_state = 'Success'
                else:
                    new_state = 'Failed'

        assert new_state is not None
        n_updated = await db.jobs.update_record(*self.id,
                                                compare_items={'state': self._state},
                                                durations=json.dumps(durations),
                                                exit_codes=json.dumps(exit_codes),
                                                state=new_state)

        if n_updated == 0:
            log.info(f'could not update job {self.id} due to db not matching expected state')
            return

        self.exit_codes = exit_codes
        self.durations = durations

        if self._state != new_state:
            log.info('job {} changed state: {} -> {}'.format(
                self.id,
                self._state,
                new_state))

        self._state = new_state

        await self._delete_pod()
        await self._delete_pvc()
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

        await app['gcs'].delete_file(self.directory, LogStore.results_file_name)

    def to_dict(self):
        result = {
            'batch_id': self.batch_id,
            'job_id': self.job_id,
            'state': self._state
        }
        if self.is_complete():
            result['exit_code'] = self.exit_codes
            result['duration'] = self.durations

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

    state = 'Ready' if len(parent_ids) == 0 else 'Pending'

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
        return pod_statuses
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

    async def get_jobs(self):
        return [Job.from_record(record) for record in await db.jobs.get_records_by_batch(self.id)]

    async def _cancel_jobs(self):
        for j in await self.get_jobs():
            await j.cancel()

    async def cancel(self):
        await db.batch.update_record(self.id, cancelled=True, closed=True)
        self.cancelled = True
        self.closed = True
        asyncio.ensure_future(self._cancel_jobs())
        log.info(f'batch {self.id} cancelled')

    async def _close_jobs(self):
        for j in await self.get_jobs():
            if j._state == 'Ready':
                await app['start_job_queue'].put(j)

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

    async def to_dict(self, include_jobs=False):
        result = {
            'id': self.id,
            'state': self.state,
            'complete': self.complete,
            'closed': self.closed
        }
        if self.attributes:
            result['attributes'] = self.attributes
        if include_jobs:
            jobs = await self.get_jobs()
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
        if k == 'complete' or k == 'success':  # params does not support deletion
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
@prom_async_time(REQUEST_TIME_POST_GET_BATCH)
@rest_authenticated_users_only
async def get_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    return jsonify(await _get_batch(batch_id, user))


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
    await batch.mark_deleted()
    return jsonify({})


@routes.get('/batches/{batch_id}')
@prom_async_time(REQUEST_TIME_GET_BATCH_UI)
@aiohttp_jinja2.template('batch.html')
@web_authenticated_users_only
async def ui_batch(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    user = userdata['username']
    batch = await _get_batch(batch_id, user)
    return {'batch': batch}


@routes.post('/batches/{batch_id}/cancel')
@prom_async_time(REQUEST_TIME_POST_CANCEL_BATCH_UI)
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
@prom_async_time(REQUEST_TIME_GET_BATCHES_UI)
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
@prom_async_time(REQUEST_TIME_GET_LOGS_UI)
@aiohttp_jinja2.template('job_log.html')
@web_authenticated_users_only
async def ui_get_job_log(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    job_log = await _get_job_log(batch_id, job_id, user)
    return {'batch_id': batch_id, 'job_id': job_id, 'job_log': job_log}


@routes.get('/batches/{batch_id}/jobs/{job_id}/pod_status')
@prom_async_time(REQUEST_TIME_GET_POD_STATUS_UI)
@aiohttp_jinja2.template('pod_status.html')
@web_authenticated_users_only
async def ui_get_pod_status(request, userdata):
    batch_id = int(request.match_info['batch_id'])
    job_id = int(request.match_info['job_id'])
    user = userdata['username']
    pod_status = await _get_pod_status(batch_id, job_id, user)
    return {'batch_id': batch_id, 'job_id': job_id, 'pod_status': pod_status}


@routes.get('/')
@web_authenticated_users_only
async def batch_id(request, userdata):
    location = request.app.router['batches'].url_for()
    raise web.HTTPFound(location=location)


async def update_job_with_pod(job, pod):
    log.info(f'update job {job.id} with pod {pod.metadata.name if pod else "None"}')
    if not pod or (pod.status and pod.status.reason == 'Evicted'):
        POD_EVICTIONS.inc()
        log.info(f'job {job.id} mark unscheduled -- pod is missing or was evicted')
        await job.mark_unscheduled()
    elif pod and pod.phase in ('Succeeded', 'Failed'):
        log.info(f'job {job.id} mark complete')
        await job.mark_complete(pod)
    elif pod and pod.phase == 'Unknown':
        log.info(f'job {job.id} mark unscheduled -- pod phase is unknown')
        await job.mark_unscheduled()


class DeblockedIterator:
    def __init__(self, it):
        self.it = it

    def __aiter__(self):
        return self

    def __anext__(self):
        return blocking_to_async(app['blocking_pool'], self.it.__next__)


async def pod_changed(pod):
    job = await Job.from_k8s_labels(pod)
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
                event_type = event['type']
                object = event['object']
                kind = object['kind']
                object_name = object['metadata']['name']
                log.info(f'received {event_type} for {kind} named {object_name}')
                await pod_changed(object)
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'k8s event stream failed due to: {exc}')
        await asyncio.sleep(5)


async def refresh_k8s_pods():
    # if we do this after we get pods, we will pick up jobs created
    # while listing pods and unnecessarily restart them
    pod_jobs = [Job.from_record(record) for record in await db.jobs.get_records_where({'state': ['Ready', 'Running']})]

    pods, err = await app['k8s'].list_pods(
        label_selector=f'app=batch-job,hail.is/batch-instance={INSTANCE_ID}')
    if err is not None:
        traceback.print_tb(err.__traceback__)
        log.info(f'could not refresh pods due to {err}, will try again later')
        return

    log.info(f'k8s had {len(pods.items)} pods')

    seen_pods = set()
    for pod in pods.items:
        pod_name = pod.metadata.name
        seen_pods.add(pod_name)

        job = await Job.from_k8s_labels(pod)
        if job and not job.is_complete():
            await update_job_with_pod(job, pod)

    log.info('restarting ready and running jobs with pods not seen in k8s')

    for job in pod_jobs:
        if job._pod_name not in seen_pods:
            log.info(f'restarting job {job.id}')
            await update_job_with_pod(job, None)


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


async def start_job(queue):
    while True:
        job = await queue.get()
        if job._state == 'Ready':
            try:
                await job._create_pod()
            except Exception as exc:  # pylint: disable=W0703
                log.exception(f'Could not create pod for job {job.id} due to exception: {exc}')
        queue.task_done()


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


batch_root = os.path.dirname(os.path.abspath(__file__))
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(os.path.join(batch_root, 'templates')))
routes.static('/static', os.path.join(batch_root, 'static'))
routes.static('/js', os.path.join(batch_root, 'js'))
app.add_routes(routes)
app.router.add_get("/metrics", server_stats)


async def on_startup(app):
    pool = concurrent.futures.ThreadPoolExecutor()
    app['blocking_pool'] = pool
    app['k8s'] = K8s(pool, KUBERNETES_TIMEOUT_IN_SECONDS, HAIL_POD_NAMESPACE, v1)
    app['log_store'] = LogStore(pool, INSTANCE_ID)
    app['start_job_queue'] = asyncio.Queue()

    asyncio.ensure_future(polling_event_loop())
    asyncio.ensure_future(kube_event_loop())
    asyncio.ensure_future(db_cleanup_event_loop())
    asyncio.ensure_future(scale_queue_consumers(app['start_job_queue'], start_job, n=16))


app.on_startup.append(on_startup)


async def on_cleanup(app):
    blocking_pool = app['blocking_pool']
    blocking_pool.shutdown()


app.on_cleanup.append(on_cleanup)
