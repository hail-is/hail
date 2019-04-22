import asyncio
import concurrent
import functools
import logging
import os
import threading
import uuid

from aiohttp import web
import aiohttp_jinja2
import cerberus
import jinja2
import jwt
import kubernetes as kube
import requests
import uvloop

import hailjwt as hj

from .globals import max_id, pod_name_job, job_id_job, batch_id_batch
from .globals import next_id, get_recent_events, add_event, blocking_to_async
from .globals import write_gs_log_file, read_gs_log_file, delete_gs_log_file
from .database import BatchDatabase

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


def schedule(ttl, fun, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    asyncio.get_event_loop().call_later(ttl, functools.partial(fun, *args, **kwargs))


KUBERNETES_TIMEOUT_IN_SECONDS = float(os.environ.get('KUBERNETES_TIMEOUT_IN_SECONDS', 5.0))
REFRESH_INTERVAL_IN_SECONDS = int(os.environ.get('REFRESH_INTERVAL_IN_SECONDS', 5 * 60))
HAIL_POD_NAMESPACE = os.environ.get('HAIL_POD_NAMESPACE', 'batch-pods')
POD_VOLUME_SIZE = os.environ.get('POD_VOLUME_SIZE', '10Mi')

log.info(f'KUBERNETES_TIMEOUT_IN_SECONDS {KUBERNETES_TIMEOUT_IN_SECONDS}')
log.info(f'REFRESH_INTERVAL_IN_SECONDS {REFRESH_INTERVAL_IN_SECONDS}')
log.info(f'HAIL_POD_NAMESPACE {HAIL_POD_NAMESPACE}')
log.info(f'POD_VOLUME_SIZE {POD_VOLUME_SIZE}')

STORAGE_CLASS_NAME = 'batch'

if 'BATCH_USE_KUBE_CONFIG' in os.environ:
    kube.config.load_kube_config()
else:
    kube.config.load_incluster_config()
v1 = kube.client.CoreV1Api()

instance_id = uuid.uuid4().hex
log.info(f'instance_id = {instance_id}')

app = web.Application()
routes = web.RouteTableDef()
aiohttp_jinja2.setup(app, loader=jinja2.PackageLoader('batch', 'templates'))

db = BatchDatabase.create_synchronous(os.environ.get('CLOUD_SQL_CONFIG_PATH',
                                                     '/batch-secrets/batch-production-cloud-sql-config.json'))


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
    def copy_task(job_id, task_name, files):
        if files is not None:
            authenticate = 'set -ex; gcloud -q auth activate-service-account --key-file=/gsa-key/key.json'

            def copy_command(src, dst):
                if not dst.startswith('gs://'):
                    mkdirs = f'mkdir -p {os.path.dirname(dst)};'
                else:
                    mkdirs = ""
                return f'{mkdirs} gsutil -m cp -R {src} {dst}'

            copies = ' && '.join([copy_command(src, dst) for (src, dst) in files])
            sh_expression = f'{authenticate} && {copies}'
            container = kube.client.V1Container(
                image='google/cloud-sdk:237.0.0-alpine',
                name=task_name,
                command=['/bin/sh', '-c', sh_expression])
            spec = kube.client.V1PodSpec(
                containers=[container],
                restart_policy='Never')
            return JobTask(job_id, task_name, spec)
        return None

    def __init__(self, job_id, name, pod_spec):
        assert pod_spec is not None

        metadata = kube.client.V1ObjectMeta(generate_name='job-{}-{}-'.format(job_id, name),
                                            labels={
                                                'app': 'batch-job',
                                                'hail.is/batch-instance': instance_id,
                                                'uuid': uuid.uuid4().hex
                                            })

        self.pod_template = kube.client.V1Pod(metadata=metadata,
                                              spec=pod_spec)
        self.name = name


class Job:
    def _next_task(self):
        self._task_idx += 1
        if self._task_idx < len(self._tasks):
            self._current_task = self._tasks[self._task_idx]

    def _has_next_task(self):
        return self._task_idx < len(self._tasks)

    def _create_pvc(self):
        pvc = v1.create_namespaced_persistent_volume_claim(
            HAIL_POD_NAMESPACE,
            kube.client.V1PersistentVolumeClaim(
                metadata=kube.client.V1ObjectMeta(
                    generate_name=f'job-{self.id}-',
                    labels={'app': 'batch-job',
                            'hail.is/batch-instance': instance_id}),
                spec=kube.client.V1PersistentVolumeClaimSpec(
                    access_modes=['ReadWriteOnce'],
                    volume_mode='Filesystem',
                    resources=kube.client.V1ResourceRequirements(
                        requests={'storage': POD_VOLUME_SIZE}),
                    storage_class_name=STORAGE_CLASS_NAME)),
            _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
        log.info(f'created pvc name: {pvc.metadata.name} for job {self.id}')
        return pvc

    # may be called twice with the same _current_task
    def _create_pod(self):
        assert self._pod_name is None
        assert self._current_task is not None

        pod = v1.create_namespaced_pod(
            HAIL_POD_NAMESPACE,
            self._current_task.pod_template,
            _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
        self._pod_name = pod.metadata.name
        pod_name_job[self._pod_name] = self

        add_event({'message': f'created pod for job {self.id}, task {self._current_task.name}',
                   'command': f'{pod.spec.containers[0].command}'})

        log.info('created pod name: {} for job {}, task {}'.format(self._pod_name,
                                                                   self.id,
                                                                   self._current_task.name))

    def _delete_pvc(self):
        if self._pvc is not None:
            log.info(f'deleting persistent volume claim {self._pvc.metadata.name} in '
                     f'{self._pvc.metadata.namespace}')
            try:
                v1.delete_namespaced_persistent_volume_claim(
                    self._pvc.metadata.name,
                    HAIL_POD_NAMESPACE,
                    _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
            except kube.client.rest.ApiException as err:
                if err.status == 404:
                    log.info(f'persistent volume claim {self._pvc.metadata.name} in '
                             f'{self._pvc.metadata.namespace} is already deleted')
                    return
                raise

    def _delete_k8s_resources(self):
        self._delete_pvc()
        if self._pod_name is not None:
            try:
                v1.delete_namespaced_pod(
                    self._pod_name,
                    HAIL_POD_NAMESPACE,
                    _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
            except kube.client.rest.ApiException as err:
                if err.status == 404:
                    pass
                raise
            del pod_name_job[self._pod_name]
            self._pod_name = None

    async def _read_logs(self):
        logs = {jt.name: await read_gs_log_file(app['blocking_pool'], instance_id, self.id, jt.name)
                for idx, jt in enumerate(self._tasks) if idx < self._task_idx}
        if self._state == 'Ready':
            if self._pod_name:
                try:
                    log = v1.read_namespaced_pod_log(
                        self._pod_name,
                        HAIL_POD_NAMESPACE,
                        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)
                    logs[self._current_task.name] = log
                except kube.client.rest.ApiException:
                    pass
            return logs
        if self._state == 'Complete':
            return logs
        assert self._state == 'Cancelled' or self._state == 'Created'
        return None

    def __init__(self, pod_spec, batch_id, attributes, callback, parent_ids,
                 scratch_folder, input_files, output_files, userdata, always_run):
        self.id = next_id()
        self.batch_id = batch_id
        self.attributes = attributes
        self.callback = callback
        self.child_ids = set([])
        self.parent_ids = parent_ids
        self.incomplete_parent_ids = set(self.parent_ids)
        self.scratch_folder = scratch_folder
        self.always_run = always_run
        self.userdata = userdata

        self._pvc = None
        self._pod_name = None
        self.exit_code = None
        self.duration = None
        self._state = 'Created'
        self._cancelled = False

        self._tasks = [JobTask.copy_task(self.id, 'input', input_files),
                       JobTask(self.id, 'main', pod_spec),
                       JobTask.copy_task(self.id, 'output', output_files)]

        self._tasks = [t for t in self._tasks if t is not None]

        for task in self._tasks:
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
                if self._pvc is None:
                    self._pvc = self._create_pvc()
                volumes.append(kube.client.V1Volume(
                    persistent_volume_claim=kube.client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=self._pvc.metadata.name),
                    name=self._pvc.metadata.name))
                volume_mounts.append(kube.client.V1VolumeMount(
                    mount_path='/io',
                    name=self._pvc.metadata.name))

            current_pod_spec = task.pod_template.spec
            if current_pod_spec.volumes is None:
                current_pod_spec.volumes = []
            current_pod_spec.volumes.extend(volumes)
            for container in current_pod_spec.containers:
                if container.volume_mounts is None:
                    container.volume_mounts = []
                container.volume_mounts.extend(volume_mounts)

        self._task_idx = -1
        self._next_task()
        assert self._current_task is not None

        job_id_job[self.id] = self

        for parent in self.parent_ids:
            job_id_job[parent].child_ids.add(self.id)

        if batch_id:
            batch = batch_id_batch[batch_id]
            batch.jobs.add(self)

        log.info('created job {}'.format(self.id))
        add_event({'message': f'created job {self.id}'})

        if not self.parent_ids:
            self.set_state('Ready')
            print("creating pod")
            self._create_pod()
        else:
            self.refresh_parents_and_maybe_create()

    # pylint incorrect error: https://github.com/PyCQA/pylint/issues/2047
    def refresh_parents_and_maybe_create(self):  # pylint: disable=invalid-name
        for parent in self.parent_ids:
            parent_job = job_id_job[parent]
            self.parent_new_state(parent_job._state, parent)

    def set_state(self, new_state):
        if self._state != new_state:
            log.info('job {} changed state: {} -> {}'.format(
                self.id,
                self._state,
                new_state))
            self._state = new_state
            self.notify_children(new_state)

    def notify_children(self, new_state):
        for child_id in self.child_ids:
            child = job_id_job.get(child_id)
            if child:
                child.parent_new_state(new_state, self.id)
            else:
                log.info(f'missing child: {child_id}')

    def parent_new_state(self, new_state, parent_id):
        if new_state in ('Cancelled', 'Complete') and parent_id in self.incomplete_parent_ids:
            self.incomplete_parent_ids.remove(parent_id)
            self.create_if_ready()

    def create_if_ready(self):
        if self._state == 'Created' and not self.incomplete_parent_ids:
            if (self.always_run or
                    (all(job_id_job[pid].is_successful() for pid in self.parent_ids) and not self._cancelled)):
                log.info(f'all parents complete for {self.id},'
                         f' creating pod')
                self.set_state('Ready')
                self._create_pod()
            else:
                log.info(f'parents deleted, cancelled, or failed: cancelling {self.id}')
                self.set_state('Cancelled')

    def cancel(self):
        # Cancelled, Complete
        if self.is_complete():
            return
        if self._state == 'Created':
            self._cancelled = True
        else:
            assert self._state == 'Ready', self._state
            self._delete_k8s_resources()
            self.set_state('Cancelled')

    async def delete(self):
        for cid in self.child_ids:
            child = job_id_job[cid]
            child.cancel()

        # remove from structures
        del job_id_job[self.id]
        if self.batch_id:
            batch = batch_id_batch[self.batch_id]
            batch.remove(self)

        for pid in self.parent_ids:
            parent = job_id_job[pid]
            parent.child_ids.remove(self.id)
        self.parent_ids = []
        children = [job_id_job[cid] for cid in self.child_ids]
        for child in children:
            child.parent_ids.remove(self.id)
            child.incomplete_parent_ids.discard(self.id)
        self.child_ids = set()

        for child in children:
            child.create_if_ready()

        self._delete_k8s_resources()
        self._state = 'Cancelled'

        for idx, jt in enumerate(self._tasks):
            if idx < self._task_idx:
                await delete_gs_log_file(app['blocking_pool'], instance_id, self.id, jt.name)

        log.info(f'job {self.id} deleted')

    def is_complete(self):
        return self._state in ('Complete', 'Cancelled')

    def is_successful(self):
        return self._state == 'Complete' and self.exit_code == 0

    def mark_unscheduled(self):
        if self._pod_name:
            del pod_name_job[self._pod_name]
            self._pod_name = None
        self._create_pod()

    async def mark_complete(self, pod):
        task_name = self._current_task.name

        terminated = pod.status.container_statuses[0].state.terminated
        self.exit_code = terminated.exit_code
        self.duration = (terminated.finished_at - terminated.started_at).total_seconds()

        pod_log = v1.read_namespaced_pod_log(
            pod.metadata.name,
            HAIL_POD_NAMESPACE,
            _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)

        add_event({'message': f'job {self.id}, {task_name} task exited', 'log': pod_log[:64000]})

        await write_gs_log_file(app['blocking_pool'], instance_id, self.id, task_name, pod_log)

        if self._pod_name:
            del pod_name_job[self._pod_name]
            self._pod_name = None

        self._next_task()
        if self.exit_code == 0:
            if self._has_next_task():
                self._create_pod()
                return
            self._delete_pvc()
        else:
            self._delete_pvc()

        log.info('job {} complete, exit_code {}'.format(self.id, self.exit_code))

        self.set_state('Complete')

        if self.callback:
            def handler(id, callback, json):
                try:
                    requests.post(callback, json=json, timeout=120)
                except requests.exceptions.RequestException as exc:
                    log.warning(
                        f'callback for job {id} failed due to an error, I will not retry. '
                        f'Error: {exc}')

            threading.Thread(target=handler, args=(self.id, self.callback, await self.to_dict())).start()

        if self.batch_id:
            await batch_id_batch[self.batch_id].mark_job_complete(self)

    async def to_dict(self):
        result = {
            'id': self.id,
            'state': self._state
        }
        if self._state == 'Complete':
            result['exit_code'] = self.exit_code
            result['duration'] = self.duration

        logs = await self._read_logs()
        if logs is not None:
            result['log'] = logs

        if self.attributes:
            result['attributes'] = self.attributes
        if self.parent_ids:
            result['parent_ids'] = self.parent_ids
        if self.scratch_folder:
            result['scratch_folder'] = self.scratch_folder
        return result


with open(os.environ.get('HAIL_JWT_SECRET_KEY_FILE', '/jwt-secret/secret-key')) as f:
    jwtclient = hj.JWTClient(f.read())


def authenticated_users_only(fun):
    def wrapped(request, *args, **kwargs):
        encoded_token = request.cookies.get('user')
        if encoded_token is not None:
            try:
                userdata = jwtclient.decode(encoded_token)
                if 'userdata' in fun.__code__.co_varnames:
                    return fun(request, *args, userdata=userdata, **kwargs)
                return fun(request, *args, **kwargs)
            except jwt.exceptions.DecodeError as de:
                log.info(f'could not decode token: {de}')
        raise web.HTTPUnauthorized(headers={'WWW-Authenticate': 'Bearer'})
    wrapped.__name__ = fun.__name__
    return wrapped


@routes.post('/jobs/create')
@authenticated_users_only
async def create_job(request, userdata):  # pylint: disable=R0912
    parameters = await request.json()

    schema = {
        # will be validated when creating pod
        'spec': schemas.pod_spec,
        'batch_id': {'type': 'integer'},
        'parent_ids': {'type': 'list', 'schema': {'type': 'integer'}},
        'scratch_folder': {'type': 'string'},
        'input_files': {
            'type': 'list',
            'schema': {'type': 'list', 'items': 2 * ({'type': 'string'},)}},
        'output_files': {
            'type': 'list',
            'schema': {'type': 'list', 'items': 2 * ({'type': 'string'},)}},
        'always_run': {'type': 'boolean'},
        'attributes': {
            'type': 'dict',
            'keyschema': {'type': 'string'},
            'valueschema': {'type': 'string'}
        },
        'callback': {'type': 'string'}
    }
    validator = cerberus.Validator(schema)
    if not validator.validate(parameters):
        abort(400, 'invalid request: {}'.format(validator.errors))

    pod_spec = v1.api_client._ApiClient__deserialize(
        parameters['spec'], kube.client.V1PodSpec)

    batch_id = parameters.get('batch_id')
    if batch_id:
        batch = batch_id_batch.get(batch_id)
        if batch is None:
            abort(404, f'invalid request: batch_id {batch_id} not found')
        if not batch.is_open:
            abort(400, f'invalid request: batch_id {batch_id} is closed')

    parent_ids = parameters.get('parent_ids', [])
    for parent_id in parent_ids:
        parent_job = job_id_job.get(parent_id, None)
        if parent_job is None:
            abort(400, f'invalid parent_id: no job with id {parent_id}')
        if parent_job.batch_id != batch_id or parent_job.batch_id is None or batch_id is None:
            abort(400,
                  f'invalid parent batch: {parent_id} is in batch '
                  f'{parent_job.batch_id} but child is in {batch_id}')

    scratch_folder = parameters.get('scratch_folder')
    input_files = parameters.get('input_files')
    output_files = parameters.get('output_files')
    always_run = parameters.get('always_run', False)

    if len(pod_spec.containers) != 1:
        abort(400, f'only one container allowed in pod_spec {pod_spec}')

    if pod_spec.containers[0].name != 'main':
        abort(400, f'container name must be "main" was {pod_spec.containers[0].name}')

    job = Job(
        pod_spec=pod_spec,
        batch_id=batch_id,
        attributes=parameters.get('attributes'),
        callback=parameters.get('callback'),
        parent_ids=parent_ids,
        scratch_folder=scratch_folder,
        input_files=input_files,
        output_files=output_files,
        userdata=userdata,
        always_run=always_run)
    return jsonify(await job.to_dict())


@routes.get('/alive')
async def get_alive(request):  # pylint: disable=W0613
    return jsonify({})


@routes.get('/jobs')
@authenticated_users_only
async def get_job_list(request):
    params = request.query

    jobs = job_id_job.values()
    for name, value in params.items():
        if name == 'complete':
            if value not in ('0', '1'):
                abort(400, f'invalid complete value, expected 0 or 1, got {value}')
            c = value == '1'
            jobs = [job for job in jobs if job.is_complete() == c]
        elif name == 'success':
            if value not in ('0', '1'):
                abort(400, f'invalid success value, expected 0 or 1, got {value}')
            s = value == '1'
            jobs = [job for job in jobs if (job._state == 'Complete' and job.exit_code == 0) == s]
        else:
            if not name.startswith('a:'):
                abort(400, f'unknown query parameter {name}')
            k = name[2:]
            jobs = [job for job in jobs
                    if job.attributes and k in job.attributes and job.attributes[k] == value]

    return jsonify([await job.to_dict() for job in jobs])


@routes.get('/jobs/{job_id}')
@authenticated_users_only
async def get_job(request):
    job_id = int(request.match_info['job_id'])
    job = job_id_job.get(job_id)
    if not job:
        abort(404)
    return jsonify(await job.to_dict())


@routes.get('/jobs/{job_id}/log')
@authenticated_users_only
async def get_job_log(request):  # pylint: disable=R1710
    job_id = int(request.match_info['job_id'])
    if job_id > max_id():
        abort(404)

    job = job_id_job.get(job_id)
    if job:
        job_log = await job._read_logs()
        if job_log:
            return jsonify(job_log)
    else:
        logs = {}
        for task_name in ['input', 'main', 'output']:
            log = await read_gs_log_file(app['blocking_pool'], instance_id, job_id, task_name)
            if log is not None:
                logs[task_name] = log
        if logs:
            return jsonify(logs)
    abort(404)


@routes.delete('/jobs/{job_id}/delete')
@authenticated_users_only
async def delete_job(request):
    job_id = int(request.match_info['job_id'])
    job = job_id_job.get(job_id)
    if not job:
        abort(404)
    await job.delete()
    return jsonify({})


@routes.patch('/jobs/{job_id}/cancel')
@authenticated_users_only
async def cancel_job(request):
    job_id = int(request.match_info['job_id'])
    job = job_id_job.get(job_id)
    if not job:
        abort(404)
    job.cancel()
    return jsonify({})


class Batch:
    MAX_TTL = 30 * 60

    def __init__(self, attributes, callback, ttl):
        self.attributes = attributes
        self.callback = callback
        self.id = next_id()
        batch_id_batch[self.id] = self
        self.jobs = set()
        self.is_open = True
        if ttl is None or ttl > Batch.MAX_TTL:
            ttl = Batch.MAX_TTL
        self.ttl = ttl
        schedule(self.ttl, self.close)

    def cancel(self):
        for j in self.jobs:
            j.cancel()

    def delete(self):
        del batch_id_batch[self.id]
        for j in self.jobs:
            assert j.batch_id == self.id
            j.batch_id = None

    def remove(self, job):
        self.jobs.remove(job)

    async def mark_job_complete(self, job):
        assert job in self.jobs
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
                args=(self.id, job.id, self.callback, await job.to_dict())
            ).start()

    def close(self):
        if self.is_open:
            log.info(f'closing batch {self.id}, ttl was {self.ttl}')
            self.is_open = False
        else:
            log.info(f're-closing batch {self.id}, ttl was {self.ttl}')

    def is_complete(self):
        return all(j.is_complete() for j in self.jobs)

    def is_successful(self):
        return all(j.is_successful() for j in self.jobs)

    async def to_dict(self):
        result = {
            'id': self.id,
            'jobs': sorted([await j.to_dict() for j in self.jobs], key=lambda j: j['id']),
            'is_open': self.is_open
        }
        if self.attributes:
            result['attributes'] = self.attributes
        return result


@routes.get('/batches')
async def get_batches_list(request):
    params = request.query

    batches = batch_id_batch.values()
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

    return jsonify([await batch.to_dict() for batch in batches])


@routes.post('/batches/create')
@authenticated_users_only
async def create_batch(request):
    parameters = await request.json()

    schema = {
        'attributes': {
            'type': 'dict',
            'keyschema': {'type': 'string'},
            'valueschema': {'type': 'string'}
        },
        'callback': {'type': 'string'},
        'ttl': {'type': 'number'}
    }
    validator = cerberus.Validator(schema)
    if not validator.validate(parameters):
        abort(400, 'invalid request: {}'.format(validator.errors))

    batch = Batch(parameters.get('attributes'), parameters.get('callback'), parameters.get('ttl'))
    return jsonify(await batch.to_dict())


@routes.get('/batches/{batch_id}')
@authenticated_users_only
async def get_batch(request):
    batch_id = int(request.match_info['batch_id'])
    batch = batch_id_batch.get(batch_id)
    if not batch:
        abort(404)
    return jsonify(await batch.to_dict())


@routes.patch('/batches/{batch_id}/cancel')
async def cancel_batch(request):
    batch_id = int(request.match_info['batch_id'])
    batch = batch_id_batch.get(batch_id)
    if not batch:
        abort(404)
    batch.cancel()
    return jsonify({})


@routes.delete('/batches/{batch_id}/delete')
@authenticated_users_only
async def delete_batch(request):
    batch_id = int(request.match_info['batch_id'])
    batch = batch_id_batch.get(batch_id)
    if not batch:
        abort(404)
    batch.delete()
    return jsonify({})


@routes.patch('/batches/{batch_id}/close')
@authenticated_users_only
async def close_batch(request):
    batch_id = int(request.match_info['batch_id'])
    batch = batch_id_batch.get(batch_id)
    if not batch:
        abort(404)
    batch.close()
    return jsonify({})


async def update_job_with_pod(job, pod):
    if pod:
        if pod.status.container_statuses:
            assert len(pod.status.container_statuses) == 1
            container_status = pod.status.container_statuses[0]
            assert container_status.name in ['input', 'main', 'output']

            if container_status.state and container_status.state.terminated:
                await job.mark_complete(pod)
    else:
        job.mark_unscheduled()


@routes.get('/recent')
@aiohttp_jinja2.template('recent.html')
@authenticated_users_only
async def recent(request):  # pylint: disable=W0613
    recent_events = get_recent_events()
    return {'recent': list(reversed(recent_events))}


class DeblockedIterator:
    def __init__(self, it):
        self.it = it

    def __aiter__(self):
        return self

    def __anext__(self):
        return blocking_to_async(app['blocking_pool'], self.it.__next__)


async def pod_changed(pod):
    job = pod_name_job.get(pod.metadata.name)

    if job and not job.is_complete():
        await update_job_with_pod(job, pod)


async def kube_event_loop():
    stream = kube.watch.Watch().stream(
        v1.list_namespaced_pod,
        HAIL_POD_NAMESPACE,
        label_selector=f'app=batch-job,hail.is/batch-instance={instance_id}')
    async for event in DeblockedIterator(stream):
        await pod_changed(event['object'])


async def refresh_k8s_state():  # pylint: disable=W0613
    log.info('started k8s state refresh')

    pods = await blocking_to_async(
        app['blocking_pool'],
        v1.list_namespaced_pod,
        HAIL_POD_NAMESPACE,
        label_selector=f'app=batch-job,hail.is/batch-instance={instance_id}',
        _request_timeout=KUBERNETES_TIMEOUT_IN_SECONDS)

    seen_pods = set()
    for pod in pods.items:
        pod_name = pod.metadata.name
        seen_pods.add(pod_name)

        job = pod_name_job.get(pod_name)
        if job and not job.is_complete():
            await update_job_with_pod(job, pod)

    for pod_name, job in pod_name_job.items():
        if pod_name not in seen_pods:
            await update_job_with_pod(job, None)

    log.info('k8s state refresh complete')


async def polling_event_loop():
    await asyncio.sleep(1)
    while True:
        try:
            await refresh_k8s_state()
        except Exception as exc:  # pylint: disable=W0703
            log.exception(f'Could not poll due to exception: {exc}')
        await asyncio.sleep(REFRESH_INTERVAL_IN_SECONDS)


def serve(port=5000):
    def if_anyone_dies_we_all_die(loop, context):
        try:
            loop.default_exception_handler(context)
        finally:
            loop.stop()
    asyncio.get_event_loop().set_exception_handler(
        if_anyone_dies_we_all_die)
    app.add_routes(routes)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        app['blocking_pool'] = pool
        asyncio.ensure_future(polling_event_loop())
        asyncio.ensure_future(kube_event_loop())
        web.run_app(app, host='0.0.0.0', port=port)
