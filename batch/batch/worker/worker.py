from typing import Optional
import os
import json
import sys
import re
import logging
import asyncio
import random
import traceback
import base64
import uuid
import shutil
import signal
import aiohttp
import aiohttp.client_exceptions
from aiohttp import web
import async_timeout
import concurrent
import aiodocker
from aiodocker.exceptions import DockerError
import google.oauth2.service_account
from hailtop.utils import (time_msecs, request_retry_transient_errors,
                           sleep_and_backoff, retry_all_errors, check_shell,
                           CalledProcessError, check_shell_output, is_google_registry_image,
                           find_spark_home)
from hailtop.httpx import client_session
from hailtop.batch_client.parse import (parse_cpu_in_mcpu, parse_image_tag,
                                        parse_memory_in_bytes, parse_storage_in_bytes)
from hailtop.batch.hail_genetics_images import HAIL_GENETICS, HAIL_GENETICS_IMAGES
from hailtop import aiotools
# import uvloop

from hailtop.config import DeployConfig
from hailtop.hail_logging import configure_logging

from ..utils import (adjust_cores_for_memory_request, cores_mcpu_to_memory_bytes,
                     adjust_cores_for_packability, adjust_cores_for_storage_request,
                     cores_mcpu_to_storage_bytes, storage_gib_to_bytes)
from ..semaphore import FIFOWeightedSemaphore
from ..log_store import LogStore
from ..globals import HTTP_CLIENT_MAX_SIZE, STATUS_FORMAT_VERSION
from ..batch_format_version import BatchFormatVersion
from ..worker_config import WorkerConfig
from ..public_gcr_images import public_gcr_images

from .flock import Flock

# uvloop.install()

configure_logging()
log = logging.getLogger('batch-worker')

MAX_DOCKER_IMAGE_PULL_SECS = 20 * 60
MAX_DOCKER_WAIT_SECS = 5 * 60
MAX_DOCKER_OTHER_OPERATION_SECS = 1 * 60

CORES = int(os.environ['CORES'])
NAME = os.environ['NAME']
NAMESPACE = os.environ['NAMESPACE']
# ACTIVATION_TOKEN
IP_ADDRESS = os.environ['IP_ADDRESS']
BATCH_LOGS_BUCKET_NAME = os.environ['BATCH_LOGS_BUCKET_NAME']
INSTANCE_ID = os.environ['INSTANCE_ID']
PROJECT = os.environ['PROJECT']
DOCKER_PREFIX = os.environ['DOCKER_PREFIX']
PUBLIC_GCR_IMAGES = public_gcr_images(DOCKER_PREFIX)
WORKER_CONFIG = json.loads(base64.b64decode(os.environ['WORKER_CONFIG']).decode())
MAX_IDLE_TIME_MSECS = int(os.environ['MAX_IDLE_TIME_MSECS'])
WORKER_DATA_DISK_MOUNT = os.environ['WORKER_DATA_DISK_MOUNT']
BATCH_WORKER_IMAGE = os.environ['BATCH_WORKER_IMAGE']

log.info(f'CORES {CORES}')
log.info(f'NAME {NAME}')
log.info(f'NAMESPACE {NAMESPACE}')
# ACTIVATION_TOKEN
log.info(f'IP_ADDRESS {IP_ADDRESS}')
log.info(f'BATCH_LOGS_BUCKET_NAME {BATCH_LOGS_BUCKET_NAME}')
log.info(f'INSTANCE_ID {INSTANCE_ID}')
log.info(f'PROJECT {PROJECT}')
log.info(f'DOCKER_PREFIX {DOCKER_PREFIX}')
log.info(f'WORKER_CONFIG {WORKER_CONFIG}')
log.info(f'MAX_IDLE_TIME_MSECS {MAX_IDLE_TIME_MSECS}')
log.info(f'WORKER_DATA_DISK_MOUNT {WORKER_DATA_DISK_MOUNT}')

worker_config = WorkerConfig(WORKER_CONFIG)
assert worker_config.cores == CORES

deploy_config = DeployConfig('gce', NAMESPACE, {})

docker: Optional[aiodocker.Docker] = None

port_allocator: Optional['PortAllocator'] = None

worker: Optional['Worker'] = None


class PortAllocator:
    def __init__(self):
        self.ports = asyncio.Queue()
        port_base = 46572
        for port in range(port_base, port_base + 10):
            self.ports.put_nowait(port)

    async def allocate(self):
        return await self.ports.get()

    def free(self, port):
        self.ports.put_nowait(port)


def docker_call_retry(timeout, name):
    async def wrapper(f, *args, **kwargs):
        delay = 0.1
        while True:
            try:
                return await asyncio.wait_for(f(*args, **kwargs), timeout)
            except DockerError as e:
                # 408 request timeout, 503 service unavailable
                if e.status == 408 or e.status == 503:
                    log.warning(f'in docker call to {f.__name__} for {name}, retrying', stack_info=True, exc_info=True)
                # DockerError(500, 'Get https://registry-1.docker.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
                # DockerError(500, 'error creating overlay mount to /var/lib/docker/overlay2/545a1337742e0292d9ed197b06fe900146c85ab06e468843cd0461c3f34df50d/merged: device or resource busy'
                # DockerError(500, 'Get https://gcr.io/v2/: dial tcp: lookup gcr.io: Temporary failure in name resolution')
                elif e.status == 500 and ("request canceled while waiting for connection" in e.message
                                          or re.match("error creating overlay mount.*device or resource busy", e.message)
                                          or "Temporary failure in name resolution" in e.message):
                    log.warning(f'in docker call to {f.__name__} for {name}, retrying', stack_info=True, exc_info=True)
                else:
                    raise
            except (aiohttp.client_exceptions.ServerDisconnectedError, asyncio.TimeoutError):
                log.warning(f'in docker call to {f.__name__} for {name}, retrying', stack_info=True, exc_info=True)
                delay = await sleep_and_backoff(delay)
    return wrapper


async def create_container(config, name):
    delay = 0.1
    error = 0

    async def handle_error(e):
        nonlocal error, delay
        error += 1
        if error < 10:
            delay = await sleep_and_backoff(delay)
            return
        raise ValueError('encountered {error} failures in create_container; aborting') from e

    while True:
        try:
            return await docker.containers.create(config, name=name)
        except DockerError as e:
            # 409 container with name already exists
            if e.status == 409:
                try:
                    delay = await sleep_and_backoff(delay)
                    return await docker.containers.get(name)
                except DockerError as eget:
                    # 404 No such container
                    if eget.status == 404:
                        await handle_error(eget)
                        continue
            # No such image: gcr.io/...
            if e.status == 404 and 'No such image' in e.message:
                await handle_error(e)
                continue
            raise


async def start_container(container):
    try:
        return await container.start()
    except DockerError as e:
        # 304 container has already started
        if e.status == 304:
            return
        if e.status == 500 and e.message == 'OCI runtime start failed: container process is already dead: unknown':
            log.info(f'restarting container {container}')
            return await container.restart()
        raise


async def stop_container(container):
    try:
        return await container.stop()
    except DockerError as e:
        # 304 container has already stopped
        if e.status == 304:
            return
        raise


async def delete_container(container, *args, **kwargs):
    try:
        return await container.delete(*args, **kwargs)
    except DockerError as e:
        # 404 container does not exist
        # 409 removal of container is already in progress
        if e.status in (404, 409):
            return
        raise


class JobDeletedError(Exception):
    pass


class JobTimeoutError(Exception):
    pass


class ContainerStepManager:
    def __init__(self, container, name, state):
        self.container = container
        self.state = state
        self.name = name
        self.timing = None
        self._deleted = False

    async def __aenter__(self):
        if self.container.job.deleted:
            self._deleted = True
            raise JobDeletedError()
        if self.state:
            log.info(f'{self.container} state changed: {self.container.state} => {self.state}')
            self.container.state = self.state
        self.timing = {}
        self.timing['start_time'] = time_msecs()
        self.container.timing[self.name] = self.timing

    async def __aexit__(self, exc_type, exc, tb):
        if self._deleted:
            return

        finish_time = time_msecs()
        self.timing['finish_time'] = finish_time
        start_time = self.timing['start_time']
        self.timing['duration'] = finish_time - start_time


def worker_fraction_in_1024ths(cpu_in_mcpu):
    return 1024 * cpu_in_mcpu // (CORES * 1000)


def user_error(e):
    if isinstance(e, DockerError):
        if e.status == 404 and 'pull access denied' in e.message:
            return True
        if e.status == 400 and 'executable file not found' in e.message:
            return True
    return False


class Container:
    def __init__(self, job, name, spec):
        self.job = job
        self.name = name
        self.spec = spec

        repository, tag = parse_image_tag(self.spec['image'])

        if not tag:
            log.info(f'adding latest tag to image {self.spec["image"]} for {self}')
            tag = 'latest'

        if repository in HAIL_GENETICS_IMAGES:
            repository_name_without_prefix = repository[len(HAIL_GENETICS):]
            repository = f'{DOCKER_PREFIX}/{repository_name_without_prefix}'

        self.repository = repository
        self.tag = tag
        self.image = self.repository + ':' + self.tag

        self.port = self.spec.get('port')
        self.host_port = None

        self.timeout = self.spec.get('timeout')

        self.container = None
        self.state = 'pending'
        self.error = None
        self.timing = {}
        self.container_status = None
        self.log = None
        self.overlay_path = None

    def container_config(self):
        weight = worker_fraction_in_1024ths(self.spec['cpu'])
        host_config = {
            'CpuShares': weight,
            'Memory': self.spec['memory'],
            'BlkioWeight': min(weight, 1000)
        }

        config = {
            "AttachStdin": False,
            "AttachStdout": False,
            "AttachStderr": False,
            "Tty": False,
            'OpenStdin': False,
            'Cmd': self.spec['command'],
            'Image': self.image,
            'Entrypoint': ''
        }

        env = self.spec.get('env', [])

        if self.port is not None:
            assert self.host_port is not None
            config['ExposedPorts'] = {
                f'{self.port}/tcp': {}
            }
            host_config['PortBindings'] = {
                f'{self.port}/tcp': [{
                    'HostIp': '',
                    'HostPort': str(self.host_port)
                }]
            }
            env = list(env)
            env.append(f'HAIL_BATCH_WORKER_PORT={self.host_port}')
            env.append(f'HAIL_BATCH_WORKER_IP={IP_ADDRESS}')

        volume_mounts = self.spec.get('volume_mounts')
        if volume_mounts:
            host_config['Binds'] = volume_mounts

        if env:
            config['Env'] = env

        network = self.spec.get('network')
        if network is None:
            network = 'public'
        host_config['NetworkMode'] = network  # not documented, I used strace to inspect the packets

        config['HostConfig'] = host_config

        return config

    def step(self, name, **kwargs):
        state = kwargs.get('state', name)
        return ContainerStepManager(self, name, state)

    async def get_container_status(self):
        if not self.container:
            return None

        try:
            c = await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(self.container.show)
        except DockerError as e:
            if e.status == 404:
                return None
            raise

        cstate = c['State']
        status = {
            'state': cstate['Status'],
            'started_at': cstate['StartedAt'],
            'finished_at': cstate['FinishedAt'],
            'out_of_memory': cstate['OOMKilled']
        }
        cerror = cstate['Error']
        if cerror:
            status['error'] = cerror
        else:
            status['exit_code'] = cstate['ExitCode']

        return status

    async def ensure_image_is_pulled(self, auth=None):
        try:
            await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(
                docker.images.get, self.image)
        except DockerError as e:
            if e.status == 404:
                await docker_call_retry(MAX_DOCKER_IMAGE_PULL_SECS, f'{self}')(
                    docker.images.pull, self.image, auth=auth)

    def current_user_access_token(self):
        key = base64.b64decode(self.job.gsa_key['key.json']).decode()
        return {'username': '_json_key', 'password': key}

    async def batch_worker_access_token(self):
        async with aiohttp.ClientSession(raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with await request_retry_transient_errors(
                    session, 'POST',
                    'http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token',
                    headers={'Metadata-Flavor': 'Google'}) as resp:
                access_token = (await resp.json())['access_token']
                return {'username': 'oauth2accesstoken', 'password': access_token}

    async def run(self, worker):
        try:
            async with self.step('pulling'):
                is_gcr_image = is_google_registry_image(self.image)
                is_public_gcr_image = self.repository in PUBLIC_GCR_IMAGES

                if not is_gcr_image:
                    await self.ensure_image_is_pulled()
                elif is_public_gcr_image:
                    auth = await self.batch_worker_access_token()
                    await self.ensure_image_is_pulled(auth=auth)
                else:
                    # Pull to verify this user has access to this
                    # image.
                    # FIXME improve the performance of this with a
                    # per-user image cache.
                    auth = self.current_user_access_token()
                    await docker_call_retry(MAX_DOCKER_IMAGE_PULL_SECS, f'{self}')(
                        docker.images.pull, self.image, auth=auth)

            if self.port is not None:
                async with self.step('allocating_port'):
                    self.host_port = await port_allocator.allocate()

            async with self.step('creating'):
                config = self.container_config()
                log.info(f'starting {self}')
                self.container = await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(
                    create_container, config, name=f'batch-{self.job.batch_id}-job-{self.job.job_id}-{self.name}')

            c = await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(self.container.show)

            merged_overlay_path = c['GraphDriver']['Data']['MergedDir']
            assert merged_overlay_path.endswith('/merged')
            self.overlay_path = merged_overlay_path[:-7].replace(WORKER_DATA_DISK_MOUNT, '/host')
            os.makedirs(f'{self.overlay_path}/', exist_ok=True)

            async with Flock('/xfsquota/projects', pool=worker.pool):
                with open('/xfsquota/projects', 'a') as f:
                    f.write(f'{self.job.project_id}:{self.overlay_path}\n')

            await check_shell_output(f'xfs_quota -x -D /xfsquota/projects -P /xfsquota/projid -c "project -s {self.job.project_name}" /host/')

            async with self.step('starting'):
                await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(
                    start_container, self.container)

            timed_out = False
            async with self.step('running'):
                try:
                    async with async_timeout.timeout(self.timeout):
                        await docker_call_retry(MAX_DOCKER_WAIT_SECS, f'{self}')(self.container.wait)
                except asyncio.TimeoutError:
                    timed_out = True

            self.container_status = await self.get_container_status()

            async with self.step('uploading_log'):
                await worker.log_store.write_log_file(
                    self.job.format_version, self.job.batch_id,
                    self.job.job_id, self.job.attempt_id, self.name,
                    await self.get_container_log())

            async with self.step('deleting'):
                await self.delete_container()

            if timed_out:
                raise JobTimeoutError(f'timed out after {self.timeout}s')

            if 'error' in self.container_status:
                self.state = 'error'
            elif self.container_status['exit_code'] == 0:
                self.state = 'succeeded'
            else:
                self.state = 'failed'
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not isinstance(e, (JobDeletedError, JobTimeoutError)):
                log.exception(f'while running {self}')

            self.state = 'error'
            self.error = traceback.format_exc()
        finally:
            await self.delete_container()

    async def get_container_log(self):
        logs = await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(
            self.container.log, stderr=True, stdout=True)
        self.log = "".join(logs)
        return self.log

    async def get_log(self):
        if self.container:
            return await self.get_container_log()
        return self.log

    async def delete_container(self):
        if self.overlay_path:
            path = self.overlay_path.replace('/', r'\/')
            async with Flock('/xfsquota/projects', pool=worker.pool):
                await check_shell(f"sed -i '/:{path}/d' /xfsquota/projects")

        if self.container:
            try:
                log.info(f'{self}: deleting container')
                await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(
                    stop_container, self.container)
                # v=True deletes anonymous volumes created by the container
                await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(
                    delete_container, self.container, v=True)
                self.container = None
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning('while deleting container, ignoring', exc_info=True)

        if self.host_port is not None:
            port_allocator.free(self.host_port)
            self.host_port = None

    async def delete(self):
        log.info(f'deleting {self}')
        await self.delete_container()

    # {
    #   name: str,
    #   state: str, (pending, pulling, creating, starting, running, uploading_log, deleting, suceeded, error, failed)
    #   timing: dict(str, float),
    #   error: str, (optional)
    #   container_status: { (from docker container state)
    #     state: str,
    #     started_at: str, (date)
    #     finished_at: str, (date)
    #     out_of_memory: boolean
    #     error: str, (one of error, exit_code will be present)
    #     exit_code: int
    #   }
    # }
    async def status(self, state=None):
        if not state:
            state = self.state
        status = {
            'name': self.name,
            'state': state,
            'timing': self.timing
        }
        if self.error:
            status['error'] = self.error
        if self.container_status:
            status['container_status'] = self.container_status
        elif self.container:
            status['container_status'] = await self.get_container_status()
        return status

    def __str__(self):
        return f'container {self.job.id}/{self.name}'


class JVMProcess:
    classpath = f'{find_spark_home()}/jars/*:/hail.jar:/log4j.properties'
    stack_size = 512 * 1024
    thread_pool = None

    def __init__(self, job, main_spec):
        self.job = job
        self.heap_size = main_spec['memory'] - self.stack_size
        self.env = {}
        for var in main_spec.get('env', []):
            self.env[var['name']] = var['value']

        self.flags = ['-classpath', self.classpath, f'-Xmx{self.heap_size}', f'-Xss{self.stack_size}']
        self.java_args = main_spec['command']

        self.proc = None
        self.timing = {'running': dict()}
        self.state = 'pending'
        self.logbuffer = bytearray()

    async def pipe_to_log(self, strm: asyncio.StreamReader):
        while not strm.at_eof():
            self.logbuffer.extend(await strm.readline())

    async def run(self, worker):
        log.info(f'running {self}')
        self.timing['running']['start_time'] = time_msecs()
        self.proc = await asyncio.create_subprocess_exec(
            'java',
            *self.flags,
            *self.java_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env)

        await asyncio.gather(self.pipe_to_log(self.proc.stdout),
                             self.pipe_to_log(self.proc.stderr))
        await self.proc.wait()

        finish_time = time_msecs()
        self.timing['running']['finish_time'] = finish_time
        start_time = self.timing['running']['start_time']
        self.timing['running']['duration'] = finish_time - start_time

        log.info(f'finished {self} with return code {self.proc.returncode}')
        log.info(f'log {self}: {self.get_log()}')

        await worker.log_store.write_log_file(
            self.job.format_version, self.job.batch_id,
            self.job.job_id, self.job.attempt_id, 'main',
            self.get_log())

        if self.proc.returncode == 0:
            self.state = 'succeeded'
        else:
            self.state = 'failed'

    async def status(self, state=None):
        d = {
            'name': 'main',
            'state': self.state if not state else state,
            'timing': self.timing
        }
        if self.proc is not None and self.proc.returncode is not None:
            d['exit_code'] = self.proc.returncode
        return d

    def get_log(self):
        return self.logbuffer.decode()

    async def delete(self):
        log.info(f'deleting {self}')
        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()

    def __str__(self):
        return f'process {self.job.id}/main'


def populate_secret_host_path(host_path, secret_data):
    os.makedirs(host_path)
    if secret_data is not None:
        for filename, data in secret_data.items():
            with open(f'{host_path}/{filename}', 'wb') as f:
                f.write(base64.b64decode(data))


async def add_gcsfuse_bucket(mount_path, bucket, key_file, read_only):
    os.makedirs(mount_path)
    options = ['allow_other']
    if read_only:
        options.append('ro')

    delay = 0.1
    error = 0
    while True:
        try:
            return await check_shell(f'''
/usr/bin/gcsfuse \
    -o {",".join(options)} \
    --file-mode 770 \
    --dir-mode 770 \
    --implicit-dirs \
    --key-file {key_file} \
    {bucket} {mount_path}
''')
        except CalledProcessError:
            error += 1
            if error == 5:
                raise

        delay = await sleep_and_backoff(delay)


def copy_container(job, name, files, volume_mounts, cpu, memory, requester_pays_project):
    assert files
    copy_spec = {
        'image': BATCH_WORKER_IMAGE,
        'name': name,
        'command': [
            '/usr/local/bin/python3',
            '-m',
            'batch.copy',
            json.dumps(requester_pays_project),
            json.dumps(files)
        ],
        'env': ['GOOGLE_APPLICATION_CREDENTIALS=/gsa-key/key.json'],
        'cpu': cpu,
        'memory': memory,
        'volume_mounts': volume_mounts
    }
    return Container(job, name, copy_spec)


class Job:
    quota_project_id = 100

    @staticmethod
    def get_next_xfsquota_project_id():
        project_id = Job.quota_project_id
        Job.quota_project_id += 1
        return project_id

    def secret_host_path(self, secret):
        return f'{self.scratch}/secrets/{secret["name"]}'

    def io_host_path(self):
        return f'{self.scratch}/io'

    def gcsfuse_path(self, bucket):
        # Make sure this path isn't in self.scratch to avoid accidental bucket deletions!
        return f'/gcsfuse/{self.token}/{bucket}'

    def gsa_key_file_path(self):
        return f'{self.scratch}/gsa-key'

    @classmethod
    def create(cls, batch_id, user, gsa_key, job_spec, format_version, task_manager):
        type = job_spec['process']['type']
        if type == 'docker':
            return DockerJob(batch_id, user, gsa_key, job_spec, format_version, task_manager)
        assert type == 'jvm'
        return JVMJob(batch_id, user, gsa_key, job_spec, format_version, task_manager)

    def __init__(self,
                 batch_id: int,
                 user: str,
                 gsa_key,
                 job_spec,
                 format_version,
                 task_manager: aiotools.BackgroundTaskManager):
        self.batch_id = batch_id
        self.user = user
        self.gsa_key = gsa_key
        self.job_spec = job_spec
        self.format_version = format_version

        self.deleted = False

        self.token = uuid.uuid4().hex
        self.scratch = f'/batch/{self.token}'

        self.state = 'pending'
        self.error = None

        self.start_time = None
        self.end_time = None

        self.input_volume_mounts = []
        self.main_volume_mounts = []
        self.output_volume_mounts = []

        io_volume_mount = f'{self.io_host_path()}:/io'
        self.input_volume_mounts.append(io_volume_mount)
        self.main_volume_mounts.append(io_volume_mount)
        self.output_volume_mounts.append(io_volume_mount)

        gcsfuse = job_spec.get('gcsfuse')
        self.gcsfuse = gcsfuse
        if gcsfuse:
            for b in gcsfuse:
                self.main_volume_mounts.append(f'{self.gcsfuse_path(b["bucket"])}:{b["mount_path"]}:shared')

        secrets = job_spec.get('secrets')
        self.secrets = secrets
        self.env = job_spec.get('env', [])

        req_cpu_in_mcpu = parse_cpu_in_mcpu(job_spec['resources']['cpu'])
        req_memory_in_bytes = parse_memory_in_bytes(job_spec['resources']['memory'])
        req_storage_in_bytes = parse_storage_in_bytes(job_spec['resources']['storage'])

        if worker_config.job_private:
            cpu_in_mcpu = CORES * 1000
            storage_in_bytes = storage_gib_to_bytes(worker_config.data_disk_size_gb)
        else:
            cpu_in_mcpu = adjust_cores_for_memory_request(req_cpu_in_mcpu, req_memory_in_bytes, worker_config.instance_type)
            cpu_in_mcpu = adjust_cores_for_storage_request(cpu_in_mcpu, req_storage_in_bytes, CORES, worker_config.local_ssd_data_disk, worker_config.data_disk_size_gb)
            cpu_in_mcpu = adjust_cores_for_packability(cpu_in_mcpu)
            storage_in_bytes = cores_mcpu_to_storage_bytes(cpu_in_mcpu, CORES, worker_config.local_ssd_data_disk, worker_config.data_disk_size_gb)

        self.cpu_in_mcpu = cpu_in_mcpu
        self.memory_in_bytes = cores_mcpu_to_memory_bytes(self.cpu_in_mcpu, worker_config.instance_type)
        self.storage_in_bytes = storage_in_bytes

        self.resources = worker_config.resources(self.cpu_in_mcpu, self.memory_in_bytes)

        self.project_name = f'batch-{self.batch_id}-job-{self.job_id}'
        self.project_id = Job.get_next_xfsquota_project_id()

        self.task_manager = task_manager

    @property
    def job_id(self):
        return self.job_spec['job_id']

    @property
    def attempt_id(self):
        return self.job_spec['attempt_id']

    @property
    def id(self):
        return (self.batch_id, self.job_id)

    async def run(self, worker):
        pass

    async def get_log(self):
        pass

    async def delete(self):
        log.info(f'deleting {self}')
        self.deleted = True

    # {
    #   version: int,
    #   worker: str,
    #   batch_id: int,
    #   job_id: int,
    #   attempt_id: int,
    #   user: str,
    #   state: str, (pending, initializing, running, succeeded, error, failed)
    #   format_version: int
    #   error: str, (optional)
    #   container_statuses: [Container.status],
    #   start_time: int,
    #   end_time: int,
    #   resources: list of dict, {name: str, quantity: int}
    # }
    async def status(self):
        status = {
            'version': STATUS_FORMAT_VERSION,
            'worker': NAME,
            'batch_id': self.batch_id,
            'job_id': self.job_spec['job_id'],
            'attempt_id': self.job_spec['attempt_id'],
            'user': self.user,
            'state': self.state,
            'format_version': self.format_version.format_version,
            'resources': self.resources
        }
        if self.error:
            status['error'] = self.error

        status['start_time'] = self.start_time
        status['end_time'] = self.end_time

        return status

    def __str__(self):
        return f'job {self.id}'


class DockerJob(Job):
    def __init__(self,
                 batch_id: int,
                 user: str,
                 gsa_key,
                 job_spec,
                 format_version,
                 task_manager: aiotools.BackgroundTaskManager):
        super().__init__(batch_id, user, gsa_key, job_spec, format_version, task_manager)
        input_files = job_spec.get('input_files')
        output_files = job_spec.get('output_files')

        requester_pays_project = job_spec.get('requester_pays_project')

        if job_spec['process'].get('mount_docker_socket'):
            self.main_volume_mounts.append('/var/run/docker.sock:/var/run/docker.sock')

        if self.secrets:
            for secret in self.secrets:
                volume_mount = f'{self.secret_host_path(secret)}:{secret["mount_path"]}'
                self.main_volume_mounts.append(volume_mount)
                # this will be the user gsa-key
                if secret.get('mount_in_copy', False):
                    self.input_volume_mounts.append(volume_mount)
                    self.output_volume_mounts.append(volume_mount)

        # create containers
        containers = {}

        if input_files:
            containers['input'] = copy_container(
                self, 'input', input_files, self.input_volume_mounts,
                self.cpu_in_mcpu, self.memory_in_bytes, requester_pays_project)

        # main container
        main_spec = {
            'command': job_spec['process']['command'],
            'image': job_spec['process']['image'],
            'name': 'main',
            'env': [f'{var["name"]}={var["value"]}' for var in self.env],
            'cpu': self.cpu_in_mcpu,
            'memory': self.memory_in_bytes,
            'volume_mounts': self.main_volume_mounts
        }
        port = job_spec.get('port')
        if port:
            main_spec['port'] = port
        timeout = job_spec.get('timeout')
        if timeout:
            main_spec['timeout'] = timeout
        network = job_spec.get('network')
        if network:
            assert network in ('public', 'private')
            main_spec['network'] = network
        containers['main'] = Container(self, 'main', main_spec)

        if output_files:
            containers['output'] = copy_container(
                self, 'output', output_files, self.output_volume_mounts,
                self.cpu_in_mcpu, self.memory_in_bytes, requester_pays_project)

        self.containers = containers

    async def run(self, worker):
        async with worker.cpu_sem(self.cpu_in_mcpu):
            self.start_time = time_msecs()

            try:
                self.task_manager.ensure_future(worker.post_job_started(self))

                log.info(f'{self}: initializing')
                self.state = 'initializing'

                os.makedirs(f'{self.scratch}/')

                async with Flock('/xfsquota/projid', pool=worker.pool):
                    with open('/xfsquota/projid', 'a') as f:
                        f.write(f'{self.project_name}:{self.project_id}\n')

                async with Flock('/xfsquota/projects', pool=worker.pool):
                    with open('/xfsquota/projects', 'a') as f:
                        f.write(f'{self.project_id}:{self.scratch}\n')

                await check_shell_output(f'xfs_quota -x -D /xfsquota/projects -P /xfsquota/projid -c "project -s {self.project_name}" /host/')
                await check_shell(f'xfs_quota -x -D /xfsquota/projects -P /xfsquota/projid -c "limit -p bsoft={self.storage_in_bytes} bhard={self.storage_in_bytes} {self.project_name}" /host/')

                os.makedirs(self.io_host_path())

                if self.secrets:
                    for secret in self.secrets:
                        populate_secret_host_path(self.secret_host_path(secret), secret['data'])

                if self.gcsfuse:
                    populate_secret_host_path(self.gsa_key_file_path(), self.gsa_key)
                    for b in self.gcsfuse:
                        bucket = b['bucket']
                        await add_gcsfuse_bucket(mount_path=self.gcsfuse_path(bucket),
                                                 bucket=bucket,
                                                 key_file=f'{self.gsa_key_file_path()}/key.json',
                                                 read_only=b['read_only'])

                self.state = 'running'

                input = self.containers.get('input')
                if input:
                    log.info(f'{self}: running input')
                    await input.run(worker)
                    log.info(f'{self} input: {input.state}')

                if not input or input.state == 'succeeded':
                    log.info(f'{self}: running main')

                    main = self.containers['main']
                    await main.run(worker)

                    log.info(f'{self} main: {main.state}')

                    output = self.containers.get('output')
                    if output:
                        log.info(f'{self}: running output')
                        await output.run(worker)
                        log.info(f'{self} output: {output.state}')

                    if main.state != 'succeeded':
                        self.state = main.state
                    elif output:
                        self.state = output.state
                    else:
                        self.state = 'succeeded'
                else:
                    self.state = input.state
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not user_error(e):
                    log.exception(f'while running {self}')

                self.state = 'error'
                self.error = traceback.format_exc()
                await self.cleanup()
            await self.cleanup()

    async def cleanup(self):
        self.end_time = time_msecs()

        if not self.deleted:
            log.info(f'{self}: marking complete')
            self.task_manager.ensure_future(worker.post_job_complete(self))

        log.info(f'{self}: cleaning up')
        try:
            if self.gcsfuse:
                for b in self.gcsfuse:
                    bucket = b['bucket']
                    mount_path = self.gcsfuse_path(bucket)
                    await check_shell(f'fusermount -u {mount_path}')
                    log.info(f'unmounted gcsfuse bucket {bucket} from {mount_path}')

            await check_shell(f'xfs_quota -x -D /xfsquota/projects -P /xfsquota/projid -c "limit -p bsoft=0 bhard=0 {self.project_name}" /host')

            async with Flock('/xfsquota/projid', pool=worker.pool):
                await check_shell(f"sed -i '/{self.project_name}:{self.project_id}/d' /xfsquota/projid")

            async with Flock('/xfsquota/projects', pool=worker.pool):
                await check_shell(f"sed -i '/{self.project_id}:/d' /xfsquota/projects")

            shutil.rmtree(self.scratch, ignore_errors=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('while deleting volumes')

    async def get_log(self):
        return {name: await c.get_log() for name, c in self.containers.items()}

    async def delete(self):
        await super().delete()
        for _, c in self.containers.items():
            await c.delete()

    async def status(self):
        status = await super().status()
        cstatuses = {
            name: await c.status() for name, c in self.containers.items()
        }
        status['container_statuses'] = cstatuses

        return status

    def __str__(self):
        return f'job {self.id}'


class JVMJob(Job):

    def secret_host_path(self, secret):
        return f'{self.scratch}/secrets{secret["mount_path"]}'

    def __init__(self,
                 batch_id: int,
                 user: str,
                 gsa_key,
                 job_spec,
                 format_version,
                 task_manager: aiotools.BackgroundTaskManager):
        super().__init__(batch_id, user, gsa_key, job_spec, format_version, task_manager)
        assert job_spec['process']['type'] == 'jvm'

        input_files = job_spec.get('input_files')
        output_files = job_spec.get('output_files')
        if input_files or output_files:
            raise Exception("i/o not supported")

        for envvar in self.env:
            assert envvar['name'] not in {'HAIL_DEPLOY_CONFIG_FILE', 'HAIL_TOKENS_FILE',
                                          'HAIL_SSL_CONFIG_DIR', 'HAIL_GSA_KEY_FILE',
                                          'HAIL_WORKER_SCRATCH_DIR'}, envvar

        self.env.append({'name': 'HAIL_DEPLOY_CONFIG_FILE',
                         'value': f'{self.scratch}/secrets/deploy-config/deploy-config.json'})
        self.env.append({'name': 'HAIL_TOKENS_FILE',
                         'value': f'{self.scratch}/secrets/user-tokens/tokens.json'})
        self.env.append({'name': 'HAIL_SSL_CONFIG_DIR',
                         'value': f'{self.scratch}/secrets/ssl-config'})
        self.env.append({'name': 'HAIL_GSA_KEY_FILE',
                         'value': f'{self.scratch}/secrets/gsa-key/key.json'})
        self.env.append({'name': 'HAIL_WORKER_SCRATCH_DIR', 'value': self.scratch})

        # main container
        self.main_spec = {
            'command': job_spec['process']['command'],  # ['is.hail.backend.service.Worker', $root, $i]
            'name': 'main',
            'env': self.env,
            'cpu': self.cpu_in_mcpu,
            'memory': self.memory_in_bytes,
        }
        self.process = JVMProcess(self, self.main_spec)

    async def run(self, worker):
        async with worker.cpu_sem(self.cpu_in_mcpu):
            self.start_time = time_msecs()

            try:
                self.task_manager.ensure_future(worker.post_job_started(self))

                log.info(f'{self}: initializing')
                self.state = 'initializing'

                os.makedirs(f'{self.scratch}/')

                async with Flock('/xfsquota/projid', pool=worker.pool):
                    with open('/xfsquota/projid', 'a') as f:
                        f.write(f'{self.project_name}:{self.project_id}\n')

                async with Flock('/xfsquota/projects', pool=worker.pool):
                    with open('/xfsquota/projects', 'a') as f:
                        f.write(f'{self.project_id}:{self.scratch}\n')

                await check_shell_output(f'xfs_quota -x -D /xfsquota/projects -P /xfsquota/projid -c "project -s {self.project_name}" /host/')
                await check_shell(f'xfs_quota -x -D /xfsquota/projects -P /xfsquota/projid -c "limit -p bsoft={self.storage_in_bytes} bhard={self.storage_in_bytes} {self.project_name}" /host/')

                if self.secrets:
                    for secret in self.secrets:
                        populate_secret_host_path(self.secret_host_path(secret), secret['data'])

                populate_secret_host_path(self.gsa_key_file_path(), self.gsa_key)
                self.state = 'running'

                log.info(f'{self}: running jvm process')

                await self.process.run(worker)
                self.state = self.process.state
                log.info(f'{self} main: {self.state}')
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(f'while running {self}')

                self.state = 'error'
                self.error = traceback.format_exc()
            # TODO Put into function call and call from except Exception and after try/except block
            finally:
                self.end_time = time_msecs()

                if not self.deleted:
                    log.info(f'{self}: marking complete')
                    self.task_manager.ensure_future(worker.post_job_complete(self))

                log.info(f'{self}: cleaning up')
                try:
                    await check_shell(f'xfs_quota -x -D /xfsquota/projects -P /xfsquota/projid -c "limit -p bsoft=0 bhard=0 {self.project_name}" /host')

                    async with Flock('/xfsquota/projid', pool=worker.pool):
                        await check_shell(f"sed     -i '/{self.project_name}:{self.project_id}/d' /xfsquota/projid")

                    async with Flock('/xfsquota/projects', pool=worker.pool):
                        await check_shell(f"sed -i '/{self.project_id}:/d' /xfsquota/projects")

                    shutil.rmtree(self.scratch, ignore_errors=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception('while deleting volumes')

    async def get_log(self):
        return {'main': await self.process.get_log()}

    async def delete(self):
        log.info(f'deleting {self}')
        self.deleted = True
        await self.process.delete()

    # {
    #   version: int,
    #   worker: str,
    #   batch_id: int,
    #   job_id: int,
    #   attempt_id: int,
    #   user: str,
    #   state: str, (pending, initializing, running, succeeded, error, failed)
    #   format_version: int
    #   error: str, (optional)
    #   container_statuses: [Container.status],
    #   start_time: int,
    #   end_time: int,
    #   resources: list of dict, {name: str, quantity: int}
    # }
    async def status(self):
        status = await super().status()
        status['container_statuses'] = {'main': await self.process.status()}
        return status

    def __str__(self):
        return f'job {self.id}'


class Worker:
    def __init__(self):
        self.cores_mcpu = CORES * 1000
        self.last_updated = time_msecs()
        self.cpu_sem = FIFOWeightedSemaphore(self.cores_mcpu)
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.jobs = {}
        self.stop_event = asyncio.Event()

        # filled in during activation
        self.log_store = None
        self.headers = None
        self.task_manager = aiotools.BackgroundTaskManager()

    def shutdown(self):
        self.task_manager.shutdown()

    async def run_job(self, job):
        try:
            await job.run(self)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(f'while running {job}, ignoring')

    async def create_job_1(self, request):
        body = await request.json()

        batch_id = body['batch_id']
        job_id = body['job_id']

        format_version = BatchFormatVersion(body['format_version'])

        if format_version.has_full_spec_in_gcs():
            token = body['token']
            start_job_id = body['start_job_id']
            addtl_spec = body['job_spec']

            job_spec = await self.log_store.read_spec_file(batch_id, token, start_job_id, job_id)
            job_spec = json.loads(job_spec)

            job_spec['attempt_id'] = addtl_spec['attempt_id']
            job_spec['secrets'] = addtl_spec['secrets']

            addtl_env = addtl_spec.get('env')
            if addtl_env:
                env = job_spec.get('env')
                if not env:
                    env = []
                    job_spec['env'] = env
                env.extend(addtl_env)
        else:
            job_spec = body['job_spec']

        assert job_spec['job_id'] == job_id
        id = (batch_id, job_id)

        # already running
        if id in self.jobs:
            return web.HTTPForbidden()

        job = Job.create(batch_id, body['user'], body['gsa_key'], job_spec, format_version, self.task_manager)

        log.info(f'created {job}, adding to jobs')

        self.jobs[job.id] = job

        self.task_manager.ensure_future(self.run_job(job))

        return web.Response()

    async def create_job(self, request):
        return await asyncio.shield(self.create_job_1(request))

    async def get_job_log(self, request):
        batch_id = int(request.match_info['batch_id'])
        job_id = int(request.match_info['job_id'])
        id = (batch_id, job_id)
        job = self.jobs.get(id)
        if not job:
            raise web.HTTPNotFound()
        return web.json_response(await job.get_log())

    async def get_job_status(self, request):
        batch_id = int(request.match_info['batch_id'])
        job_id = int(request.match_info['job_id'])
        id = (batch_id, job_id)
        job = self.jobs.get(id)
        if not job:
            raise web.HTTPNotFound()
        return web.json_response(await job.status())

    async def delete_job_1(self, request):
        batch_id = int(request.match_info['batch_id'])
        job_id = int(request.match_info['job_id'])
        id = (batch_id, job_id)

        log.info(f'deleting job {id}, removing from jobs')

        job = self.jobs.pop(id, None)
        if job is None:
            raise web.HTTPNotFound()

        self.last_updated = time_msecs()

        self.task_manager.ensure_future(job.delete())

        return web.Response()

    async def delete_job(self, request):
        return await asyncio.shield(self.delete_job_1(request))

    async def healthcheck(self, request):  # pylint: disable=unused-argument
        body = {'name': NAME}
        return web.json_response(body)

    async def run(self):
        app = web.Application(client_max_size=HTTP_CLIENT_MAX_SIZE)
        app.add_routes([
            web.post('/api/v1alpha/kill', self.kill),
            web.post('/api/v1alpha/batches/jobs/create', self.create_job),
            web.delete('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/delete', self.delete_job),
            web.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log', self.get_job_log),
            web.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/status', self.get_job_status),
            web.get('/healthcheck', self.healthcheck)
        ])
        try:
            await asyncio.wait_for(self.activate(), MAX_IDLE_TIME_MSECS / 1000)
        except asyncio.TimeoutError:
            log.exception(f'could not activate after trying for {MAX_IDLE_TIME_MSECS} ms, exiting')
            return

        app_runner = web.AppRunner(app)
        await app_runner.setup()
        site = web.TCPSite(app_runner, '0.0.0.0', 5000)
        await site.start()

        try:
            while True:
                try:
                    await asyncio.wait_for(self.stop_event.wait(), 15)
                    log.info('received stop event')
                    break
                except asyncio.TimeoutError:
                    idle_duration = time_msecs() - self.last_updated
                    if not self.jobs and idle_duration >= MAX_IDLE_TIME_MSECS:
                        log.info(f'idle {idle_duration} ms, exiting')
                        break
                    log.info(f'n_jobs {len(self.jobs)} free_cores {self.cpu_sem.value / 1000} idle {idle_duration}')
        finally:
            log.info('shutting down')
            await site.stop()
            log.info('stopped site')
            await app_runner.cleanup()
            log.info('cleaned up app runner')
            await self.deactivate()
            log.info('deactivated')

    async def deactivate(self):
        async with client_session() as session:
            # Don't retry.  If it doesn't go through, the driver
            # monitoring loops will recover.  If the driver is
            # gone (e.g. testing a PR), this would go into an
            # infinite loop and the instance won't be deleted.
            await session.post(
                deploy_config.url('batch-driver', '/api/v1alpha/instances/deactivate'),
                headers=self.headers)

    async def kill_1(self, request):  # pylint: disable=unused-argument
        log.info('killed')
        self.stop_event.set()

    async def kill(self, request):
        return await asyncio.shield(self.kill_1(request))

    async def post_job_complete_1(self, job):
        run_duration = job.end_time - job.start_time

        full_status = await retry_all_errors(f'error while getting status for {job}')(job.status)

        if job.format_version.has_full_status_in_gcs():
            await retry_all_errors(f'error while writing status file to gcs for {job}')(
                self.log_store.write_status_file,
                job.batch_id,
                job.job_id,
                job.attempt_id,
                json.dumps(full_status))

        db_status = job.format_version.db_status(full_status)

        status = {
            'version': full_status['version'],
            'batch_id': full_status['batch_id'],
            'job_id': full_status['job_id'],
            'attempt_id': full_status['attempt_id'],
            'state': full_status['state'],
            'start_time': full_status['start_time'],
            'end_time': full_status['end_time'],
            'resources': full_status['resources'],
            'status': db_status
        }

        body = {
            'status': status
        }

        start_time = time_msecs()
        delay_secs = 0.1
        while True:
            try:
                async with client_session() as session:
                    await session.post(
                        deploy_config.url('batch-driver', '/api/v1alpha/instances/job_complete'),
                        json=body, headers=self.headers)
                    return
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as e:
                if isinstance(e, aiohttp.ClientResponseError) and e.status == 404:   # pylint: disable=no-member
                    raise
                log.warning(f'failed to mark {job} complete, retrying', exc_info=True)

            # unlist job after 3m or half the run duration
            now = time_msecs()
            elapsed = now - start_time
            if (job.id in self.jobs
                    and elapsed > 180 * 1000
                    and elapsed > run_duration / 2):
                log.info(f'too much time elapsed marking {job} complete, removing from jobs, will keep retrying')
                del self.jobs[job.id]
                self.last_updated = time_msecs()

            await asyncio.sleep(
                delay_secs * random.uniform(0.7, 1.3))
            # exponentially back off, up to (expected) max of 2m
            delay_secs = min(delay_secs * 2, 2 * 60.0)

    async def post_job_complete(self, job):
        try:
            await self.post_job_complete_1(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(f'error while marking {job} complete', stack_info=True)
        finally:
            log.info(f'{job} marked complete, removing from jobs')
            if job.id in self.jobs:
                del self.jobs[job.id]
                self.last_updated = time_msecs()

    async def post_job_started_1(self, job):
        full_status = await job.status()

        status = {
            'version': full_status['version'],
            'batch_id': full_status['batch_id'],
            'job_id': full_status['job_id'],
            'attempt_id': full_status['attempt_id'],
            'start_time': full_status['start_time'],
            'resources': full_status['resources']
        }

        body = {
            'status': status
        }

        async with client_session() as session:
            await request_retry_transient_errors(
                session, 'POST',
                deploy_config.url('batch-driver', '/api/v1alpha/instances/job_started'),
                json=body, headers=self.headers)

    async def post_job_started(self, job):
        try:
            await self.post_job_started_1(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(f'error while posting {job} started')

    async def activate(self):
        async with client_session() as session:
            resp = await request_retry_transient_errors(
                session, 'GET',
                deploy_config.url('batch-driver', '/api/v1alpha/instances/gsa_key'),
                headers={
                    'X-Hail-Instance-Name': NAME,
                    'Authorization': f'Bearer {os.environ["ACTIVATION_TOKEN"]}'
                })
            resp_json = await resp.json()

            with open('key.json', 'w') as f:
                f.write(json.dumps(resp_json['key']))

            credentials = google.oauth2.service_account.Credentials.from_service_account_file(
                'key.json')
            self.log_store = LogStore(BATCH_LOGS_BUCKET_NAME, INSTANCE_ID, self.pool,
                                      project=PROJECT, credentials=credentials)

            resp = await request_retry_transient_errors(
                session, 'POST',
                deploy_config.url('batch-driver', '/api/v1alpha/instances/activate'),
                json={'ip_address': os.environ['IP_ADDRESS']},
                headers={
                    'X-Hail-Instance-Name': NAME,
                    'Authorization': f'Bearer {os.environ["ACTIVATION_TOKEN"]}'
                })
            resp_json = await resp.json()

            self.headers = {
                'X-Hail-Instance-Name': NAME,
                'Authorization': f'Bearer {resp_json["token"]}'
            }


async def async_main():
    global port_allocator, worker, docker

    docker = aiodocker.Docker()

    port_allocator = PortAllocator()
    worker = Worker()
    try:
        await worker.run()
    finally:
        try:
            worker.shutdown()
            log.info('worker shutdown')
        finally:
            await docker.close()
            log.info('docker closed')


def dump_all_stacktraces():
    for t in asyncio.all_tasks():
        print(t)
        t.print_stack()


loop = asyncio.get_event_loop()
loop.add_signal_handler(signal.SIGUSR1, dump_all_stacktraces)
loop.run_until_complete(async_main())
log.info('closing loop')
loop.close()
log.info('closed')
sys.exit(0)
