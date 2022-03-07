import asyncio
import base64
import concurrent
import errno
import json
import logging
import os
import random
import re
import shutil
import signal
import sys
import tempfile
import traceback
import uuid
import warnings
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from typing import Any, Awaitable, Callable, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union

import aiodocker  # type: ignore
import aiodocker.images
import aiohttp
import aiohttp.client_exceptions
import aiorwlock
import async_timeout
from aiodocker.exceptions import DockerError  # type: ignore
from aiohttp import web

from gear.clients import get_cloud_async_fs, get_compute_client
from hailtop import aiotools, httpx
from hailtop.aiotools import LocalAsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.batch.hail_genetics_images import HAIL_GENETICS_IMAGES
from hailtop.config import DeployConfig
from hailtop.hail_logging import configure_logging
from hailtop.utils import (
    CalledProcessError,
    blocking_to_async,
    check_exec_output,
    check_shell,
    check_shell_output,
    dump_all_stacktraces,
    find_spark_home,
    parse_docker_image_reference,
    periodically_call,
    request_retry_transient_errors,
    retry_transient_errors,
    sleep_and_backoff,
    time_msecs,
    time_msecs_str,
)

from ..batch_format_version import BatchFormatVersion
from ..cloud.azure.worker.worker_api import AzureWorkerAPI
from ..cloud.gcp.worker.worker_api import GCPWorkerAPI
from ..cloud.resource_utils import is_valid_storage_request, storage_gib_to_bytes
from ..file_store import FileStore
from ..globals import HTTP_CLIENT_MAX_SIZE, RESERVED_STORAGE_GB_PER_CORE, STATUS_FORMAT_VERSION
from ..publicly_available_images import publicly_available_images
from ..semaphore import FIFOWeightedSemaphore
from ..utils import Box
from ..worker.worker_api import CloudWorkerAPI
from .credentials import CloudUserCredentials
from .jvm_entryway_protocol import EndOfStream, read_bool, read_int, read_str, write_int, write_str

# import uvloop


# uvloop.install()

with open('/subdomains.txt', 'r', encoding='utf-8') as subdomains_file:
    HAIL_SERVICES = [line.rstrip() for line in subdomains_file.readlines()]

oldwarn = warnings.warn


def deeper_stack_level_warn(*args, **kwargs):
    if 'stacklevel' in kwargs:
        kwargs['stacklevel'] = max(kwargs['stacklevel'], 5)
    else:
        kwargs['stacklevel'] = 5
    return oldwarn(*args, **kwargs)


warnings.warn = deeper_stack_level_warn


def compose_auth_header_urlsafe(orig_f):
    def compose(auth: Union[MutableMapping, str, bytes], registry_addr: str = None):
        orig_auth_header = orig_f(auth, registry_addr=registry_addr)
        auth = json.loads(base64.b64decode(orig_auth_header))
        auth_json = json.dumps(auth).encode('ascii')
        return base64.urlsafe_b64encode(auth_json).decode('ascii')

    return compose


# We patched aiodocker's utility function `compose_auth_header` because it does not base64 encode strings
# in urlsafe mode which is required for Azure's credentials.
# https://github.com/aio-libs/aiodocker/blob/17e08844461664244ea78ecd08d1672b1779acc1/aiodocker/utils.py#L297
aiodocker.images.compose_auth_header = compose_auth_header_urlsafe(aiodocker.images.compose_auth_header)


configure_logging()
log = logging.getLogger('batch-worker')

MAX_DOCKER_IMAGE_PULL_SECS = 20 * 60
MAX_DOCKER_WAIT_SECS = 5 * 60
MAX_DOCKER_OTHER_OPERATION_SECS = 1 * 60

IPTABLES_WAIT_TIMEOUT_SECS = 60

CLOUD = os.environ['CLOUD']
CORES = int(os.environ['CORES'])
NAME = os.environ['NAME']
NAMESPACE = os.environ['NAMESPACE']
# ACTIVATION_TOKEN
IP_ADDRESS = os.environ['IP_ADDRESS']
INTERNAL_GATEWAY_IP = os.environ['INTERNAL_GATEWAY_IP']
BATCH_LOGS_STORAGE_URI = os.environ['BATCH_LOGS_STORAGE_URI']
INSTANCE_ID = os.environ['INSTANCE_ID']
DOCKER_PREFIX = os.environ['DOCKER_PREFIX']
PUBLIC_IMAGES = publicly_available_images(DOCKER_PREFIX)
INSTANCE_CONFIG = json.loads(base64.b64decode(os.environ['INSTANCE_CONFIG']).decode())
MAX_IDLE_TIME_MSECS = int(os.environ['MAX_IDLE_TIME_MSECS'])
BATCH_WORKER_IMAGE = os.environ['BATCH_WORKER_IMAGE']
BATCH_WORKER_IMAGE_ID = os.environ['BATCH_WORKER_IMAGE_ID']
INTERNET_INTERFACE = os.environ['INTERNET_INTERFACE']
UNRESERVED_WORKER_DATA_DISK_SIZE_GB = int(os.environ['UNRESERVED_WORKER_DATA_DISK_SIZE_GB'])
assert UNRESERVED_WORKER_DATA_DISK_SIZE_GB >= 0
ACCEPTABLE_QUERY_JAR_URL_PREFIX = os.environ['ACCEPTABLE_QUERY_JAR_URL_PREFIX']
assert len(ACCEPTABLE_QUERY_JAR_URL_PREFIX) > 3  # x:// where x is one or more characters

CLOUD_WORKER_API: CloudWorkerAPI = GCPWorkerAPI.from_env() if CLOUD == 'gcp' else AzureWorkerAPI.from_env()

log.info(f'CLOUD {CLOUD}')
log.info(f'CORES {CORES}')
log.info(f'NAME {NAME}')
log.info(f'NAMESPACE {NAMESPACE}')
# ACTIVATION_TOKEN
log.info(f'IP_ADDRESS {IP_ADDRESS}')
log.info(f'BATCH_LOGS_STORAGE_URI {BATCH_LOGS_STORAGE_URI}')
log.info(f'INSTANCE_ID {INSTANCE_ID}')
log.info(f'DOCKER_PREFIX {DOCKER_PREFIX}')
log.info(f'INSTANCE_CONFIG {INSTANCE_CONFIG}')
log.info(f'CLOUD_WORKER_API {CLOUD_WORKER_API}')
log.info(f'MAX_IDLE_TIME_MSECS {MAX_IDLE_TIME_MSECS}')
log.info(f'INTERNET_INTERFACE {INTERNET_INTERFACE}')
log.info(f'UNRESERVED_WORKER_DATA_DISK_SIZE_GB {UNRESERVED_WORKER_DATA_DISK_SIZE_GB}')
log.info(f'ACCEPTABLE_QUERY_JAR_URL_PREFIX {ACCEPTABLE_QUERY_JAR_URL_PREFIX}')

instance_config = CLOUD_WORKER_API.instance_config_from_config_dict(INSTANCE_CONFIG)
assert instance_config.cores == CORES
assert instance_config.cloud == CLOUD


N_SLOTS = 4 * CORES  # Jobs are allowed at minimum a quarter core

deploy_config = DeployConfig('gce', NAMESPACE, {})

docker: Optional[aiodocker.Docker] = None

port_allocator: Optional['PortAllocator'] = None
network_allocator: Optional['NetworkAllocator'] = None

worker: Optional['Worker'] = None

image_configs: Dict[str, Dict[str, Any]] = {}

image_lock = aiorwlock.RWLock()


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


class NetworkNamespace:
    def __init__(self, subnet_index: int, private: bool, internet_interface: str):
        assert subnet_index <= 255
        self.subnet_index = subnet_index
        self.private = private
        self.internet_interface = internet_interface
        self.network_ns_name = uuid.uuid4().hex[:5]
        self.hostname = 'hostname-' + uuid.uuid4().hex[:10]
        self.veth_host = self.network_ns_name + '-host'
        self.veth_job = self.network_ns_name + '-job'

        if private:
            self.host_ip = f'172.20.{subnet_index}.10'
            self.job_ip = f'172.20.{subnet_index}.11'
        else:
            self.host_ip = f'172.21.{subnet_index}.10'
            self.job_ip = f'172.21.{subnet_index}.11'

        self.port = None
        self.host_port = None

    async def init(self):
        await self.create_netns()
        await self.enable_iptables_forwarding()

        os.makedirs(f'/etc/netns/{self.network_ns_name}')
        with open(f'/etc/netns/{self.network_ns_name}/hosts', 'w', encoding='utf-8') as hosts:
            hosts.write('127.0.0.1 localhost\n')
            hosts.write(f'{self.job_ip} {self.hostname}\n')
            if NAMESPACE == 'default':
                for service in HAIL_SERVICES:
                    hosts.write(f'{INTERNAL_GATEWAY_IP} {service}.hail\n')
            hosts.write(f'{INTERNAL_GATEWAY_IP} internal.hail\n')

        # Jobs on the private network should have access to the metadata server
        # and our vdc. The public network should not so we use google's public
        # resolver.
        with open(f'/etc/netns/{self.network_ns_name}/resolv.conf', 'w', encoding='utf-8') as resolv:
            if self.private:
                resolv.write(f'nameserver {CLOUD_WORKER_API.nameserver_ip}\n')
                if CLOUD == 'gcp':
                    resolv.write('search c.hail-vdc.internal google.internal\n')
            else:
                resolv.write('nameserver 8.8.8.8\n')

    async def create_netns(self):
        await check_shell(
            f'''
ip netns add {self.network_ns_name} && \
ip link add name {self.veth_host} type veth peer name {self.veth_job} && \
ip link set dev {self.veth_host} up && \
ip link set {self.veth_job} netns {self.network_ns_name} && \
ip address add {self.host_ip}/24 dev {self.veth_host}
ip -n {self.network_ns_name} link set dev {self.veth_job} up && \
ip -n {self.network_ns_name} link set dev lo up && \
ip -n {self.network_ns_name} address add {self.job_ip}/24 dev {self.veth_job} && \
ip -n {self.network_ns_name} route add default via {self.host_ip}'''
        )

    async def enable_iptables_forwarding(self):
        await check_shell(
            f'''
iptables -w {IPTABLES_WAIT_TIMEOUT_SECS} --append FORWARD --out-interface {self.veth_host} --in-interface {self.internet_interface} --jump ACCEPT && \
iptables -w {IPTABLES_WAIT_TIMEOUT_SECS} --append FORWARD --out-interface {self.veth_host} --in-interface {self.veth_host} --jump ACCEPT'''
        )

    async def expose_port(self, port, host_port):
        self.port = port
        self.host_port = host_port
        await self.expose_port_rule(action='append')

    async def expose_port_rule(self, action: str):
        # Appending to PREROUTING means this is only exposed to external traffic.
        # To expose for locally created packets, we would append instead to the OUTPUT chain.
        await check_shell(
            f'iptables -w {IPTABLES_WAIT_TIMEOUT_SECS} --table nat --{action} PREROUTING '
            f'--match addrtype --dst-type LOCAL '
            f'--protocol tcp '
            f'--match tcp --dport {self.host_port} '
            f'--jump DNAT --to-destination {self.job_ip}:{self.port}'
        )

    async def cleanup(self):
        if self.host_port:
            assert self.port
            await self.expose_port_rule(action='delete')
        self.host_port = None
        self.port = None
        await check_shell(
            f'''
ip link delete {self.veth_host} && \
ip netns delete {self.network_ns_name}'''
        )
        await self.create_netns()


class NetworkAllocator:
    def __init__(self):
        self.private_networks = asyncio.Queue()
        self.public_networks = asyncio.Queue()
        self.internet_interface = INTERNET_INTERFACE

    async def reserve(self):
        for subnet_index in range(N_SLOTS):
            public = NetworkNamespace(subnet_index, private=False, internet_interface=self.internet_interface)
            await public.init()
            self.public_networks.put_nowait(public)

            private = NetworkNamespace(subnet_index, private=True, internet_interface=self.internet_interface)

            await private.init()
            self.private_networks.put_nowait(private)

    async def allocate_private(self) -> NetworkNamespace:
        return await self.private_networks.get()

    async def allocate_public(self) -> NetworkNamespace:
        return await self.public_networks.get()

    def free(self, netns: NetworkNamespace):
        asyncio.ensure_future(self._free(netns))

    async def _free(self, netns: NetworkNamespace):
        await netns.cleanup()
        if netns.private:
            self.private_networks.put_nowait(netns)
        else:
            self.public_networks.put_nowait(netns)


def docker_call_retry(timeout, name):
    async def wrapper(f, *args, **kwargs):
        delay = 0.1
        while True:
            try:
                return await asyncio.wait_for(f(*args, **kwargs), timeout)
            except DockerError as e:
                # 408 request timeout, 503 service unavailable
                if e.status in (408, 503):
                    log.warning(f'in docker call to {f.__name__} for {name}, retrying', stack_info=True, exc_info=True)
                # DockerError(500, 'Get https://registry-1.docker.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
                # DockerError(500, 'error creating overlay mount to /var/lib/docker/overlay2/545a1337742e0292d9ed197b06fe900146c85ab06e468843cd0461c3f34df50d/merged: device or resource busy'
                # DockerError(500, 'Get https://gcr.io/v2/: dial tcp: lookup gcr.io: Temporary failure in name resolution')
                elif e.status == 500 and (
                    "request canceled while waiting for connection" in e.message
                    or re.match("error creating overlay mount.*device or resource busy", e.message)
                    or "Temporary failure in name resolution" in e.message
                ):
                    log.warning(f'in docker call to {f.__name__} for {name}, retrying', stack_info=True, exc_info=True)
                else:
                    raise
            except (aiohttp.client_exceptions.ServerDisconnectedError, asyncio.TimeoutError):
                log.warning(f'in docker call to {f.__name__} for {name}, retrying', stack_info=True, exc_info=True)
                delay = await sleep_and_backoff(delay)

    return wrapper


async def send_signal_and_wait(proc, signal, timeout=None):
    try:
        if signal == 'SIGTERM':
            proc.terminate()
        else:
            assert signal == 'SIGKILL'
            proc.kill()
    except ProcessLookupError:
        pass
    else:
        await asyncio.wait_for(proc.wait(), timeout=timeout)


class JobDeletedError(Exception):
    pass


class JobTimeoutError(Exception):
    pass


class Timings:
    def __init__(self, is_deleted: Callable[[], bool]):
        self.timings: Dict[str, Dict[str, float]] = {}
        self.is_deleted = is_deleted

    def step(self, name: str, ignore_job_deletion: bool = False):
        assert name not in self.timings
        self.timings[name] = {}
        return ContainerStepManager(self.timings[name], self.is_deleted, ignore_job_deletion=ignore_job_deletion)

    def to_dict(self):
        return self.timings


class ContainerStepManager:
    def __init__(self, timing: Dict[str, float], is_deleted: Callable[[], bool], ignore_job_deletion: bool = False):
        self.timing: Dict[str, float] = timing
        self.is_deleted = is_deleted
        self.ignore_job_deletion = ignore_job_deletion

    def __enter__(self):
        if self.is_deleted() and not self.ignore_job_deletion:
            raise JobDeletedError
        self.timing['start_time'] = time_msecs()

    def __exit__(self, exc_type, exc, tb):
        if self.is_deleted() and not self.ignore_job_deletion:
            return
        finish_time = time_msecs()
        self.timing['finish_time'] = finish_time
        self.timing['duration'] = finish_time - self.timing['start_time']


def worker_fraction_in_1024ths(cpu_in_mcpu):
    return 1024 * cpu_in_mcpu // (CORES * 1000)


def user_error(e):
    if isinstance(e, DockerError):
        if e.status == 404 and 'pull access denied' in e.message:
            return True
        if e.status == 404 and ('not found: manifest unknown' in e.message or 'no such image' in e.message):
            return True
        if e.status == 400 and 'executable file not found' in e.message:
            return True
    if isinstance(e, CalledProcessError):
        # Opening GCS connection...\n', b'daemonize.Run: readFromProcess: sub-process: mountWithArgs: mountWithConn:
        # fs.NewServer: create file system: SetUpBucket: OpenBucket: Bad credentials for bucket "BUCKET". Check the
        # bucket name and your credentials.\n')
        if b'Bad credentials for bucket' in e.stderr:
            return True
    return False


class Container:
    def __init__(self, job, name, spec, client_session: httpx.ClientSession, worker: 'Worker'):
        self.job = job
        self.name = name
        self.spec = spec
        self.client_session = client_session
        self.worker = worker
        self.deleted_event = asyncio.Event()

        image_ref = parse_docker_image_reference(self.spec['image'])
        if image_ref.tag is None and image_ref.digest is None:
            log.info(f'adding latest tag to image {self.spec["image"]} for {self}')
            image_ref.tag = 'latest'

        if image_ref.name() in HAIL_GENETICS_IMAGES:
            # We want the "hailgenetics/python-dill" translate to (based on the prefix):
            # * gcr.io/hail-vdc/hailgenetics/python-dill
            # * us-central1-docker.pkg.dev/hail-vdc/hail/hailgenetics/python-dill
            image_ref.path = image_ref.name()
            image_ref.domain = DOCKER_PREFIX.split('/', maxsplit=1)[0]
            image_ref.path = '/'.join(DOCKER_PREFIX.split('/')[1:] + [image_ref.path])

        self.image_ref = image_ref
        self.image_ref_str = str(image_ref)
        self.image_id = None

        self.port = self.spec.get('port')
        self.host_port = None

        self.timeout = self.spec.get('timeout')

        self.state = 'pending'
        self.error = None
        self.short_error = None
        self.container_status = None
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None

        self.timings = Timings(self.is_job_deleted)

        self.logbuffer = bytearray()
        self.overlay_path = None

        self.image_config = None
        self.rootfs_path = None
        scratch = self.spec['scratch']
        self.container_scratch = f'{scratch}/{self.name}'
        self.container_overlay_path = f'{self.container_scratch}/rootfs_overlay'
        self.config_path = f'{self.container_scratch}/config'
        self.log_path = f'{self.container_scratch}/container.log'

        self.overlay_mounted = False

        self.container_name = f'batch-{self.job.batch_id}-job-{self.job.job_id}-{self.name}'

        self.netns: Optional[NetworkNamespace] = None
        # regarding no-member: https://github.com/PyCQA/pylint/issues/4223
        self.process: Optional[asyncio.subprocess.Process] = None  # pylint: disable=no-member

        assert self.worker.fs is not None

    async def run(self):
        try:

            async def localize_rootfs():
                async def _localize_rootfs():
                    async with image_lock.reader_lock:
                        # FIXME Authentication is entangled with pulling images. We need a way to test
                        # that a user has access to a cached image without pulling.
                        await self.pull_image()
                        self.image_config = image_configs[self.image_ref_str]
                        self.image_id = self.image_config['Id'].split(":")[1]
                        self.worker.image_data[self.image_id] += 1

                        self.rootfs_path = f'/host/rootfs/{self.image_id}'

                        image_data = self.worker.image_data[self.image_id]
                        async with image_data.lock:
                            if not image_data.extracted:
                                try:
                                    await self.extract_rootfs()
                                    image_data.extracted = True
                                    log.info(
                                        f'Added expanded image to cache: {self.image_ref_str}, ID: {self.image_id}'
                                    )
                                except asyncio.CancelledError:
                                    raise
                                except Exception:
                                    log.exception(f'while extracting image {self.image_ref_str}, ID: {self.image_id}')
                                    await blocking_to_async(worker.pool, shutil.rmtree, self.rootfs_path)

                await asyncio.shield(_localize_rootfs())

            with self.step('pulling'):
                await self.run_until_done_or_deleted(localize_rootfs)

            with self.step('setting up overlay'):
                await self.run_until_done_or_deleted(self.setup_overlay)

            with self.step('setting up network'):
                await self.run_until_done_or_deleted(self.setup_network_namespace)

            with self.step('running'):
                timed_out = await self.run_until_done_or_deleted(self.run_container)

            self.container_status = self.get_container_status()

            if timed_out:
                self.short_error = 'timed out'
                raise JobTimeoutError(f'timed out after {self.timeout}s')

            if self.container_status['exit_code'] == 0:
                self.state = 'succeeded'
            else:
                if self.container_status['out_of_memory']:
                    self.short_error = 'out of memory'
                self.state = 'failed'
        except asyncio.CancelledError:
            raise
        except JobDeletedError:
            self.state = 'cancelled'
        except Exception as e:
            if not isinstance(e, JobTimeoutError) and not user_error(e):
                log.exception(f'while running {self}')
            self.state = 'error'
            self.error = traceback.format_exc()
        finally:
            try:
                with self.step('uploading_log', ignore_job_deletion=True):
                    await self.upload_log()
            finally:
                try:
                    await self.delete_container()
                finally:
                    if self.image_id:
                        self.worker.image_data[self.image_id] -= 1

    async def run_until_done_or_deleted(self, f: Callable[[], Awaitable[Any]]):
        step = asyncio.ensure_future(f())
        deleted = asyncio.ensure_future(self.deleted_event.wait())
        try:
            await asyncio.wait([deleted, step], return_when=asyncio.FIRST_COMPLETED)
            if deleted.done():
                raise JobDeletedError
            assert step.done()
            return step.result()
        finally:
            for t in (step, deleted):
                if t.done():
                    e = t.exception()
                    if e and not user_error(e):
                        log.exception(e)
                else:
                    t.cancel()

    def is_job_deleted(self) -> bool:
        return self.job.deleted

    def step(self, name: str, ignore_job_deletion: bool = False):
        return self.timings.step(name, ignore_job_deletion=ignore_job_deletion)

    async def pull_image(self):
        is_cloud_image = (CLOUD == 'gcp' and self.image_ref.hosted_in('google')) or (
            CLOUD == 'azure' and self.image_ref.hosted_in('azure')
        )
        is_public_image = self.image_ref.name() in PUBLIC_IMAGES

        try:
            if not is_cloud_image:
                await self.ensure_image_is_pulled()
            elif is_public_image:
                auth = await self.batch_worker_access_token()
                await self.ensure_image_is_pulled(auth=auth)
            else:
                # Pull to verify this user has access to this
                # image.
                # FIXME improve the performance of this with a
                # per-user image cache.
                auth = self.current_user_access_token()
                await docker_call_retry(MAX_DOCKER_IMAGE_PULL_SECS, f'{self}')(
                    docker.images.pull, self.image_ref_str, auth=auth
                )
        except DockerError as e:
            if e.status == 404 and 'pull access denied' in e.message:
                self.short_error = 'image cannot be pulled'
            elif 'not found: manifest unknown' in e.message:
                self.short_error = 'image not found'
            raise

        image_config, _ = await check_exec_output('docker', 'inspect', self.image_ref_str)
        image_configs[self.image_ref_str] = json.loads(image_config)[0]

    async def ensure_image_is_pulled(self, auth=None):
        try:
            await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, f'{self}')(docker.images.get, self.image_ref_str)
        except DockerError as e:
            if e.status == 404:
                await docker_call_retry(MAX_DOCKER_IMAGE_PULL_SECS, f'{self}')(
                    docker.images.pull, self.image_ref_str, auth=auth
                )
            else:
                raise

    async def batch_worker_access_token(self):
        return await CLOUD_WORKER_API.worker_access_token(self.client_session)

    def current_user_access_token(self):
        return {'username': self.job.credentials.username, 'password': self.job.credentials.password}

    async def extract_rootfs(self):
        assert self.rootfs_path
        os.makedirs(self.rootfs_path)
        await check_shell(
            f'id=$(docker create {self.image_id}) && docker export $id | tar -C {self.rootfs_path} -xf - && docker rm $id'
        )

    async def setup_overlay(self):
        lower_dir = self.rootfs_path
        upper_dir = f'{self.container_overlay_path}/upper'
        work_dir = f'{self.container_overlay_path}/work'
        merged_dir = f'{self.container_overlay_path}/merged'
        for d in (upper_dir, work_dir, merged_dir):
            os.makedirs(d)
        await check_shell(
            f'mount -t overlay overlay -o lowerdir={lower_dir},upperdir={upper_dir},workdir={work_dir} {merged_dir}'
        )
        self.overlay_mounted = True

    async def setup_network_namespace(self):
        network = self.spec.get('network')
        if network is None or network is True:
            self.netns = await network_allocator.allocate_public()
        else:
            assert network == 'private'
            self.netns = await network_allocator.allocate_private()
        if self.port is not None:
            self.host_port = await port_allocator.allocate()
            await self.netns.expose_port(self.port, self.host_port)

    async def run_container(self) -> bool:
        self.started_at = time_msecs()
        try:
            await self.write_container_config()
            async with async_timeout.timeout(self.timeout):
                with open(self.log_path, 'w', encoding='utf-8') as container_log:
                    log.info(f'Creating the crun run process for {self}')
                    self.process = await asyncio.create_subprocess_exec(
                        'crun',
                        'run',
                        '--bundle',
                        f'{self.container_overlay_path}/merged',
                        '--config',
                        f'{self.config_path}/config.json',
                        self.container_name,
                        stdout=container_log,
                        stderr=container_log,
                    )
                    await self.process.wait()
                    log.info(f'crun process completed for {self}')
        except asyncio.TimeoutError:
            return True
        finally:
            self.finished_at = time_msecs()

        return False

    async def write_container_config(self):
        os.makedirs(self.config_path)
        with open(f'{self.config_path}/config.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(await self.container_config()))

    # https://github.com/opencontainers/runtime-spec/blob/master/config.md
    async def container_config(self):
        uid, gid = await self._get_in_container_user()
        weight = worker_fraction_in_1024ths(self.spec['cpu'])
        workdir = self.image_config['Config']['WorkingDir']
        default_docker_capabilities = [
            'CAP_CHOWN',
            'CAP_DAC_OVERRIDE',
            'CAP_FSETID',
            'CAP_FOWNER',
            'CAP_MKNOD',
            'CAP_NET_RAW',
            'CAP_SETGID',
            'CAP_SETUID',
            'CAP_SETFCAP',
            'CAP_SETPCAP',
            'CAP_NET_BIND_SERVICE',
            'CAP_SYS_CHROOT',
            'CAP_KILL',
            'CAP_AUDIT_WRITE',
        ]
        config = {
            'ociVersion': '1.0.1',
            'root': {
                'path': '.',
                'readonly': False,
            },
            'hostname': self.netns.hostname,
            'mounts': self._mounts(uid, gid),
            'process': {
                'user': {  # uid/gid *inside the container*
                    'uid': uid,
                    'gid': gid,
                },
                'args': self.spec['command'],
                'env': self._env(),
                'cwd': workdir if workdir != "" else "/",
                'capabilities': {
                    'bounding': default_docker_capabilities,
                    'effective': default_docker_capabilities,
                    'inheritable': default_docker_capabilities,
                    'permitted': default_docker_capabilities,
                },
            },
            'linux': {
                'namespaces': [
                    {'type': 'pid'},
                    {
                        'type': 'network',
                        'path': f'/var/run/netns/{self.netns.network_ns_name}',
                    },
                    {'type': 'mount'},
                    {'type': 'ipc'},
                    {'type': 'uts'},
                    {'type': 'cgroup'},
                ],
                'uidMappings': [],
                'gidMappings': [],
                'resources': {
                    'cpu': {'shares': weight},
                    'memory': {
                        'limit': self.spec['memory'],
                        'reservation': self.spec['memory'],
                    },
                    # 'blockIO': {'weight': min(weight, 1000)}, FIXME blkio.weight not supported
                },
                'maskedPaths': [
                    '/proc/asound',
                    '/proc/acpi',
                    '/proc/kcore',
                    '/proc/keys',
                    '/proc/latency_stats',
                    '/proc/timer_list',
                    '/proc/timer_stats',
                    '/proc/sched_debug',
                    '/proc/scsi',
                    '/sys/firmware',
                ],
                'readonlyPaths': [
                    '/proc/bus',
                    '/proc/fs',
                    '/proc/irq',
                    '/proc/sys',
                    '/proc/sysrq-trigger',
                ],
            },
        }

        if self.spec.get('unconfined'):
            config['linux']['maskedPaths'] = []
            config['linux']['readonlyPaths'] = []
            config['process']['apparmorProfile'] = 'unconfined'
            config['linux']['seccomp'] = {'defaultAction': "SCMP_ACT_ALLOW"}

        return config

    async def _get_in_container_user(self):
        user = self.image_config['Config']['User']
        if not user:
            uid, gid = 0, 0
        elif ":" in user:
            uid, gid = user.split(":")
        else:
            uid, gid = await self._read_user_from_rootfs(user)
        return int(uid), int(gid)

    async def _read_user_from_rootfs(self, user) -> Tuple[str, str]:
        with open(f'{self.rootfs_path}/etc/passwd', 'r', encoding='utf-8') as passwd:
            for record in passwd:
                if record.startswith(user):
                    _, _, uid, gid, _, _, _ = record.split(":")
                    return uid, gid
            raise ValueError("Container user not found in image's /etc/passwd")

    def _mounts(self, uid, gid):
        # Only supports empty volumes
        external_volumes = []
        volumes = self.image_config['Config']['Volumes']
        if volumes:
            for v_container_path in volumes:
                if not v_container_path.startswith('/'):
                    v_container_path = '/' + v_container_path
                v_host_path = f'{self.container_scratch}/volumes{v_container_path}'
                os.makedirs(v_host_path)
                if uid != 0 or gid != 0:
                    os.chown(v_host_path, uid, gid)
                external_volumes.append(
                    {
                        'source': v_host_path,
                        'destination': v_container_path,
                        'type': 'none',
                        'options': ['rbind', 'rw', 'shared'],
                    }
                )

        return (
            self.spec.get('volume_mounts')
            + external_volumes
            + [
                # Recommended filesystems:
                # https://github.com/opencontainers/runtime-spec/blob/master/config-linux.md#default-filesystems
                {
                    'source': 'proc',
                    'destination': '/proc',
                    'type': 'proc',
                    'options': ['nosuid', 'noexec', 'nodev'],
                },
                {
                    'source': 'tmpfs',
                    'destination': '/dev',
                    'type': 'tmpfs',
                    'options': ['nosuid', 'strictatime', 'mode=755', 'size=65536k'],
                },
                {
                    'source': 'sysfs',
                    'destination': '/sys',
                    'type': 'sysfs',
                    'options': ['nosuid', 'noexec', 'nodev', 'ro'],
                },
                {
                    'source': 'cgroup',
                    'destination': '/sys/fs/cgroup',
                    'type': 'cgroup',
                    'options': ['nosuid', 'noexec', 'nodev', 'ro'],
                },
                {
                    'source': 'devpts',
                    'destination': '/dev/pts',
                    'type': 'devpts',
                    'options': ['nosuid', 'noexec', 'nodev'],
                },
                {
                    'source': 'mqueue',
                    'destination': '/dev/mqueue',
                    'type': 'mqueue',
                    'options': ['nosuid', 'noexec', 'nodev'],
                },
                {
                    'source': 'shm',
                    'destination': '/dev/shm',
                    'type': 'tmpfs',
                    'options': ['nosuid', 'noexec', 'nodev', 'mode=1777', 'size=67108864'],
                },
                {
                    'source': f'/etc/netns/{self.netns.network_ns_name}/resolv.conf',
                    'destination': '/etc/resolv.conf',
                    'type': 'none',
                    'options': ['rbind', 'ro'],
                },
                {
                    'source': f'/etc/netns/{self.netns.network_ns_name}/hosts',
                    'destination': '/etc/hosts',
                    'type': 'none',
                    'options': ['rbind', 'ro'],
                },
            ]
        )

    def _env(self):
        env = self.image_config['Config']['Env'] + self.spec.get('env', [])
        if self.port is not None:
            assert self.host_port is not None
            env.append(f'HAIL_BATCH_WORKER_PORT={self.host_port}')
            env.append(f'HAIL_BATCH_WORKER_IP={IP_ADDRESS}')
        return env

    async def delete_container(self):
        if self.container_is_running():
            assert self.process is not None
            try:
                log.info(f'{self} container is still running, killing crun process')
                try:
                    await check_exec_output('crun', 'kill', '--all', self.container_name, 'SIGKILL')
                except CalledProcessError as e:
                    not_extant_message = (
                        b'error opening file `/run/crun/'
                        + self.container_name.encode()
                        + b'/status`: No such file or directory'
                    )
                    if not (e.returncode == 1 and not_extant_message in e.stderr):
                        log.exception(f'while deleting container {self}', exc_info=True)
            finally:
                try:
                    await send_signal_and_wait(self.process, 'SIGTERM', timeout=5)
                except asyncio.TimeoutError:
                    try:
                        await send_signal_and_wait(self.process, 'SIGKILL', timeout=5)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception(f'could not kill process for container {self}')
                finally:
                    self.process = None

        if self.overlay_mounted:
            try:
                await check_shell(f'umount -l {self.container_overlay_path}/merged')
                self.overlay_mounted = False
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(f'while unmounting overlay in {self}', exc_info=True)

        if self.host_port is not None:
            port_allocator.free(self.host_port)
            self.host_port = None

        if self.netns:
            network_allocator.free(self.netns)
            self.netns = None

    async def delete(self):
        log.info(f'deleting {self}')
        self.deleted_event.set()

    # {
    #   name: str,
    #   state: str, (pending, pulling, creating, starting, running, uploading_log, deleting, succeeded, error, failed)
    #   timing: dict(str, float),
    #   error: str, (optional)
    #   short_error: str, (optional)
    #   container_status: {
    #     state: str,
    #     started_at: int, (date)
    #     finished_at: int, (date)
    #     out_of_memory: bool,
    #     exit_code: int
    #   }
    # }
    def status(self):
        status = {'name': self.name, 'state': self.state, 'timing': self.timings.to_dict()}
        if self.error:
            status['error'] = self.error
        if self.short_error:
            status['short_error'] = self.short_error
        if self.container_status:
            status['container_status'] = self.container_status
        elif self.container_is_running():
            status['container_status'] = self.get_container_status()
        return status

    def get_container_status(self):
        if not self.process:
            return None

        status = {
            'started_at': self.started_at,
            'finished_at': self.finished_at,
        }
        if self.container_is_running():
            status['state'] = 'running'
            status['out_of_memory'] = False
        else:
            status['state'] = 'finished'
            status['exit_code'] = self.process.returncode
            status['out_of_memory'] = self.process.returncode == 137

        return status

    def container_is_running(self):
        return self.process is not None and self.process.returncode is None

    def container_finished(self):
        return self.process is not None and self.process.returncode is not None

    async def upload_log(self):
        await self.worker.file_store.write_log_file(
            self.job.format_version,
            self.job.batch_id,
            self.job.job_id,
            self.job.attempt_id,
            self.name,
            await self.get_log(),
        )

    async def get_log(self):
        if os.path.exists(self.log_path):
            return (await self.worker.fs.read(self.log_path)).decode()
        return ''

    def __str__(self):
        return f'container {self.job.id}/{self.name}'


def populate_secret_host_path(host_path: str, secret_data: Optional[Dict[str, bytes]]):
    os.makedirs(host_path, exist_ok=True)
    if secret_data is not None:
        for filename, data in secret_data.items():
            with open(f'{host_path}/{filename}', 'wb') as f:
                f.write(base64.b64decode(data))


def copy_container(
    job: 'Job',
    name: str,
    files,
    volume_mounts,
    cpu,
    memory,
    scratch: str,
    requester_pays_project: str,
    client_session: httpx.ClientSession,
    worker: 'Worker',
) -> Container:
    assert files
    copy_spec = {
        'image': BATCH_WORKER_IMAGE,
        'name': name,
        'command': [
            '/usr/bin/python3',
            '-m',
            'hailtop.aiotools.copy',
            json.dumps(requester_pays_project),
            json.dumps(files),
            '-v',
        ],
        'env': [f'{job.credentials.cloud_env_name}={job.credentials.mount_path}'],
        'cpu': cpu,
        'memory': memory,
        'scratch': scratch,
        'volume_mounts': volume_mounts,
    }
    return Container(job, name, copy_spec, client_session, worker)


class Job:
    quota_project_id = 100

    @staticmethod
    def get_next_xfsquota_project_id():
        project_id = Job.quota_project_id
        Job.quota_project_id += 1
        return project_id

    def secret_host_path(self, secret) -> str:
        return f'{self.scratch}/secrets/{secret["name"]}'

    def io_host_path(self) -> str:
        return f'{self.scratch}/io'

    def cloudfuse_base_path(self):
        # Make sure this path isn't in self.scratch to avoid accidental bucket deletions!
        path = f'/cloudfuse/{self.token}'
        assert os.path.commonpath([path, self.scratch]) == '/'
        return path

    def cloudfuse_data_path(self, bucket: str) -> str:
        # Make sure this path isn't in self.scratch to avoid accidental bucket deletions!
        path = f'{self.cloudfuse_base_path()}/{bucket}/data'
        assert os.path.commonpath([path, self.scratch]) == '/'
        return path

    def cloudfuse_tmp_path(self, bucket: str) -> str:
        # Make sure this path isn't in self.scratch to avoid accidental bucket deletions!
        path = f'{self.cloudfuse_base_path()}/{bucket}/tmp'
        assert os.path.commonpath([path, self.scratch]) == '/'
        return path

    def cloudfuse_credentials_path(self, bucket: str) -> str:
        return f'{self.scratch}/cloudfuse/{bucket}'

    def credentials_host_dirname(self) -> str:
        return f'{self.scratch}/{self.credentials.secret_name}'

    def credentials_host_file_path(self) -> str:
        return f'{self.credentials_host_dirname()}/{self.credentials.file_name}'

    @staticmethod
    def create(
        batch_id,
        user,
        credentials: CloudUserCredentials,
        job_spec: dict,
        format_version: BatchFormatVersion,
        task_manager: aiotools.BackgroundTaskManager,
        pool: concurrent.futures.ThreadPoolExecutor,
        client_session: httpx.ClientSession,
        worker: 'Worker',
    ) -> 'Job':
        type = job_spec['process']['type']
        if type == 'docker':
            return DockerJob(
                batch_id, user, credentials, job_spec, format_version, task_manager, pool, client_session, worker
            )
        assert type == 'jvm'
        return JVMJob(batch_id, user, credentials, job_spec, format_version, task_manager, pool, worker)

    def __init__(
        self,
        batch_id: int,
        user: str,
        credentials: CloudUserCredentials,
        job_spec,
        format_version: BatchFormatVersion,
        task_manager: aiotools.BackgroundTaskManager,
        pool: concurrent.futures.ThreadPoolExecutor,
        worker: 'Worker',
    ):
        self.batch_id = batch_id
        self.user = user
        self.credentials = credentials
        self.job_spec = job_spec
        self.format_version = format_version
        self.task_manager = task_manager
        self.pool = pool
        self.worker = worker

        self.deleted = False

        self.token = uuid.uuid4().hex
        self.scratch = f'/batch/{self.token}'

        self.disk = None
        self.state = 'pending'
        self.error = None

        self.start_time = None
        self.end_time = None

        self.cpu_in_mcpu = job_spec['resources']['cores_mcpu']
        self.memory_in_bytes = job_spec['resources']['memory_bytes']
        extra_storage_in_gib = job_spec['resources']['storage_gib']
        assert extra_storage_in_gib == 0 or is_valid_storage_request(CLOUD, extra_storage_in_gib)

        if instance_config.job_private:
            self.external_storage_in_gib = 0
            self.data_disk_storage_in_gib = extra_storage_in_gib
        else:
            self.external_storage_in_gib = extra_storage_in_gib
            # The reason for not giving each job 5 Gi (for example) is the
            # maximum number of simultaneous jobs on a worker is 64 which
            # basically fills the disk not allowing for caches etc. Most jobs
            # would need an external disk in that case.
            self.data_disk_storage_in_gib = min(
                RESERVED_STORAGE_GB_PER_CORE, self.cpu_in_mcpu / 1000 * RESERVED_STORAGE_GB_PER_CORE
            )

        self.resources = instance_config.quantified_resources(
            self.cpu_in_mcpu, self.memory_in_bytes, self.external_storage_in_gib
        )

        self.input_volume_mounts = []
        self.main_volume_mounts = []
        self.output_volume_mounts = []

        io_volume_mount = {
            'source': self.io_host_path(),
            'destination': '/io',
            'type': 'none',
            'options': ['rbind', 'rw'],
        }
        self.input_volume_mounts.append(io_volume_mount)
        self.main_volume_mounts.append(io_volume_mount)
        self.output_volume_mounts.append(io_volume_mount)

        cloudfuse = job_spec.get('cloudfuse') or job_spec.get('gcsfuse')
        self.cloudfuse = cloudfuse
        if cloudfuse:
            for config in cloudfuse:
                config['mounted'] = False
                bucket = config['bucket']
                assert bucket
                self.main_volume_mounts.append(
                    {
                        'source': f'{self.cloudfuse_data_path(bucket)}',
                        'destination': config['mount_path'],
                        'type': 'none',
                        'options': ['rbind', 'rw', 'shared'],
                    }
                )

        secrets = job_spec.get('secrets')
        self.secrets = secrets
        self.env = job_spec.get('env', [])

        self.project_id = Job.get_next_xfsquota_project_id()

    @property
    def job_id(self):
        return self.job_spec['job_id']

    @property
    def attempt_id(self):
        return self.job_spec['attempt_id']

    @property
    def id(self):
        return (self.batch_id, self.job_id)

    async def run(self):
        pass

    async def get_log(self):
        pass

    async def delete(self):
        log.info(f'deleting {self}')
        self.deleted = True

    async def mark_complete(self):
        self.end_time = time_msecs()

        full_status = self.status()

        if self.format_version.has_full_status_in_gcs():
            await retry_transient_errors(
                self.worker.file_store.write_status_file,
                self.batch_id,
                self.job_id,
                self.attempt_id,
                json.dumps(full_status),
            )

        if not self.deleted:
            log.info(f'{self}: marking complete')
            self.task_manager.ensure_future(self.worker.post_job_complete(self, full_status))

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
    def status(self):
        status = {
            'version': STATUS_FORMAT_VERSION,
            'worker': NAME,
            'batch_id': self.batch_id,
            'job_id': self.job_spec['job_id'],
            'attempt_id': self.job_spec['attempt_id'],
            'user': self.user,
            'state': self.state,
            'format_version': self.format_version.format_version,
            'resources': self.resources,
        }
        if self.error:
            status['error'] = self.error

        status['start_time'] = self.start_time
        status['end_time'] = self.end_time

        return status

    def __str__(self):
        return f'job {self.id}'


class DockerJob(Job):
    def __init__(
        self,
        batch_id: int,
        user: str,
        credentials: CloudUserCredentials,
        job_spec,
        format_version,
        task_manager: aiotools.BackgroundTaskManager,
        pool: concurrent.futures.ThreadPoolExecutor,
        client_session: httpx.ClientSession,
        worker: 'Worker',
    ):
        super().__init__(batch_id, user, credentials, job_spec, format_version, task_manager, pool, worker)
        input_files = job_spec.get('input_files')
        output_files = job_spec.get('output_files')

        requester_pays_project = job_spec.get('requester_pays_project')

        self.timings = Timings(lambda: False)

        if self.secrets:
            for secret in self.secrets:
                volume_mount = {
                    'source': self.secret_host_path(secret),
                    'destination': secret["mount_path"],
                    'type': 'none',
                    'options': ['rbind', 'rw'],
                }
                self.main_volume_mounts.append(volume_mount)
                # this will be the user credentials
                if secret.get('mount_in_copy', False):
                    self.input_volume_mounts.append(volume_mount)
                    self.output_volume_mounts.append(volume_mount)

        # create containers
        containers = {}

        if input_files:
            containers['input'] = copy_container(
                self,
                'input',
                input_files,
                self.input_volume_mounts,
                self.cpu_in_mcpu,
                self.memory_in_bytes,
                self.scratch,
                requester_pays_project,
                client_session,
                worker,
            )

        # main container
        main_spec = {
            'command': job_spec['process']['command'],
            'image': job_spec['process']['image'],
            'name': 'main',
            'env': [f'{var["name"]}={var["value"]}' for var in self.env],
            'cpu': self.cpu_in_mcpu,
            'memory': self.memory_in_bytes,
            'volume_mounts': self.main_volume_mounts,
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
        unconfined = job_spec.get('unconfined')
        if unconfined:
            main_spec['unconfined'] = unconfined
        main_spec['scratch'] = self.scratch
        containers['main'] = Container(self, 'main', main_spec, client_session, worker)

        if output_files:
            containers['output'] = copy_container(
                self,
                'output',
                output_files,
                self.output_volume_mounts,
                self.cpu_in_mcpu,
                self.memory_in_bytes,
                self.scratch,
                requester_pays_project,
                client_session,
                worker,
            )

        self.containers = containers

    def step(self, name: str):
        return self.timings.step(name)

    async def setup_io(self):
        if not instance_config.job_private:
            if self.worker.data_disk_space_remaining.value < self.external_storage_in_gib:
                log.info(
                    f'worker data disk storage is full: {self.external_storage_in_gib}Gi requested and {self.worker.data_disk_space_remaining}Gi remaining'
                )

                # disk name must be 63 characters or less
                # https://cloud.google.com/compute/docs/reference/rest/v1/disks#resource:-disk
                # under the information for the name field
                uid = self.token[:20]
                self.disk = CLOUD_WORKER_API.create_disk(
                    instance_name=NAME,
                    disk_name=f'batch-disk-{uid}',
                    size_in_gb=self.external_storage_in_gib,
                    mount_path=self.io_host_path(),
                )
                labels = {'namespace': NAMESPACE, 'batch': '1', 'instance-name': NAME, 'uid': uid}
                await self.disk.create(labels=labels)
                log.info(f'created disk {self.disk.name} for job {self.id}')
                return

            self.worker.data_disk_space_remaining.value -= self.external_storage_in_gib
            log.info(
                f'acquired {self.external_storage_in_gib}Gi from worker data disk storage with {self.worker.data_disk_space_remaining}Gi remaining'
            )

        assert self.disk is None, self.disk
        os.makedirs(self.io_host_path())

    async def run(self):
        async with self.worker.cpu_sem(self.cpu_in_mcpu):
            self.start_time = time_msecs()

            try:
                self.task_manager.ensure_future(self.worker.post_job_started(self))

                log.info(f'{self}: initializing')
                self.state = 'initializing'

                os.makedirs(f'{self.scratch}/')

                with self.step('setup_io'):
                    await self.setup_io()

                if not self.disk:
                    data_disk_storage_in_bytes = storage_gib_to_bytes(
                        self.external_storage_in_gib + self.data_disk_storage_in_gib
                    )
                else:
                    data_disk_storage_in_bytes = storage_gib_to_bytes(self.data_disk_storage_in_gib)

                with self.step('configuring xfsquota'):
                    # Quota will not be applied to `/io` if the job has an attached disk mounted there
                    await check_shell_output(f'xfs_quota -x -c "project -s -p {self.scratch} {self.project_id}" /host/')
                    await check_shell_output(
                        f'xfs_quota -x -c "limit -p bsoft={data_disk_storage_in_bytes} bhard={data_disk_storage_in_bytes} {self.project_id}" /host/'
                    )

                with self.step('populating secrets'):
                    if self.secrets:
                        for secret in self.secrets:
                            populate_secret_host_path(self.secret_host_path(secret), secret['data'])

                with self.step('adding cloudfuse support'):
                    if self.cloudfuse:
                        os.makedirs(self.cloudfuse_base_path())

                        await check_shell_output(
                            f'xfs_quota -x -c "project -s -p {self.cloudfuse_base_path()} {self.project_id}" /host/'
                        )

                        for config in self.cloudfuse:
                            bucket = config['bucket']
                            assert bucket

                            credentials = self.credentials.cloudfuse_credentials(config)
                            credentials_path = CLOUD_WORKER_API.write_cloudfuse_credentials(
                                self.scratch, credentials, bucket
                            )

                            os.makedirs(self.cloudfuse_data_path(bucket), exist_ok=True)
                            os.makedirs(self.cloudfuse_tmp_path(bucket), exist_ok=True)

                            await CLOUD_WORKER_API.mount_cloudfuse(
                                credentials_path,
                                self.cloudfuse_data_path(bucket),
                                self.cloudfuse_tmp_path(bucket),
                                config,
                            )
                            config['mounted'] = True

                self.state = 'running'

                input = self.containers.get('input')
                if input:
                    log.info(f'{self}: running input')
                    await input.run()
                    log.info(f'{self} input: {input.state}')

                if not input or input.state == 'succeeded':
                    log.info(f'{self}: running main')

                    main = self.containers['main']
                    await main.run()

                    log.info(f'{self} main: {main.state}')

                    output = self.containers.get('output')
                    if output:
                        log.info(f'{self}: running output')
                        await output.run()
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
            except JobDeletedError:
                self.state = 'cancelled'
            except Exception as e:
                if not user_error(e):
                    log.exception(f'while running {self}')

                self.state = 'error'
                self.error = traceback.format_exc()
            finally:
                with self.step('post-job finally block'):
                    try:
                        await self.cleanup()
                    finally:
                        await self.mark_complete()

    async def cleanup(self):
        log.info(f'{self}: cleaning up')

        if self.disk:
            try:
                await self.disk.delete()
                log.info(f'deleted disk {self.disk.name} for {self.id}')
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(f'while detaching and deleting disk {self.disk.name} for {self.id}')
            finally:
                await self.disk.close()
        else:
            self.worker.data_disk_space_remaining.value += self.external_storage_in_gib

        if self.cloudfuse:
            for config in self.cloudfuse:
                if config['mounted']:
                    bucket = config['bucket']
                    assert bucket
                    mount_path = self.cloudfuse_data_path(bucket)

                    try:
                        await CLOUD_WORKER_API.unmount_cloudfuse(mount_path)
                        log.info(f'unmounted fuse blob storage {bucket} from {mount_path}')
                        config['mounted'] = False
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception(f'while unmounting fuse blob storage {bucket} from {mount_path}')

        await check_shell(f'xfs_quota -x -c "limit -p bsoft=0 bhard=0 {self.project_id}" /host')

        try:
            await blocking_to_async(self.pool, shutil.rmtree, self.scratch, ignore_errors=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('while deleting volumes')

    async def get_log(self):
        return {name: await c.get_log() for name, c in self.containers.items()}

    async def delete(self):
        await super().delete()
        await asyncio.wait([c.delete() for c in self.containers.values()])

    def status(self):
        status = super().status()
        cstatuses = {name: c.status() for name, c in self.containers.items()}
        status['container_statuses'] = cstatuses
        status['timing'] = self.timings.to_dict()

        return status

    def __str__(self):
        return f'job {self.id}'


class JVMJob(Job):
    def __init__(
        self,
        batch_id: int,
        user: str,
        credentials: CloudUserCredentials,
        job_spec,
        format_version,
        task_manager: aiotools.BackgroundTaskManager,
        pool: concurrent.futures.ThreadPoolExecutor,
        worker: 'Worker',
    ):
        super().__init__(batch_id, user, credentials, job_spec, format_version, task_manager, pool, worker)
        assert job_spec['process']['type'] == 'jvm'
        assert worker is not None

        input_files = job_spec.get('input_files')
        output_files = job_spec.get('output_files')
        if input_files or output_files:
            raise Exception("i/o not supported")

        self.user_command_string = job_spec['process']['command']
        assert len(self.user_command_string) >= 3, self.user_command_string
        self.revision = self.user_command_string[1]
        self.jar_url = self.user_command_string[2]

        self.deleted = False
        self.timings = Timings(lambda: self.deleted)
        self.state = 'pending'

        self.jvm: Optional[JVM] = None
        self.jvm_name: Optional[str] = None

        self.log_file = f'{self.scratch}/log'

        assert self.worker.fs is not None

    def step(self, name):
        return self.timings.step(name)

    def verify_is_acceptable_query_jar_url(self, url: str):
        if not url.startswith(ACCEPTABLE_QUERY_JAR_URL_PREFIX):
            log.error(f'user submitted unacceptable JAR url: {url} for {self}. {ACCEPTABLE_QUERY_JAR_URL_PREFIX}')
            raise ValueError(f'unacceptable JAR url: {url}')

    def secret_host_path(self, secret):
        return f'{self.scratch}/secrets/{secret["mount_path"]}'

    async def run(self):
        async with self.worker.cpu_sem(self.cpu_in_mcpu):
            self.start_time = time_msecs()
            os.makedirs(f'{self.scratch}/')

            try:
                with self.step('connecting_to_jvm'):
                    self.jvm = await self.worker.borrow_jvm()
                    self.jvm_name = str(self.jvm)

                self.task_manager.ensure_future(self.worker.post_job_started(self))

                log.info(f'{self}: initializing')
                self.state = 'initializing'

                await check_shell_output(f'xfs_quota -x -c "project -s -p {self.scratch} {self.project_id}" /host/')
                await check_shell_output(
                    f'xfs_quota -x -c "limit -p bsoft={self.data_disk_storage_in_gib} bhard={self.data_disk_storage_in_gib} {self.project_id}" /host/'
                )

                if self.secrets:
                    for secret in self.secrets:
                        populate_secret_host_path(self.secret_host_path(secret), secret['data'])

                self.state = 'running'

                log.info(f'{self}: downloading JAR')
                with self.step('downloading_jar'):
                    async with self.worker.jar_download_locks[self.revision]:
                        local_jar_location = f'/hail-jars/{self.revision}.jar'
                        if not os.path.isfile(local_jar_location):
                            self.verify_is_acceptable_query_jar_url(self.jar_url)
                            temporary_file = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
                                delete=False
                            )
                            try:
                                async with await self.worker.fs.open(self.jar_url) as jar_data:
                                    while True:
                                        b = await jar_data.read(256 * 1024)
                                        if not b:
                                            break
                                        written = await blocking_to_async(worker.pool, temporary_file.write, b)
                                        assert written == len(b)
                                temporary_file.close()
                                os.rename(temporary_file.name, local_jar_location)
                            finally:
                                temporary_file.close()  # close is idempotent
                                try:
                                    os.remove(temporary_file.name)
                                except OSError as err:
                                    if err.errno != errno.ENOENT:
                                        raise

                log.info(f'{self}: running jvm process')
                with self.step('running'):
                    await self.jvm.execute(local_jar_location, self.scratch, self.log_file, self.user_command_string)
                self.state = 'succeeded'
                log.info(f'{self} main: {self.state}')
            except asyncio.CancelledError:
                raise
            except JVMUserError:
                self.state = 'failed'
                self.error = traceback.format_exc()
                await self.cleanup()
            except JobDeletedError:
                self.state = 'cancelled'
                await self.cleanup()
            except Exception:
                log.exception(f'while running {self}')

                self.state = 'error'
                self.error = traceback.format_exc()

                await self.cleanup()
            else:
                await self.cleanup()
            finally:
                await self.mark_complete()

    async def cleanup(self):
        if self.jvm is not None:
            worker.return_jvm(self.jvm)
            self.jvm = None

        # I really want this to be a timed step but I CANT RAISE EXCEPTIONS IN CLEANUP!!
        # with self.step('uploading_log'):
        log.info(f'{self}: uploading log')
        await worker.file_store.write_log_file(
            self.format_version, self.batch_id, self.job_id, self.attempt_id, 'main', await self._get_log()
        )

        log.info(f'{self}: cleaning up')
        try:
            await check_shell(f'xfs_quota -x -c "limit -p bsoft=0 bhard=0 {self.project_id}" /host')
            await blocking_to_async(self.pool, shutil.rmtree, self.scratch, ignore_errors=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('while deleting volumes')

    async def _get_log(self):
        if os.path.exists(self.log_file):
            return (await self.worker.fs.read(self.log_file)).decode()
        return ''

    async def get_log(self):
        return {'main': await self._get_log()}

    async def delete(self):
        log.info(f'deleting {self} {self.jvm}')
        self.deleted = True
        if self.jvm is not None:
            log.info(f'{self.jvm} interrupting')
            self.jvm.interrupt()

    # {
    #   version: int,
    #   worker: str,
    #   batch_id: int,
    #   job_id: int,
    #   attempt_id: int,
    #   user: str,
    #   state: str, (pending, initializing, running, succeeded, error, failed, cancelled)
    #   format_version: int
    #   error: str, (optional)
    #   container_statuses: [Container.status],
    #   start_time: int,
    #   end_time: int,
    #   resources: list of dict, {name: str, quantity: int},
    #   jvm: str
    # }
    def status(self):
        status = super().status()
        status['container_statuses'] = {}
        status['container_statuses']['main'] = {'name': 'main', 'state': self.state, 'timing': self.timings.to_dict()}
        status['jvm'] = self.jvm_name
        return status

    def __str__(self):
        return f'job {self.id}'


class ImageData:
    def __init__(self):
        self.ref_count = 0
        self.time_created = time_msecs()
        self.last_accessed = time_msecs()
        self.lock = asyncio.Lock()
        self.extracted = False

    def __add__(self, other):
        self.ref_count += other
        self.last_accessed = time_msecs()
        return self

    def __sub__(self, other):
        self.ref_count -= other
        assert self.ref_count >= 0
        self.last_accessed = time_msecs()
        return self

    def __str__(self):
        return (
            f'ImageData('
            f'ref_count={self.ref_count}, '
            f'time_created={time_msecs_str(self.time_created)}, '
            f'last_accessed={time_msecs_str(self.last_accessed)}'
            f')'
        )


@contextmanager
def scoped_ensure_future(coro_or_future, *, loop=None) -> Iterator[asyncio.Future]:
    fut = asyncio.ensure_future(coro_or_future, loop=loop)
    try:
        yield fut
    finally:
        fut.cancel()


class BufferedOutputProcess:
    @classmethod
    async def create(cls, *args, **kwargs):
        assert 'stdout' not in kwargs
        assert 'stderr' not in kwargs

        process = await asyncio.create_subprocess_exec(
            *args, **kwargs, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stop_event = asyncio.Event()
        return cls(process, stop_event)

    def __init__(self, process, stop_event: asyncio.Event):
        self.process = process
        self.stop_event = stop_event
        self.buf = bytearray()
        assert process.stdout is not None
        self.stdout_pump = asyncio.ensure_future(self.pump_to_buffer(process.stdout))
        assert process.stderr is not None
        self.stderr_pump = asyncio.ensure_future(self.pump_to_buffer(process.stderr))

    async def pump_to_buffer(self, strm: asyncio.StreamReader):
        with scoped_ensure_future(self.stop_event.wait()) as stop_fut:
            while not strm.at_eof() and not self.stop_event.is_set():
                with scoped_ensure_future(strm.readline()) as read_fut:
                    await asyncio.wait([read_fut, stop_fut], return_when=asyncio.FIRST_COMPLETED)
                    if read_fut.done():
                        result = read_fut.result()
                        self.buf.extend(result)

    def output(self) -> str:
        return self.buf.decode()

    def retrieve_and_clear_output(self) -> str:
        buf = self.buf.decode()
        self.buf = bytearray()
        return buf

    def kill(self):
        return self.process.kill()

    @property
    def returncode(self) -> Optional[int]:
        return self.process.returncode

    def close(self):
        try:
            self.kill()
        finally:
            try:
                self.stdout_pump.cancel()
            finally:
                self.stderr_pump.cancel()


class JVMUserError(Exception):
    pass


class JVM:
    SPARK_HOME = find_spark_home()

    FINISH_USER_EXCEPTION = 0
    FINISH_ENTRYWAY_EXCEPTION = 1
    FINISH_NORMAL = 2
    FINISH_CANCELLED = 3
    FINISH_JVM_EOS = 4

    @classmethod
    async def create_process(cls, socket_file: str) -> BufferedOutputProcess:
        # JVM and Hail both treat MB as 1024 * 1024 bytes.
        # JVMs only start in standard workers which have 3.75 GiB == 3840 MiB per core.
        # We only allocate 3700 MiB so that we stay well below the machine's max memory.
        # We allocate 60% of memory per core to off heap memory: 1480 + 2220 = 3700.
        return await BufferedOutputProcess.create(
            'java',
            '-Xmx1480M',
            '-cp',
            f'/jvm-entryway:/jvm-entryway/junixsocket-selftest-2.3.3-jar-with-dependencies.jar:{JVM.SPARK_HOME}/jars/*',
            'is.hail.JVMEntryway',
            socket_file,
            env={'HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB': '2220'},
        )

    @classmethod
    async def create_process_and_connect(cls, index: int, socket_file: str) -> Tuple[BufferedOutputProcess, str]:
        process = await cls.create_process(socket_file)
        try:
            attempts = 0
            delay = 0.25
            while True:
                try:
                    log.info(f'JVM-{index}: trying to open socket')
                    reader, writer = await asyncio.open_unix_connection(socket_file)
                    try:
                        log.info(f'JVM-{index}: establishing connection')
                        b = await read_bool(reader)
                        assert b, f'expected true, got {b}'
                        writer.write(b'\0x01')
                        break
                    finally:
                        writer.close()
                except ConnectionRefusedError:
                    output = process.retrieve_and_clear_output()
                    log.warning(f'JVM-{index}: connection refused. {output}')
                    raise
                except FileNotFoundError as err:
                    attempts += 1
                    if attempts == 240:
                        raise ValueError(
                            f'JVM-{index}: failed to establish connection after {240 * delay} seconds'
                        ) from err
                    await asyncio.sleep(delay)
            startup_output = process.retrieve_and_clear_output()
            return process, startup_output
        except:
            process.close()
            raise

    @classmethod
    async def create(cls, index: int):
        assert worker is not None

        while True:
            try:
                token = uuid.uuid4().hex
                socket_file = '/socket-' + token
                root_dir = '/root-' + token
                output_file = root_dir + '/output'
                should_interrupt = asyncio.Event()
                await blocking_to_async(worker.pool, os.mkdir, root_dir)
                process, startup_output = await cls.create_process_and_connect(index, socket_file)
                log.info(f'JVM-{index}: startup output: {startup_output}')
                return cls(index, socket_file, root_dir, output_file, should_interrupt, process)
            except ConnectionRefusedError:
                pass

    async def new_connection(self):
        while True:
            try:
                interim_output = self.process.retrieve_and_clear_output()
                if len(interim_output) > 0:
                    log.warning(f'{self}: unexpected output between jobs')

                return await asyncio.open_unix_connection(self.socket_file)
            except ConnectionRefusedError:
                log.warning(f'{self}: unexpected exit between jobs', extra=dict(output=self.process.output()))
                os.remove(self.socket_file)
                process, startup_output = await self.create_process_and_connect(self.index, self.socket_file)
                self.process = process
                log.info(f'JVM-{self.index}: startup output: {startup_output}')

    def __init__(
        self,
        index: int,
        socket_file: str,
        root_dir: str,
        output_file: str,
        should_interrupt: asyncio.Event,
        process: BufferedOutputProcess,
    ):
        self.index = index
        self.socket_file = socket_file
        self.root_dir = root_dir
        self.output_file = output_file
        self.should_interrupt = should_interrupt
        self.process = process

    def __str__(self):
        return f'JVM-{self.index}'

    def __repr__(self):
        return f'JVM-{self.index}'

    def interrupt(self):
        self.should_interrupt.set()

    def reset(self):
        self.should_interrupt.clear()

    def kill(self):
        if self.process is not None:
            self.process.kill()

    def close(self):
        self.process.close()

    async def execute(self, classpath: str, scratch_dir: str, log_file: str, command_string: List[str]):
        assert worker is not None

        log.info(f'{self}: execute')

        with ExitStack() as stack:
            reader: asyncio.StreamReader
            writer: asyncio.StreamWriter
            reader, writer = await self.new_connection()
            stack.callback(writer.close)
            log.info(f'{self}: connection acquired')

            command_string = [classpath, 'is.hail.backend.service.Main', scratch_dir, log_file, *command_string]

            write_int(writer, len(command_string))
            for arg in command_string:
                assert isinstance(arg, str)
                write_str(writer, arg)
            await writer.drain()

            wait_for_message_from_process: asyncio.Future = asyncio.ensure_future(read_int(reader))
            stack.callback(wait_for_message_from_process.cancel)
            wait_for_interrupt: asyncio.Future = asyncio.ensure_future(self.should_interrupt.wait())
            stack.callback(wait_for_interrupt.cancel)

            await asyncio.wait([wait_for_message_from_process, wait_for_interrupt], return_when=asyncio.FIRST_COMPLETED)

            if wait_for_interrupt.done():
                await wait_for_interrupt  # retrieve exceptions
                if not wait_for_message_from_process.done():
                    write_int(writer, 0)  # tell process to cancel
                    await writer.drain()

            eos_exception = None
            try:
                message = await wait_for_message_from_process
            except EndOfStream as exc:
                try:
                    self.kill()
                except ProcessLookupError:
                    log.warning(f'{self}: JVM died after we received EOS')
                message = JVM.FINISH_JVM_EOS
                eos_exception = exc

            if message == JVM.FINISH_NORMAL:
                log.info(f'{self}: finished normally (interrupted: {wait_for_interrupt.done()})')
            elif message == JVM.FINISH_CANCELLED:
                assert wait_for_interrupt.done()
                log.info(f'{self}: was cancelled')
                raise JobDeletedError
            elif message == JVM.FINISH_USER_EXCEPTION:
                log.info(f'{self}: user exception encountered (interrupted: {wait_for_interrupt.done()})')
                exception = await read_str(reader)
                raise JVMUserError(exception)
            elif message == JVM.FINISH_ENTRYWAY_EXCEPTION:
                log.info(f'{self}: entryway exception encountered (interrupted: {wait_for_interrupt.done()})')
                exception = await read_str(reader)
                raise ValueError(exception)
            elif message == JVM.FINISH_JVM_EOS:
                assert eos_exception is not None
                log.warning(f'{self}: unexpected end of stream in jvm (interrupted: {wait_for_interrupt.done()})')
                raise ValueError('unexpected end of stream in jvm') from eos_exception


class Worker:
    def __init__(self, client_session: httpx.ClientSession):
        self.active = False
        self.cores_mcpu = CORES * 1000
        self.last_updated = time_msecs()
        self.cpu_sem = FIFOWeightedSemaphore(self.cores_mcpu)
        self.data_disk_space_remaining = Box(UNRESERVED_WORKER_DATA_DISK_SIZE_GB)
        self.pool = concurrent.futures.ThreadPoolExecutor()
        self.jobs: Dict[Tuple[int, int], Job] = {}
        self.stop_event = asyncio.Event()
        self.task_manager = aiotools.BackgroundTaskManager()
        os.mkdir('/hail-jars/')
        self.jar_download_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.client_session = client_session

        self.image_data: Dict[str, ImageData] = defaultdict(ImageData)
        self.image_data[BATCH_WORKER_IMAGE_ID] += 1

        # filled in during activation
        self.fs = None
        self.file_store = None
        self.headers = None
        self.compute_client = None

        self._jvm_initializer_task = asyncio.ensure_future(self._initialize_jvms())
        self._jvms: List[JVM] = []

    async def _initialize_jvms(self):
        if instance_config.worker_type() in ('standard', 'D'):
            self._jvms = await asyncio.gather(*[JVM.create(i) for i in range(CORES)])
        log.info(f'JVMs initialized {self._jvms}')

    async def borrow_jvm(self) -> JVM:
        if instance_config.worker_type() not in ('standard', 'D'):
            raise ValueError(f'JVM jobs not allowed on {instance_config.worker_type()}')
        await self._jvm_initializer_task
        assert self._jvms
        return self._jvms.pop()

    def return_jvm(self, jvm: JVM):
        if instance_config.worker_type() not in ('standard', 'D'):
            raise ValueError(f'JVM jobs not allowed on {instance_config.worker_type()}')
        jvm.reset()
        self._jvms.append(jvm)

    async def shutdown(self):
        log.info('Worker.shutdown')
        try:
            with ExitStack() as cleanup:
                for jvm in self._jvms:
                    cleanup.callback(jvm.close)
        finally:
            try:
                self.task_manager.shutdown()
                log.info('shutdown task manager')
            finally:
                try:
                    if self.fs:
                        await self.fs.close()
                        log.info('closed worker file system')
                finally:
                    try:
                        if self.compute_client:
                            await self.compute_client.close()
                            log.info('closed compute client')
                    finally:
                        try:
                            if self.file_store:
                                await self.file_store.close()
                                log.info('closed file store')
                        finally:
                            await self.client_session.close()
                            log.info('closed client session')

    async def run_job(self, job):  # pylint: disable=no-self-use
        try:
            await job.run()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not user_error(e):
                log.exception(f'while running {job}, ignoring')

    async def create_job_1(self, request):
        body = await request.json()

        batch_id = body['batch_id']
        job_id = body['job_id']

        format_version = BatchFormatVersion(body['format_version'])

        token = body['token']
        start_job_id = body['start_job_id']
        addtl_spec = body['job_spec']

        job_spec = await self.file_store.read_spec_file(batch_id, token, start_job_id, job_id)
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

        assert job_spec['job_id'] == job_id
        id = (batch_id, job_id)

        # already running
        if id in self.jobs:
            return web.HTTPForbidden()

        # check worker hasn't started shutting down
        if not self.active:
            return web.HTTPServiceUnavailable()

        credentials = CLOUD_WORKER_API.user_credentials(body['gsa_key'])

        job = Job.create(
            batch_id,
            body['user'],
            credentials,
            job_spec,
            format_version,
            self.task_manager,
            self.pool,
            self.client_session,
            self,
        )

        log.info(f'created {job}, adding to jobs')

        self.jobs[job.id] = job

        self.task_manager.ensure_future(self.run_job(job))

        return web.Response()

    async def create_job(self, request):
        if not self.active:
            raise web.HTTPServiceUnavailable
        return await asyncio.shield(self.create_job_1(request))

    async def get_job_log(self, request):
        if not self.active:
            raise web.HTTPServiceUnavailable
        batch_id = int(request.match_info['batch_id'])
        job_id = int(request.match_info['job_id'])
        id = (batch_id, job_id)
        job = self.jobs.get(id)
        if not job:
            raise web.HTTPNotFound()
        return web.json_response(await job.get_log())

    async def get_job_status(self, request):
        if not self.active:
            raise web.HTTPServiceUnavailable
        batch_id = int(request.match_info['batch_id'])
        job_id = int(request.match_info['job_id'])
        id = (batch_id, job_id)
        job = self.jobs.get(id)
        if not job:
            raise web.HTTPNotFound()
        return web.json_response(job.status())

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
        if not self.active:
            raise web.HTTPServiceUnavailable
        return await asyncio.shield(self.delete_job_1(request))

    async def healthcheck(self, request):  # pylint: disable=unused-argument
        if not self.active:
            raise web.HTTPServiceUnavailable
        body = {'name': NAME}
        return web.json_response(body)

    async def run(self):
        app = web.Application(client_max_size=HTTP_CLIENT_MAX_SIZE)
        app.add_routes(
            [
                web.post('/api/v1alpha/kill', self.kill),
                web.post('/api/v1alpha/batches/jobs/create', self.create_job),
                web.delete('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/delete', self.delete_job),
                web.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log', self.get_job_log),
                web.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/status', self.get_job_status),
                web.get('/healthcheck', self.healthcheck),
            ]
        )

        self.task_manager.ensure_future(periodically_call(60, self.cleanup_old_images))

        app_runner = web.AppRunner(app)
        await app_runner.setup()
        site = web.TCPSite(app_runner, '0.0.0.0', 5000)
        await site.start()

        try:
            await asyncio.wait_for(self.activate(), MAX_IDLE_TIME_MSECS / 1000)
        except asyncio.TimeoutError:
            log.exception(f'could not activate after trying for {MAX_IDLE_TIME_MSECS} ms, exiting')
            return

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
                    log.info(
                        f'n_jobs {len(self.jobs)} free_cores {self.cpu_sem.value / 1000} idle {idle_duration} '
                        f'free worker data disk storage {self.data_disk_space_remaining.value}Gi'
                    )
        finally:
            self.active = False
            log.info('shutting down')
            await site.stop()
            log.info('stopped site')
            await app_runner.cleanup()
            log.info('cleaned up app runner')
            await self.deactivate()
            log.info('deactivated')

    async def deactivate(self):
        # Don't retry.  If it doesn't go through, the driver
        # monitoring loops will recover.  If the driver is
        # gone (e.g. testing a PR), this would go into an
        # infinite loop and the instance won't be deleted.
        await self.client_session.post(
            deploy_config.url('batch-driver', '/api/v1alpha/instances/deactivate'), headers=self.headers
        )

    async def kill_1(self, request):  # pylint: disable=unused-argument
        log.info('killed')
        self.stop_event.set()

    async def kill(self, request):
        if not self.active:
            raise web.HTTPServiceUnavailable
        return await asyncio.shield(self.kill_1(request))

    async def post_job_complete_1(self, job, full_status):
        run_duration = job.end_time - job.start_time
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
            'status': db_status,
        }

        body = {'status': status}

        start_time = time_msecs()
        delay_secs = 0.1
        while True:
            try:
                await self.client_session.post(
                    deploy_config.url('batch-driver', '/api/v1alpha/instances/job_complete'),
                    json=body,
                    headers=self.headers,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if isinstance(e, aiohttp.ClientResponseError) and e.status == 404:  # pylint: disable=no-member
                    raise
                log.warning(f'failed to mark {job} complete, retrying', exc_info=True)

            # unlist job after 3m or half the run duration
            now = time_msecs()
            elapsed = now - start_time
            if job.id in self.jobs and elapsed > 180 * 1000 and elapsed > run_duration / 2:
                log.info(f'too much time elapsed marking {job} complete, removing from jobs, will keep retrying')
                del self.jobs[job.id]
                self.last_updated = time_msecs()

            await asyncio.sleep(delay_secs * random.uniform(0.7, 1.3))
            # exponentially back off, up to (expected) max of 2m
            delay_secs = min(delay_secs * 2, 2 * 60.0)

    async def post_job_complete(self, job, full_status):
        try:
            await self.post_job_complete_1(job, full_status)
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
        full_status = job.status()

        status = {
            'version': full_status['version'],
            'batch_id': full_status['batch_id'],
            'job_id': full_status['job_id'],
            'attempt_id': full_status['attempt_id'],
            'start_time': full_status['start_time'],
            'resources': full_status['resources'],
        }

        body = {'status': status}

        await request_retry_transient_errors(
            self.client_session,
            'POST',
            deploy_config.url('batch-driver', '/api/v1alpha/instances/job_started'),
            json=body,
            headers=self.headers,
        )

    async def post_job_started(self, job):
        try:
            await self.post_job_started_1(job)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(f'error while posting {job} started')

    async def activate(self):
        log.info('activating')
        resp = await request_retry_transient_errors(
            self.client_session,
            'GET',
            deploy_config.url('batch-driver', '/api/v1alpha/instances/credentials'),
            headers={'X-Hail-Instance-Name': NAME, 'Authorization': f'Bearer {os.environ["ACTIVATION_TOKEN"]}'},
        )
        resp_json = await resp.json()

        credentials_file = '/worker-key.json'
        with open(credentials_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(resp_json['key']))

        self.fs = RouterAsyncFS(
            'file',
            filesystems=[
                LocalAsyncFS(self.pool),
                get_cloud_async_fs(credentials_file=credentials_file),
            ],
        )

        fs = get_cloud_async_fs(credentials_file=credentials_file)
        self.file_store = FileStore(fs, BATCH_LOGS_STORAGE_URI, INSTANCE_ID)

        self.compute_client = get_compute_client(credentials_file=credentials_file)

        resp = await request_retry_transient_errors(
            self.client_session,
            'POST',
            deploy_config.url('batch-driver', '/api/v1alpha/instances/activate'),
            json={'ip_address': os.environ['IP_ADDRESS']},
            headers={'X-Hail-Instance-Name': NAME, 'Authorization': f'Bearer {os.environ["ACTIVATION_TOKEN"]}'},
        )
        resp_json = await resp.json()
        self.headers = {'X-Hail-Instance-Name': NAME, 'Authorization': f'Bearer {resp_json["token"]}'}
        self.active = True
        self.last_updated = time_msecs()
        log.info('activated')

    async def cleanup_old_images(self):
        try:
            async with image_lock.writer_lock:
                log.info(f"Obtained writer lock. The image ref counts are: {self.image_data}")
                for image_id in list(self.image_data.keys()):
                    now = time_msecs()
                    image_data = self.image_data[image_id]
                    if image_data.ref_count == 0 and (now - image_data.last_accessed) > 10 * 60 * 1000:
                        assert image_id != BATCH_WORKER_IMAGE_ID
                        log.info(f'Found an unused image with ID {image_id}')
                        await check_shell(f'docker rmi -f {image_id}')
                        image_path = f'/host/rootfs/{image_id}'
                        await blocking_to_async(self.pool, shutil.rmtree, image_path)
                        del self.image_data[image_id]
                        log.info(f'Deleted image from cache with ID {image_id}')
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception(f'Error while deleting unused image: {e}')


async def async_main():
    global port_allocator, network_allocator, worker, docker

    docker = aiodocker.Docker()

    port_allocator = PortAllocator()
    network_allocator = NetworkAllocator()
    await network_allocator.reserve()

    worker = Worker(httpx.client_session())
    try:
        await worker.run()
    finally:
        try:
            await worker.shutdown()
            log.info('worker shutdown', exc_info=True)
        finally:
            try:
                await docker.close()
                log.info('docker closed')
            finally:
                asyncio.get_event_loop().set_debug(True)
                log.debug('Tasks immediately after docker close')
                dump_all_stacktraces()
                other_tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
                if other_tasks:
                    _, pending = await asyncio.wait(other_tasks, timeout=10 * 60, return_when=asyncio.ALL_COMPLETED)
                    for t in pending:
                        log.debug('Dangling task:')
                        t.print_stack()
                        t.cancel()


loop = asyncio.get_event_loop()
loop.add_signal_handler(signal.SIGUSR1, dump_all_stacktraces)
loop.run_until_complete(async_main())
log.info('closing loop')
loop.close()
log.info('closed')
sys.exit(0)
