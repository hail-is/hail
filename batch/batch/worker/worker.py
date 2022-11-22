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
from contextlib import AsyncExitStack, ExitStack, contextmanager
from typing import (
    Any,
    Awaitable,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import aiodocker  # type: ignore
import aiodocker.images
import aiohttp
import aiohttp.client_exceptions
import aiorwlock
import async_timeout
import humanize
import orjson
from aiodocker.exceptions import DockerError  # type: ignore
from aiohttp import web
from sortedcontainers import SortedSet

from gear.clients import get_cloud_async_fs, get_compute_client
from hailtop import aiotools, httpx
from hailtop.aiotools import AsyncFS, LocalAsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.batch.hail_genetics_images import HAIL_GENETICS_IMAGES
from hailtop.config import DeployConfig
from hailtop.hail_logging import AccessLogger, configure_logging
from hailtop.utils import (
    CalledProcessError,
    Timings,
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
    retry_transient_errors_with_debug_string,
    time_msecs,
    time_msecs_str,
)

from ..batch_format_version import BatchFormatVersion
from ..cloud.azure.worker.worker_api import AzureWorkerAPI
from ..cloud.gcp.worker.worker_api import GCPWorkerAPI
from ..cloud.resource_utils import (
    is_valid_storage_request,
    storage_gib_to_bytes,
    worker_memory_per_core_bytes,
    worker_memory_per_core_mib,
)
from ..file_store import FileStore
from ..globals import HTTP_CLIENT_MAX_SIZE, RESERVED_STORAGE_GB_PER_CORE, STATUS_FORMAT_VERSION
from ..publicly_available_images import publicly_available_images
from ..resource_usage import ResourceUsageMonitor
from ..semaphore import FIFOWeightedSemaphore
from ..utils import Box
from ..worker.worker_api import CloudDisk, CloudWorkerAPI
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


class BatchWorkerAccessLogger(AccessLogger):
    def __init__(self, logger: logging.Logger, log_format: str):
        super().__init__(logger, log_format)

        self.exclude = [
            ('GET', re.compile('/healthcheck')),
            ('POST', re.compile('/api/v1alpha/batches/jobs/create')),
        ]

    def log(self, request, response, time):
        for method, path_expr in self.exclude:
            if path_expr.fullmatch(request.path) and method == request.method:
                return

        super().log(request, response, time)


def compose_auth_header_urlsafe(orig_f):
    def compose(auth: Union[MutableMapping, str, bytes], registry_addr: Optional[str] = None):
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
log.info(f'BATCH_WORKER_IMAGE {BATCH_WORKER_IMAGE}')
log.info(f'BATCH_WORKER_IMAGE_ID {BATCH_WORKER_IMAGE_ID}')
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

image_lock: Optional[aiorwlock.RWLock] = None


class PortAllocator:
    def __init__(self):
        self.ports: asyncio.Queue[int] = asyncio.Queue()
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
        self.private_networks: asyncio.Queue[NetworkNamespace] = asyncio.Queue()
        self.public_networks: asyncio.Queue[NetworkNamespace] = asyncio.Queue()
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


def docker_call_retry(timeout, name, f, *args, **kwargs):
    debug_string = f'In docker call to {f.__name__} for {name}'

    async def timed_out_f(*args, **kwargs):
        return await asyncio.wait_for(f(*args, **kwargs), timeout)

    return retry_transient_errors_with_debug_string(debug_string, timed_out_f, *args, **kwargs)


class ImageCannotBePulled(Exception):
    pass


class ImageNotFound(Exception):
    pass


class InvalidImageRepository(Exception):
    pass


class Image:
    def __init__(
        self,
        name: str,
        credentials: Union[CloudUserCredentials, 'JVMUserCredentials', 'CopyStepCredentials'],
        client_session: httpx.ClientSession,
        pool: concurrent.futures.ThreadPoolExecutor,
    ):
        self.image_name = name
        self.credentials = credentials
        self.client_session = client_session
        self.pool = pool

        image_ref = parse_docker_image_reference(name)
        if image_ref.tag is None and image_ref.digest is None:
            log.info(f'adding latest tag to image {name} for {self}')
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
        self.image_config: Optional[Dict[str, Any]] = None
        self.image_id: Optional[str] = None

    @property
    def is_cloud_image(self):
        return (CLOUD == 'gcp' and self.image_ref.hosted_in('google')) or (
            CLOUD == 'azure' and self.image_ref.hosted_in('azure')
        )

    @property
    def is_public_image(self):
        return self.image_ref.name() in PUBLIC_IMAGES

    @property
    def rootfs_path(self) -> str:
        assert self.image_id is not None
        return f'/host/rootfs/{self.image_id}'

    async def _pull_image(self):
        assert docker

        try:
            if not self.is_cloud_image:
                await self._ensure_image_is_pulled()
            elif self.is_public_image:
                auth = await self._batch_worker_access_token()
                await self._ensure_image_is_pulled(auth=auth)
            elif self.image_ref_str == BATCH_WORKER_IMAGE and isinstance(
                self.credentials, (JVMUserCredentials, CopyStepCredentials)
            ):
                pass
            else:
                # Pull to verify this user has access to this
                # image.
                # FIXME improve the performance of this with a
                # per-user image cache.
                auth = self._current_user_access_token()
                await docker_call_retry(
                    MAX_DOCKER_IMAGE_PULL_SECS, str(self), docker.images.pull, self.image_ref_str, auth=auth
                )
        except DockerError as e:
            if e.status == 404 and 'pull access denied' in e.message:
                raise ImageCannotBePulled from e
            if (
                e.status == 500
                and 'Permission "artifactregistry.repositories.downloadArtifacts" denied on resource' in e.message
            ):
                raise ImageCannotBePulled from e
            if 'not found: manifest unknown' in e.message:
                raise ImageNotFound from e
            if 'Invalid repository name' in e.message:
                raise InvalidImageRepository from e
            raise

        image_config, _ = await check_exec_output('docker', 'inspect', self.image_ref_str)
        image_configs[self.image_ref_str] = json.loads(image_config)[0]

    async def _ensure_image_is_pulled(self, auth: Optional[Dict[str, str]] = None):
        assert docker

        try:
            await docker_call_retry(MAX_DOCKER_OTHER_OPERATION_SECS, str(self), docker.images.get, self.image_ref_str)
        except DockerError as e:
            if e.status == 404:
                await docker_call_retry(
                    MAX_DOCKER_IMAGE_PULL_SECS, str(self), docker.images.pull, self.image_ref_str, auth=auth
                )
            else:
                raise

    async def _batch_worker_access_token(self) -> Dict[str, str]:
        return await CLOUD_WORKER_API.worker_access_token(self.client_session)

    def _current_user_access_token(self) -> Dict[str, str]:
        assert self.credentials and isinstance(self.credentials, CloudUserCredentials)
        return {'username': self.credentials.username, 'password': self.credentials.password}

    async def _extract_rootfs(self):
        assert self.image_id
        os.makedirs(self.rootfs_path)
        await check_shell(
            f'id=$(docker create {self.image_id}) && docker export $id | tar -C {self.rootfs_path} -xf - && docker rm $id'
        )

    async def _localize_rootfs(self):
        assert image_lock
        async with image_lock.reader:
            # FIXME Authentication is entangled with pulling images. We need a way to test
            # that a user has access to a cached image without pulling.
            await self._pull_image()
            self.image_config = image_configs[self.image_ref_str]
            self.image_id = self.image_config['Id'].split(":")[1]
            assert self.image_id

            assert worker
            worker.image_data[self.image_id] += 1

            image_data = worker.image_data[self.image_id]
            async with image_data.lock:
                if not image_data.extracted:
                    try:
                        await self._extract_rootfs()
                        image_data.extracted = True
                        log.info(f'Added expanded image to cache: {self.image_ref_str}, ID: {self.image_id}')
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception(f'while extracting image {self.image_ref_str}, ID: {self.image_id}')
                        await blocking_to_async(self.pool, shutil.rmtree, self.rootfs_path)
                        raise

    async def pull(self):
        await asyncio.shield(self._localize_rootfs())

    def release(self):
        assert worker
        if self.image_id is not None:
            worker.image_data[self.image_id] -= 1


class StepInterruptedError(Exception):
    pass


async def run_until_done_or_deleted(event: asyncio.Event, f: Callable[..., Awaitable[Any]], *args, **kwargs):
    step = asyncio.ensure_future(f(*args, **kwargs))
    deleted = asyncio.ensure_future(event.wait())
    try:
        await asyncio.wait([deleted, step], return_when=asyncio.FIRST_COMPLETED)
        if deleted.done():
            raise StepInterruptedError
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


class ContainerDeletedError(Exception):
    pass


class ContainerTimeoutError(Exception):
    pass


class ContainerCreateError(Exception):
    pass


class ContainerStartError(Exception):
    pass


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
    if isinstance(e, (ImageNotFound, ImageCannotBePulled, InvalidImageRepository)):
        return True
    if isinstance(e, (ContainerTimeoutError, ContainerDeletedError)):
        return True
    return False


class Container:
    def __init__(
        self,
        fs: AsyncFS,
        name: str,
        image: Image,
        scratch_dir: str,
        command: List[str],
        cpu_in_mcpu: int,
        memory_in_bytes: int,
        network: Optional[Union[bool, str]] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
        unconfined: Optional[bool] = None,
        volume_mounts: Optional[List[dict]] = None,
        env: Optional[List[str]] = None,
        stdin: Optional[str] = None,
    ):
        self.fs = fs
        assert self.fs

        self.name = name
        self.image = image
        self.command = command
        self.cpu_in_mcpu = cpu_in_mcpu
        self.memory_in_bytes = memory_in_bytes
        self.network = network
        self.port = port
        self.timeout = timeout
        self.unconfined = unconfined
        self.volume_mounts = volume_mounts or []
        self.env = env or []
        self.stdin = stdin

        self.deleted_event = asyncio.Event()

        self.host_port = None

        self.state = 'pending'
        self.error: Optional[str] = None
        self.short_error: Optional[str] = None
        self.container_status: Optional[dict] = None
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None

        self.timings = Timings()

        self.overlay_path = None

        self.container_scratch = scratch_dir
        self.container_overlay_path = f'{self.container_scratch}/rootfs_overlay'
        self.config_path = f'{self.container_scratch}/config'
        self.log_path = f'{self.container_scratch}/container.log'
        self.resource_usage_path = f'{self.container_scratch}/resource_usage'

        self.overlay_mounted = False

        self.netns: Optional[NetworkNamespace] = None
        # regarding no-member: https://github.com/PyCQA/pylint/issues/4223
        self.process: Optional[asyncio.subprocess.Process] = None  # pylint: disable=no-member

        self._run_fut: Optional[asyncio.Future] = None
        self._cleanup_lock = asyncio.Lock()

        self._killed = False
        self._cleaned_up = False

    async def create(self):
        self.state = 'creating'
        try:
            with self._step('pulling'):
                await self._run_until_done_or_deleted(self.image.pull)

            with self._step('setting up overlay'):
                await self._run_until_done_or_deleted(self._setup_overlay)

            with self._step('setting up network'):
                await self._run_until_done_or_deleted(self._setup_network_namespace)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if isinstance(e, ImageNotFound):
                self.short_error = 'image not found'
            elif isinstance(e, ImageCannotBePulled):
                self.short_error = 'image cannot be pulled'
            elif isinstance(e, InvalidImageRepository):
                self.short_error = 'image repository is invalid'

            self.state = 'error'
            self.error = traceback.format_exc()

            if not isinstance(e, ContainerDeletedError) and not user_error(e):
                log.exception(f'while creating {self}')
                raise ContainerCreateError from e
            raise

    async def start(self):
        async def _run():
            self.state = 'running'
            try:
                with self._step('running'):
                    timed_out = await self._run_until_done_or_deleted(self._run_container)

                self.container_status = self.get_container_status()
                assert self.container_status is not None

                if timed_out:
                    self.short_error = 'timed out'
                    raise ContainerTimeoutError(f'timed out after {self.timeout}s')

                if self.container_status['exit_code'] == 0:
                    self.state = 'succeeded'
                else:
                    if self.container_status['out_of_memory']:
                        self.short_error = 'out of memory'
                    self.state = 'failed'
            except asyncio.CancelledError:
                raise
            except ContainerDeletedError:
                self.state = 'cancelled'
            except Exception as e:
                self.state = 'error'
                self.error = traceback.format_exc()

                if not isinstance(e, ContainerTimeoutError) and not user_error(e):
                    log.exception(f'while running {self}')
                    raise ContainerStartError from e
                raise

        self._run_fut = asyncio.ensure_future(self._run_until_done_or_deleted(_run))

    async def wait(self):
        assert self._run_fut
        try:
            await self._run_fut
        finally:
            self._run_fut = None

    async def run(self, on_completion: Callable[..., Awaitable[Any]], *args, **kwargs):
        async with self._cleanup_lock:
            try:
                await self.create()
                await self.start()
                await self.wait()
            finally:
                try:
                    await on_completion(*args, **kwargs)
                finally:
                    try:
                        await self._kill()
                    finally:
                        await self._cleanup()

    async def _kill(self):
        if self._killed:
            return

        try:
            if self._run_fut is not None:
                await self._run_fut
        except ContainerDeletedError:
            pass
        finally:
            try:
                if self.container_is_running():
                    assert self.process is not None
                    try:
                        log.info(f'{self} container is still running, killing crun process')
                        try:
                            await check_exec_output('crun', 'kill', '--all', self.name, 'SIGKILL')
                        except CalledProcessError as e:
                            not_extant_message = (
                                b'error opening file `/run/crun/'
                                + self.name.encode()
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
            finally:
                self._run_fut = None
                self._killed = True

    async def _cleanup(self):
        if self._cleaned_up:
            return

        assert self._run_fut is None
        try:
            if self.overlay_mounted:
                try:
                    await check_shell(f'umount -l {self.container_overlay_path}/merged')
                    self.overlay_mounted = False
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(f'while unmounting overlay in {self}', exc_info=True)

            if self.host_port is not None:
                assert port_allocator
                port_allocator.free(self.host_port)
                self.host_port = None

            if self.netns:
                assert network_allocator
                network_allocator.free(self.netns)
                self.netns = None
        finally:
            try:
                self.image.release()
            finally:
                self._cleaned_up = True

    async def remove(self):
        self.deleted_event.set()
        async with self._cleanup_lock:
            try:
                await self._kill()
            finally:
                await self._cleanup()

    async def _run_until_done_or_deleted(self, f: Callable[..., Awaitable[Any]], *args, **kwargs):
        try:
            return await run_until_done_or_deleted(self.deleted_event, f, *args, **kwargs)
        except StepInterruptedError as e:
            raise ContainerDeletedError from e

    def _step(self, name: str) -> ContextManager:
        return self.timings.step(name)

    async def _setup_overlay(self):
        lower_dir = self.image.rootfs_path
        upper_dir = f'{self.container_overlay_path}/upper'
        work_dir = f'{self.container_overlay_path}/work'
        merged_dir = f'{self.container_overlay_path}/merged'
        for d in (upper_dir, work_dir, merged_dir):
            os.makedirs(d)
        await check_shell(
            f'mount -t overlay overlay -o lowerdir={lower_dir},upperdir={upper_dir},workdir={work_dir} {merged_dir}'
        )
        self.overlay_mounted = True

    async def _setup_network_namespace(self):
        assert network_allocator
        assert port_allocator
        if self.network == 'private':
            self.netns = await network_allocator.allocate_private()
        else:
            assert self.network is None or self.network == 'public'
            self.netns = await network_allocator.allocate_public()

        if self.port is not None:
            self.host_port = await port_allocator.allocate()
            await self.netns.expose_port(self.port, self.host_port)

    async def _run_container(self) -> bool:
        self.started_at = time_msecs()
        try:
            await self._write_container_config()
            async with async_timeout.timeout(self.timeout):
                with open(self.log_path, 'w', encoding='utf-8') as container_log:
                    stdin = asyncio.subprocess.PIPE if self.stdin else None

                    self.process = await asyncio.create_subprocess_exec(
                        'crun',
                        'run',
                        '--bundle',
                        f'{self.container_overlay_path}/merged',
                        '--config',
                        f'{self.config_path}/config.json',
                        self.name,
                        stdin=stdin,
                        stdout=container_log,
                        stderr=container_log,
                    )

                    async with ResourceUsageMonitor(self.name, self.resource_usage_path):
                        if self.stdin is not None:
                            await self.process.communicate(self.stdin.encode('utf-8'))
                        await self.process.wait()
        except asyncio.TimeoutError:
            return True
        finally:
            self.finished_at = time_msecs()

        return False

    async def _write_container_config(self):
        os.makedirs(self.config_path)
        with open(f'{self.config_path}/config.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(await self.container_config()))

    # https://github.com/opencontainers/runtime-spec/blob/master/config.md
    async def container_config(self):
        assert self.image.image_config
        assert self.netns

        uid, gid = await self._get_in_container_user()
        weight = worker_fraction_in_1024ths(self.cpu_in_mcpu)
        workdir = self.image.image_config['Config']['WorkingDir']
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
                'args': self.command,
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
                        'limit': self.memory_in_bytes,
                        'reservation': self.memory_in_bytes,
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

        if self.unconfined:
            config['linux']['maskedPaths'] = []
            config['linux']['readonlyPaths'] = []
            config['process']['apparmorProfile'] = 'unconfined'
            config['linux']['seccomp'] = {'defaultAction': "SCMP_ACT_ALLOW"}

        return config

    async def _get_in_container_user(self):
        assert self.image.image_config
        user = self.image.image_config['Config']['User']
        if not user:
            return 0, 0
        if ":" in user:
            uid, gid = user.split(":")
        else:
            uid, gid = await self._read_user_from_rootfs(user)
        return int(uid), int(gid)

    async def _read_user_from_rootfs(self, user) -> Tuple[str, str]:
        with open(f'{self.image.rootfs_path}/etc/passwd', 'r', encoding='utf-8') as passwd:
            for record in passwd:
                if record.startswith(user):
                    _, _, uid, gid, _, _, _ = record.split(":")
                    return uid, gid
            raise ValueError("Container user not found in image's /etc/passwd")

    def _mounts(self, uid, gid):
        assert self.image.image_config
        assert self.netns
        # Only supports empty volumes
        external_volumes = []
        volumes = self.image.image_config['Config']['Volumes']
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
            self.volume_mounts
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
        assert self.image.image_config
        env = self.image.image_config['Config']['Env'] + self.env
        if self.port is not None:
            assert self.host_port is not None
            env.append(f'HAIL_BATCH_WORKER_PORT={self.host_port}')
            env.append(f'HAIL_BATCH_WORKER_IP={IP_ADDRESS}')
        return env

    # {
    #   name: str,
    #   state: str, (pending, running, succeeded, error, failed)
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

    def get_container_status(self) -> Optional[dict]:
        if not self.process:
            return None

        status: dict = {
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

    async def get_log(self, offset: Optional[int] = None):
        if os.path.exists(self.log_path):
            if offset is None:
                return (await self.fs.read(self.log_path)).decode()
            return (await self.fs.read_from(self.log_path, offset)).decode()
        return ''

    async def get_resource_usage(self) -> bytes:
        if os.path.exists(self.resource_usage_path):
            return await self.fs.read(self.resource_usage_path)
        return ResourceUsageMonitor.no_data()

    async def get_resource_usage_file_size(self) -> int:
        if os.path.exists(self.resource_usage_path):
            return os.path.getsize(self.resource_usage_path)
        return 0

    def __str__(self):
        return f'container {self.name}'


def populate_secret_host_path(host_path: str, secret_data: Optional[Dict[str, bytes]]):
    os.makedirs(host_path, exist_ok=True)
    if secret_data is not None:
        for filename, data in secret_data.items():
            with open(f'{host_path}/{filename}', 'wb') as f:
                f.write(base64.b64decode(data))


def copy_container(
    job: 'DockerJob',
    task_name: str,
    files: List[dict],
    volume_mounts: List[dict],
    cpu_in_mcpu: int,
    memory_in_bytes: int,
    scratch: str,
    requester_pays_project: str,
    client_session: httpx.ClientSession,
) -> Container:
    assert files
    assert job.worker.fs is not None

    command = [
        '/usr/bin/python3',
        '-m',
        'hailtop.aiotools.copy',
        json.dumps(requester_pays_project),
        '-',
        '-v',
    ]

    return Container(
        fs=job.worker.fs,
        name=job.container_name(task_name),
        image=Image(BATCH_WORKER_IMAGE, CopyStepCredentials(), client_session, job.pool),
        scratch_dir=f'{scratch}/{task_name}',
        command=command,
        cpu_in_mcpu=cpu_in_mcpu,
        memory_in_bytes=memory_in_bytes,
        volume_mounts=volume_mounts,
        env=[f'{job.credentials.cloud_env_name}={job.credentials.mount_path}'],
        stdin=json.dumps(files),
    )


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

        assert worker
        self.worker: Worker = worker

        self.deleted_event = asyncio.Event()

        self.token = uuid.uuid4().hex
        self.scratch = f'/batch/{self.token}'

        self.disk: Optional[CloudDisk] = None
        self.state = 'pending'
        self.error: Optional[str] = None

        self.start_time: Optional[int] = None
        self.end_time: Optional[int] = None

        self.marked_job_started = False

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

        requester_pays_project = job_spec.get('requester_pays_project')
        cloudfuse = job_spec.get('cloudfuse') or job_spec.get('gcsfuse')
        self.cloudfuse = cloudfuse
        if cloudfuse:
            for config in cloudfuse:
                if requester_pays_project:
                    config['requester_pays_project'] = requester_pays_project
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

        self.mjs_fut: Optional[asyncio.Future] = None

    def write_batch_config(self):
        os.makedirs(f'{self.scratch}/batch-config')
        with open(f'{self.scratch}/batch-config/batch-config.json', 'wb') as config:
            config.write(orjson.dumps({'version': 1, 'batch_id': self.batch_id}))

    @property
    def job_id(self):
        return self.job_spec['job_id']

    @property
    def attempt_id(self):
        return self.job_spec['attempt_id']

    @property
    def id(self):
        return (self.batch_id, self.job_id)

    @property
    def deleted(self):
        return self.deleted_event.is_set()

    async def run(self):
        pass

    async def get_log(self):
        pass

    async def get_resource_usage(self) -> Dict[str, Optional[bytes]]:
        raise NotImplementedError

    async def delete(self):
        log.info(f'deleting {self}')
        self.deleted_event.set()

    def mark_started(self):
        self.mjs_fut = self.task_manager.ensure_future(self.worker.post_job_started(self))

    async def mark_complete(self):
        self.end_time = time_msecs()

        full_status = self.status()

        if self.format_version.has_full_status_in_gcs():
            assert self.worker.file_store
            await retry_transient_errors(
                self.worker.file_store.write_status_file,
                self.batch_id,
                self.job_id,
                self.attempt_id,
                json.dumps(full_status),
            )

        if not self.deleted:
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

    def done(self):
        return self.state in ('succeeded', 'error', 'failed')

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
        assert worker.fs

        input_files = job_spec.get('input_files')
        output_files = job_spec.get('output_files')

        requester_pays_project = job_spec.get('requester_pays_project')

        self.timings: Timings = Timings()

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
        containers: Dict[str, Container] = {}

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
            )

        assert self.worker.fs
        containers['main'] = Container(
            fs=self.worker.fs,
            name=self.container_name('main'),
            image=Image(job_spec['process']['image'], self.credentials, client_session, pool),
            scratch_dir=f'{self.scratch}/main',
            command=job_spec['process']['command'],
            cpu_in_mcpu=self.cpu_in_mcpu,
            memory_in_bytes=self.memory_in_bytes,
            network=job_spec.get('network'),
            port=job_spec.get('port'),
            timeout=job_spec.get('timeout'),
            unconfined=job_spec.get('unconfined'),
            volume_mounts=self.main_volume_mounts,
            env=[f'{var["name"]}={var["value"]}' for var in self.env],
        )

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
            )

        self.containers = containers

    def step(self, name: str) -> ContextManager:
        return self.timings.step(name)

    def container_name(self, task_name: str):
        return f'batch-{self.batch_id}-job-{self.job_id}-{task_name}'

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

    async def run_container(self, container: Container, task_name: str):
        async def on_completion():
            with container._step('uploading_log'):
                assert self.worker.file_store
                await self.worker.file_store.write_log_file(
                    self.format_version,
                    self.batch_id,
                    self.job_id,
                    self.attempt_id,
                    task_name,
                    await container.get_log(),
                )

            with container._step('uploading_resource_usage'):
                await self.worker.file_store.write_resource_usage_file(
                    self.format_version,
                    self.batch_id,
                    self.job_id,
                    self.attempt_id,
                    task_name,
                    await container.get_resource_usage(),
                )

        try:
            await container.run(on_completion)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def run(self):
        async with self.worker.cpu_sem(self.cpu_in_mcpu):
            self.start_time = time_msecs()

            try:
                self.mark_started()

                self.state = 'initializing'

                os.makedirs(f'{self.scratch}/')
                self.write_batch_config()

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
                    await self.run_container(input, 'input')

                if not input or input.state == 'succeeded':
                    main = self.containers['main']
                    await self.run_container(main, 'main')

                    output = self.containers.get('output')

                    always_copy_output = self.job_spec.get('always_copy_output', True)
                    copy_output = output and (main.state == 'succeeded' or always_copy_output)

                    if copy_output:
                        assert output
                        await self.run_container(output, 'output')

                    if main.state != 'succeeded':
                        self.state = main.state
                    elif copy_output:
                        assert output
                        self.state = output.state
                    else:
                        self.state = 'succeeded'
                else:
                    self.state = input.state
            except asyncio.CancelledError:
                raise
            except ContainerDeletedError:
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
                        _, exc, _ = sys.exc_info()
                        if not isinstance(exc, asyncio.CancelledError):
                            await self.mark_complete()

    async def cleanup(self):
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
        logs = {}
        for name, container in self.containers.items():
            c_log = await container.get_log()
            if c_log is None:
                c_log = ''
            logs[name] = c_log
        return logs

    async def get_resource_usage(self):
        return {name: await c.get_resource_usage() for name, c in self.containers.items()}

    async def get_resource_usage_file_sizes(self):
        return {name: await c.get_resource_usage_file_size() for name, c in self.containers.items()}

    async def delete(self):
        await super().delete()
        await asyncio.wait([c.remove() for c in self.containers.values()])

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

        assert job_spec['process']['jar_spec']['type'] == 'jar_url'
        self.jar_url = job_spec['process']['jar_spec']['value']
        self.argv = job_spec['process']['command']

        self.timings = Timings()
        self.state = 'pending'

        self.jvm: Optional[JVM] = None
        self.jvm_name: Optional[str] = None

        self.log_file = f'{self.scratch}/log'

        assert self.worker.fs is not None

    def step(self, name):
        return self.timings.step(name)

    async def run_until_done_or_deleted(self, f: Callable[..., Awaitable[Any]], *args, **kwargs):
        try:
            return await run_until_done_or_deleted(self.deleted_event, f, *args, **kwargs)
        except StepInterruptedError as e:
            raise JobDeletedError from e

    def secret_host_path(self, secret):
        return f'{self.scratch}/secrets/{secret["mount_path"]}'

    async def download_jar(self):
        assert self.worker
        assert self.worker.pool

        async with self.worker.jar_download_locks[self.jar_url]:
            unique_key = self.jar_url.replace('_', '__').replace('/', '_')
            local_jar_location = f'/hail-jars/{unique_key}.jar'
            if not os.path.isfile(local_jar_location):
                assert self.jar_url.startswith(ACCEPTABLE_QUERY_JAR_URL_PREFIX)

                async def download_jar():
                    temporary_file = tempfile.NamedTemporaryFile(delete=False)  # pylint: disable=consider-using-with
                    try:
                        assert self.worker.fs is not None
                        async with await self.worker.fs.open(self.jar_url) as jar_data:
                            while True:
                                b = await jar_data.read(256 * 1024)
                                if not b:
                                    break
                                written = await blocking_to_async(self.worker.pool, temporary_file.write, b)
                                assert written == len(b)
                        temporary_file.close()
                        os.rename(temporary_file.name, local_jar_location)
                    finally:
                        temporary_file.close()  # close is idempotent
                        try:
                            await blocking_to_async(self.worker.pool, os.remove, temporary_file.name)
                        except OSError as err:
                            if err.errno != errno.ENOENT:
                                raise

                await retry_transient_errors(download_jar)

            return local_jar_location

    async def run(self):
        async with self.worker.cpu_sem(self.cpu_in_mcpu):
            self.start_time = time_msecs()
            os.makedirs(f'{self.scratch}/')
            self.write_batch_config()

            try:
                with self.step('connecting_to_jvm'):
                    self.jvm = await self.worker.borrow_jvm(self.cpu_in_mcpu // 1000)
                    self.jvm_name = str(self.jvm)

                self.mark_started()

                self.state = 'initializing'

                await check_shell_output(f'xfs_quota -x -c "project -s -p {self.scratch} {self.project_id}" /host/')
                await check_shell_output(
                    f'xfs_quota -x -c "limit -p bsoft={self.data_disk_storage_in_gib} bhard={self.data_disk_storage_in_gib} {self.project_id}" /host/'
                )

                if self.secrets:
                    for secret in self.secrets:
                        populate_secret_host_path(self.secret_host_path(secret), secret['data'])

                self.state = 'running'

                with self.step('downloading_jar'):
                    local_jar_location = await self.download_jar()

                with self.step('running'):
                    await self.jvm.execute(local_jar_location, self.scratch, self.log_file, self.jar_url, self.argv)

                self.state = 'succeeded'
            except asyncio.CancelledError:
                raise
            except JVMUserError:
                self.state = 'failed'
                self.error = traceback.format_exc()
                await self.cleanup()
            except JobDeletedError:
                self.state = 'cancelled'
                await self.cleanup()
            except JVMCreationError:
                self.state = 'error'
                log.exception(f'while running {self}')
                await self.cleanup()
                raise
            except Exception:
                log.exception(f'while running {self}')
                self.state = 'error'
                self.error = traceback.format_exc()
                await self.cleanup()
            else:
                await self.cleanup()
            finally:
                _, exc, _ = sys.exc_info()
                if not isinstance(exc, asyncio.CancelledError):
                    await self.mark_complete()

    async def cleanup(self):
        assert self.worker
        assert self.worker.file_store is not None
        assert self.worker.fs

        if self.jvm is not None:
            self.worker.return_jvm(self.jvm)
            self.jvm = None

        with self.step('uploading_log'):
            await self.worker.file_store.write_log_file(
                self.format_version, self.batch_id, self.job_id, self.attempt_id, 'main', await self._get_log()
            )

        try:
            await check_shell(f'xfs_quota -x -c "limit -p bsoft=0 bhard=0 {self.project_id}" /host')
            await blocking_to_async(self.pool, shutil.rmtree, self.scratch, ignore_errors=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('while deleting volumes')

    async def _get_log(self):
        assert self.worker
        assert self.worker.fs is not None
        if os.path.exists(self.log_file):
            return (await self.worker.fs.read(self.log_file)).decode()
        return ''

    async def get_log(self):
        return {'main': await self._get_log()}

    async def get_resource_usage(self):
        return {'main': ResourceUsageMonitor.no_data()}

    async def delete(self):
        await super().delete()
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


class JVMCreationError(Exception):
    pass


class JVMUserCredentials:
    pass


class CopyStepCredentials:
    pass


class JVMContainer:
    @staticmethod
    async def create_and_start(
        index: int,
        n_cores: int,
        socket_file: str,
        root_dir: str,
        client_session: httpx.ClientSession,
        pool: concurrent.futures.ThreadPoolExecutor,
    ):
        assert os.path.commonpath([socket_file, root_dir]) == root_dir
        assert os.path.isdir(root_dir)

        total_memory_bytes = n_cores * worker_memory_per_core_bytes(CLOUD, instance_config.worker_type())

        # We allocate 60% of memory per core to off heap memory
        memory_per_core_mib = worker_memory_per_core_mib(CLOUD, instance_config.worker_type())
        memory_mib = n_cores * memory_per_core_mib
        heap_memory_mib = int(0.4 * memory_mib)
        off_heap_memory_per_core_mib = memory_mib - heap_memory_mib

        command = [
            'java',
            f'-Xmx{heap_memory_mib}M',
            '-cp',
            f'/jvm-entryway:/jvm-entryway/junixsocket-selftest-2.3.3-jar-with-dependencies.jar:{JVM.SPARK_HOME}/jars/*',
            'is.hail.JVMEntryway',
            socket_file,
        ]

        volume_mounts = [
            {
                'source': JVM.SPARK_HOME,
                'destination': JVM.SPARK_HOME,
                'type': 'none',
                'options': ['rbind', 'rw'],
            },
            {
                'source': '/jvm-entryway',
                'destination': '/jvm-entryway',
                'type': 'none',
                'options': ['rbind', 'rw'],
            },
            {
                'source': '/hail-jars',
                'destination': '/hail-jars',
                'type': 'none',
                'options': ['rbind', 'rw'],
            },
            {
                'source': root_dir,
                'destination': root_dir,
                'type': 'none',
                'options': ['rbind', 'rw'],
            },
            {
                'source': '/batch',
                'destination': '/batch',
                'type': 'none',
                'options': ['rbind', 'rw'],
            },
        ]

        fs = LocalAsyncFS(pool)  # worker does not have a fs when initializing JVMs

        c = Container(
            fs=fs,
            name=f'jvm-{index}',
            image=Image(BATCH_WORKER_IMAGE, JVMUserCredentials(), client_session, pool),
            scratch_dir=f'{root_dir}/container',
            command=command,
            cpu_in_mcpu=n_cores * 1000,
            memory_in_bytes=total_memory_bytes,
            env=[f'HAIL_WORKER_OFF_HEAP_MEMORY_PER_CORE_MB={off_heap_memory_per_core_mib}', f'HAIL_CLOUD={CLOUD}'],
            volume_mounts=volume_mounts,
        )

        await c.create()
        await c.start()

        return JVMContainer(c, fs)

    def __init__(self, container: Container, fs: LocalAsyncFS):
        self.container = container
        self.fs: Optional[LocalAsyncFS] = fs

    @property
    def returncode(self) -> Optional[int]:
        if self.container.process is None:
            return None
        return self.container.process.returncode

    async def remove(self):
        if self.fs is not None:
            await self.fs.close()
            self.fs = None
        await self.container.remove()


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
    async def create_container_and_connect(
        cls,
        index: int,
        n_cores: int,
        socket_file: str,
        root_dir: str,
        client_session: httpx.ClientSession,
        pool: concurrent.futures.ThreadPoolExecutor,
    ) -> JVMContainer:
        try:
            container = await JVMContainer.create_and_start(index, n_cores, socket_file, root_dir, client_session, pool)

            attempts = 0
            delay = 0.25

            while True:
                try:
                    if attempts % 8 == 0:
                        log.info(
                            f'JVM-{index}: trying to establish connection; elapsed time: {attempts * delay} seconds'
                        )

                    reader, writer = await asyncio.open_unix_connection(socket_file)
                    try:
                        b = await read_bool(reader)
                        assert b, f'expected true, got {b}'
                        writer.write(b'\0x01')
                        break
                    finally:
                        writer.close()
                except (FileNotFoundError, ConnectionRefusedError) as err:
                    attempts += 1
                    if attempts == 240:
                        jvm_output = await container.container.get_log() or ''
                        raise ValueError(
                            f'JVM-{index}: failed to establish connection after {240 * delay} seconds. '
                            'JVM output:\n\n' + jvm_output
                        ) from err
                    await asyncio.sleep(delay)
            return container
        except Exception as e:
            raise JVMCreationError from e

    @classmethod
    async def create(cls, index: int, n_cores: int, worker: 'Worker'):
        token = uuid.uuid4().hex
        root_dir = f'/host/jvm-{token}'
        socket_file = root_dir + '/socket'
        output_file = root_dir + '/output'
        should_interrupt = asyncio.Event()
        await blocking_to_async(worker.pool, os.makedirs, root_dir)
        container = await cls.create_container_and_connect(
            index, n_cores, socket_file, root_dir, worker.client_session, worker.pool
        )
        return cls(
            index,
            n_cores,
            socket_file,
            root_dir,
            output_file,
            should_interrupt,
            container,
            worker.client_session,
            worker.pool,
        )

    def __init__(
        self,
        index: int,
        n_cores: int,
        socket_file: str,
        root_dir: str,
        output_file: str,
        should_interrupt: asyncio.Event,
        container: JVMContainer,
        client_session: httpx.ClientSession,
        pool: concurrent.futures.ThreadPoolExecutor,
    ):
        self.index = index
        self.n_cores = n_cores
        self.socket_file = socket_file
        self.root_dir = root_dir
        self.output_file = output_file
        self.should_interrupt = should_interrupt
        self.container = container
        self.client_session = client_session
        self.pool = pool

    def __str__(self):
        return f'JVM-{self.index}'

    def __repr__(self):
        return f'JVM-{self.index}'

    def interrupt(self):
        self.should_interrupt.set()

    def reset(self):
        self.should_interrupt.clear()

    async def kill(self):
        if self.container is not None:
            await self.container.remove()

    async def new_connection(self):
        while True:
            try:
                return await asyncio.open_unix_connection(self.socket_file)
            except ConnectionRefusedError:
                os.remove(self.socket_file)
                if self.container:
                    await self.container.remove()

                await blocking_to_async(self.pool, shutil.rmtree, f'{self.root_dir}/container', ignore_errors=True)

                container = await self.create_container_and_connect(
                    self.index, self.n_cores, self.socket_file, self.root_dir, self.client_session, self.pool
                )
                self.container = container

    async def execute(self, classpath: str, scratch_dir: str, log_file: str, jar_url: str, argv: List[str]):
        assert worker is not None

        with ExitStack() as stack:
            reader: asyncio.StreamReader
            writer: asyncio.StreamWriter
            reader, writer = await self.new_connection()
            stack.callback(writer.close)

            command = [classpath, 'is.hail.backend.service.Main', scratch_dir, log_file, jar_url, *argv]

            write_int(writer, len(command))
            for part in command:
                assert isinstance(part, str)
                write_str(writer, part)
            await writer.drain()

            wait_for_message_from_container: asyncio.Future = asyncio.ensure_future(read_int(reader))
            stack.callback(wait_for_message_from_container.cancel)
            wait_for_interrupt: asyncio.Future = asyncio.ensure_future(self.should_interrupt.wait())
            stack.callback(wait_for_interrupt.cancel)

            await asyncio.wait(
                [wait_for_message_from_container, wait_for_interrupt], return_when=asyncio.FIRST_COMPLETED
            )

            if wait_for_interrupt.done():
                await wait_for_interrupt  # retrieve exceptions
                if not wait_for_message_from_container.done():
                    write_int(writer, 0)  # tell process to cancel
                    await writer.drain()

            eos_exception = None
            try:
                message = await wait_for_message_from_container
            except EndOfStream as exc:
                try:
                    await self.kill()
                except ProcessLookupError:
                    log.warning(f'{self}: JVM died after we received EOS')
                message = JVM.FINISH_JVM_EOS
                eos_exception = exc

            if message == JVM.FINISH_NORMAL:
                pass
            elif message == JVM.FINISH_CANCELLED:
                assert wait_for_interrupt.done()
                raise JobDeletedError
            elif message == JVM.FINISH_USER_EXCEPTION:
                exception = await read_str(reader)
                raise JVMUserError(exception)
            elif message == JVM.FINISH_ENTRYWAY_EXCEPTION:
                log.warning(f'{self}: entryway exception encountered (interrupted: {wait_for_interrupt.done()})')
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
        self.fs: Optional[RouterAsyncFS] = None
        self.file_store: Optional[FileStore] = None
        self.headers: Optional[Dict[str, str]] = None
        self.compute_client = None

        self._jvm_initializer_task = asyncio.ensure_future(self._initialize_jvms())
        self._jvms: SortedSet[JVM] = SortedSet([], key=lambda jvm: jvm.n_cores)

    async def _initialize_jvms(self):
        if instance_config.worker_type() in ('standard', 'D', 'highmem', 'E'):
            jvms: List[Awaitable[JVM]] = []
            for jvm_cores in (1, 2, 4, 8):
                for _ in range(CORES // jvm_cores):
                    jvms.append(JVM.create(len(jvms), jvm_cores, self))
            self._jvms.update(await asyncio.gather(*jvms))
        log.info(f'JVMs initialized {self._jvms}')

    async def borrow_jvm(self, n_cores: int) -> JVM:
        if instance_config.worker_type() not in ('standard', 'D', 'highmem', 'E'):
            raise ValueError(f'no JVMs available on {instance_config.worker_type()}')
        await self._jvm_initializer_task
        assert self._jvms
        index = self._jvms.bisect_key_left(n_cores)
        assert index < len(self._jvms), index
        return self._jvms.pop(index)

    def return_jvm(self, jvm: JVM):
        jvm.reset()
        self._jvms.add(jvm)

    async def shutdown(self):
        log.info('Worker.shutdown')
        try:
            async with AsyncExitStack() as cleanup:
                for jvm in self._jvms:
                    cleanup.callback(jvm.kill)
        finally:
            try:
                await self.task_manager.shutdown_and_wait()
                log.info('shutdown task manager')
            finally:
                try:
                    if self.file_store:
                        await self.file_store.close()
                        log.info('closed file store')
                finally:
                    try:
                        if self.compute_client:
                            await self.compute_client.close()
                            log.info('closed compute client')
                    finally:
                        try:
                            if self.fs:
                                await self.fs.close()
                                log.info('closed worker file system')
                        finally:
                            await self.client_session.close()
                            log.info('closed client session')

    async def run_job(self, job):
        try:
            await job.run()
        except asyncio.CancelledError:
            raise
        except JVMCreationError:
            self.stop_event.set()
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

        assert self.file_store
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

        log.info(f'created {job} attempt {job.attempt_id}')

        self.jobs[job.id] = job

        self.task_manager.ensure_future(self.run_job(job))

        return web.Response()

    async def create_job(self, request):
        if not self.active:
            raise web.HTTPServiceUnavailable
        return await asyncio.shield(self.create_job_1(request))

    def _job_from_request(self, request):
        batch_id = int(request.match_info['batch_id'])
        job_id = int(request.match_info['job_id'])
        id = (batch_id, job_id)
        job = self.jobs.get(id)
        if not job:
            raise web.HTTPNotFound()
        return job

    async def get_job_log(self, request):
        if not self.active:
            raise web.HTTPServiceUnavailable
        job = self._job_from_request(request)
        return web.json_response(await job.get_log())

    async def get_job_resource_usage(self, request):
        if not self.active:
            raise web.HTTPServiceUnavailable

        job = self._job_from_request(request)
        resource_usage = await job.get_resource_usage()
        data = {task: base64.b64encode(df).decode('utf-8') for task, df in resource_usage.items()}
        return web.json_response(data)

    async def get_job_status(self, request):
        if not self.active:
            raise web.HTTPServiceUnavailable
        job = self._job_from_request(request)
        return web.json_response(job.status())

    async def delete_job_1(self, request):
        batch_id = int(request.match_info['batch_id'])
        job_id = int(request.match_info['job_id'])
        id = (batch_id, job_id)

        if id not in self.jobs:
            raise web.HTTPNotFound()

        log.info(f'deleting job {id}, removing from jobs')

        async def delete_then_remove_job():
            await self.jobs[id].delete()
            del self.jobs[id]

        self.last_updated = time_msecs()

        self.task_manager.ensure_future(delete_then_remove_job())

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
                web.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/resource_usage', self.get_job_resource_usage),
                web.get('/api/v1alpha/batches/{batch_id}/jobs/{job_id}/status', self.get_job_status),
                web.get('/healthcheck', self.healthcheck),
            ]
        )

        self.task_manager.ensure_future(periodically_call(60, self.cleanup_old_images))

        app_runner = web.AppRunner(app, access_log_class=BatchWorkerAccessLogger)
        await app_runner.setup()
        site = web.TCPSite(app_runner, '0.0.0.0', 5000)
        await site.start()

        try:
            await asyncio.wait_for(self.activate(), MAX_IDLE_TIME_MSECS / 1000)
        except asyncio.TimeoutError:
            log.exception(f'could not activate after trying for {MAX_IDLE_TIME_MSECS} ms, exiting')
            return

        self.task_manager.ensure_future(periodically_call(60, self.send_billing_update))
        self.task_manager.ensure_future(periodically_call(60, self.monitor_resource_usage))

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

    async def post_job_complete_1(self, job: Job, full_status):
        assert job.end_time
        assert job.start_time
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

        body = {
            'status': status,
            'marked_job_started': job.marked_job_started,
        }

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
        if job.mjs_fut is not None:
            await job.mjs_fut
        try:
            await self.post_job_complete_1(job, full_status)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(f'error while marking {job} complete', stack_info=True)
        finally:
            log.info(
                f'{job} attempt {job.attempt_id} marked complete after {time_msecs() - job.end_time}ms: {job.state}'
            )
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

        async def post_started_if_job_still_running():
            # If the job is already complete, just send MJC. No need for MJS
            if not job.done():
                url = deploy_config.url('batch-driver', '/api/v1alpha/instances/job_started')
                await self.client_session.post(url, json=body, headers=self.headers)

        await retry_transient_errors(post_started_if_job_still_running)

    async def post_job_started(self, job):
        try:
            await self.post_job_started_1(job)
            job.marked_job_started = True
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
            assert image_lock
            async with image_lock.writer:
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

    async def send_billing_update(self):
        async def update():
            update_timestamp = time_msecs()
            running_attempts = []
            for (batch_id, job_id), job in self.jobs.items():
                if not job.marked_job_started or job.end_time is not None:
                    continue
                running_attempts.append(
                    {
                        'batch_id': batch_id,
                        'job_id': job_id,
                        'attempt_id': job.attempt_id,
                    }
                )

            if running_attempts:
                billing_update_data = {'timestamp': update_timestamp, 'attempts': running_attempts}

                await self.client_session.post(
                    deploy_config.url('batch-driver', '/api/v1alpha/billing_update'),
                    json=billing_update_data,
                    headers=self.headers,
                )
                log.info(f'sent billing update for {time_msecs_str(update_timestamp)}')

        await retry_transient_errors(update)

    async def monitor_resource_usage(self):
        stdout, _ = await check_shell_output('xfs_quota -x -c "report -h -p" /host/; df -kh')
        log.info(stdout)
        for job in self.jobs.values():
            if isinstance(job, DockerJob):
                file_sizes = await job.get_resource_usage_file_sizes()
                file_sizes = {name: humanize.naturalsize(size) for name, size in file_sizes.items()}
                log.info(f'{job} {file_sizes}')


async def async_main():
    global port_allocator, network_allocator, worker, docker, image_lock

    image_lock = aiorwlock.RWLock()
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
