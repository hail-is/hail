import os
import sys
from shlex import quote as shq
import time
import logging
import asyncio
import random
import json
import traceback
import base64
import uuid
import shutil
import aiohttp
from aiohttp import web
import concurrent
import aiodocker
from aiodocker.exceptions import DockerError
from hailtop.utils import request_retry_transient_errors

# import uvloop

from hailtop.config import DeployConfig
from gear import configure_logging

from .utils import parse_cpu_in_mcpu, parse_image_tag
from .semaphore import WeightedSemaphore
from .log_store import LogStore
from .google_storage import GCS

# uvloop.install()

configure_logging()
log = logging.getLogger('batch2-agent')

docker = aiodocker.Docker()

MAX_IDLE_TIME_WITH_PODS = 60 * 2  # seconds
MAX_IDLE_TIME_WITHOUT_PODS = 60 * 1  # seconds


async def docker_call_retry(f, *args, **kwargs):
    delay = 0.1
    while True:
        try:
            return await f(*args, **kwargs)
        except DockerError as e:
            # 408 request timeout, 503 service unavailable
            if e.status == 408 or e.status == 503:
                log.exception('in docker call, retrying')
            else:
                raise
        except asyncio.TimeoutError:
            log.exception('in docker call, retrying')
        # exponentially back off, up to (expected) max of 30s
        t = delay * random.random()
        await asyncio.sleep(t)
        delay = min(delay * 2, 60.0)


class Container:
    def __init__(self, pod, name, spec):
        self.pod = pod
        self.name = name
        self.spec = spec

        image = spec['image']
        tag = parse_image_tag(self.spec['image'])
        if not tag:
            log.info(f'adding latest tag to image {self.spec["image"]} for container {self.pod.name}/{self.name}')
            image += ':latest'
        self.image = image

        self.cpu_in_mcpu = parse_cpu_in_mcpu(spec['cpu'])

        self.container = None
        self.state = 'pending'
        self.error = None
        self.container_status = None
        self.log = None

    def container_config(self):
        config = {
            "AttachStdin": False,
            "AttachStdout": False,
            "AttachStderr": False,
            "Tty": False,
            'OpenStdin': False,
            'Cmd': self.spec['command'],
            'Image': self.image,
            'HostConfig': {'CpuPeriod': 100000,
                           'CpuQuota': self.cpu_in_mcpu * 100}
        }

        volume_mounts = self.spec.get('volume_mounts')
        if volume_mounts:
            config['HostConfig']['Binds'] = volume_mounts

        return config

    async def get_container_status(self):
        c = await docker_call_retry(self.container.show)
        log.info(f'container {self.pod.name}/{self.name} info {c}')
        cstate = c['State']
        status = {
            'state': cstate['Status'],
            'started_at': cstate['StartedAt'],
            'finished_at': cstate['FinishedAt']
        }
        cerror = cstate['Error']
        if cerror:
            status['error'] = cerror
        else:
            status['exit_code'] = cstate['ExitCode']
        return status

    async def run(self, worker):
        try:
            log.info(f'container {self.pod.name}/{self.name}: pulling {self.image}')
            self.state = 'pulling'

            try:
                await docker_call_retry(docker.images.get, self.image)
            except DockerError as e:
                if e.status == 404:
                    await docker_call_retry(docker.images.pull, self.image)

            log.info(f'container {self.pod.name}/{self.name}: creating')
            self.state = 'creating'

            self.container = await docker_call_retry(docker.containers.create, self.container_config())

            async with worker.cpu_sem(self.cpu_in_mcpu):
                log.info(f'container {self.pod.name}/{self.name}: starting')
                self.state = 'starting'

                await docker_call_retry(self.container.start)

                log.info(f'container {self.pod.name}/{self.name}: running')
                self.state = 'running'

                await docker_call_retry(self.container.wait)

            self.container_status = await self.get_container_status()
            log.info(f'container {self.pod.name}/{self.name}: container status {self.container_status}')

            log.info(f'container {self.pod.name}/{self.name}: uploading log')
            self.state = 'uploading_log'

            self.log = await self.get_container_log()
            log_path = LogStore.container_log_path(self.pod.output_directory, self.name)
            await worker.gcs_client.write_gs_file(log_path, self.log)

            log.info(f'container {self.pod.name}/{self.name}: deleting')

            await docker_call_retry(self.container.stop)
            await docker_call_retry(self.container.delete)
            self.container = None

            if 'error' in self.container_status:
                self.state = 'error'
            elif self.container_status['exit_code'] == 0:
                self.state = 'succeeded'
            else:
                self.state = 'failed'
        except Exception:
            log.exception(f'while running container {self.pod.name}/{self.name}')

            self.state = 'error'
            self.error = traceback.format_exc()
        finally:
            if self.container:
                log.exception(f'cleaning up container')
                try:
                    await docker_call_retry(self.container.stop)
                    await docker_call_retry(self.container.delete)
                except Exception:
                    log.exception('while deleting container')
                finally:
                    self.container = None

    async def get_container_log(self):
        logs = await docker_call_retry(self.container.log, stderr=True, stdout=True)
        return "".join(logs)

    async def get_log(self):
        if self.log:
            return self.log

        if self.container:
            return await self.get_container_log()

        return None

    # {
    #   name: str,
    #   state: str,
    #   error: str, (optional)
    #   container_status: { (from docker container state)
    #     state: str,
    #     started_at: str, (date)
    #     finished_at: str, (date)
    #     error: str, (one of error, exit_code will be present)
    #     exit_code: int
    #   }
    # }
    async def status(self, state=None):
        if not state:
            state = self.state
        status = {
            'name': self.name,
            'state': state
        }
        if self.error:
            status['error'] = self.error
        if self.container_status:
            status['container_status'] = self.container_status
        elif self.container:
            status['container_status'] = await self.get_container_status()
        return status


def populate_secret_host_path(host_path, secret_data):
    os.makedirs(host_path)
    if secret_data is not None:
        for filename, data in secret_data.items():
            with open(f'{host_path}/{filename}', 'w') as f:
                f.write(base64.b64decode(data).decode())


def copy_command(src, dst):
    if not dst.startswith('gs://'):
        mkdirs = f'mkdir -p {shq(os.path.dirname(dst))};'
    else:
        mkdirs = ""
    return f'{mkdirs} gsutil -m cp -R {shq(src)} {shq(dst)}'


def copy(files):
    if files is None:
        return 'true'

    authenticate = 'set -ex; gcloud -q auth activate-service-account --key-file=/gsa-key/privateKeyData'
    copies = ' && '.join([copy_command(f['from'], f['to']) for f in files])
    return f'{authenticate} && {copies}'


def copy_container(pod, name, files, volume_mounts):
    sh_expression = copy(files)
    copy_spec = {
        'image': 'google/cloud-sdk:237.0.0-alpine',
        'name': name,
        'command': ['/bin/sh', '-c', sh_expression],
        'cpu': '500m' if files else '100m',
        'volume_mounts': volume_mounts
    }
    return Container(pod, name, copy_spec)


class Pod:
    def secret_host_path(self, secret):
        return f'{self.scratch}/{secret["name"]}'

    def __init__(self, name, batch_id, user, job_spec, output_directory):
        self.name = name
        self.batch_id = batch_id
        self.user = user
        self.job_spec = job_spec
        self.output_directory = output_directory

        token = uuid.uuid4().hex
        self.scratch = f'/batch/pods/{token}'

        self.state = 'pending'
        self.error = None

        pvc_size = job_spec.get('pvc_size')
        input_files = job_spec.get('input_files')
        output_files = job_spec.get('output_files')

        copy_volume_mounts = []
        main_volume_mounts = []

        if job_spec.get('mount_docker_socket'):
            main_volume_mounts.append('/var/run/docker.sock:/var/run/docker.sock')

        if pvc_size or input_files or output_files:
            self.mount_io = True
            volume_mount = 'io:/io'
            main_volume_mounts.append(volume_mount)
            copy_volume_mounts.append(volume_mount)
        else:
            self.mount_io = False

        secrets = job_spec.get('secrets')
        self.secrets = secrets
        if secrets:
            for secret in job_spec['secrets']:
                volume_mount = f'{self.secret_host_path(secret)}:{secret["mount_path"]}'
                main_volume_mounts.append(volume_mount)
                # this will be the user gsa-key
                if secret.get('mount_in_copy', False):
                    copy_volume_mounts.append(volume_mount)

        # create containers
        containers = {}

        # FIXME
        # if input_files:
        containers['setup'] = copy_container(
            self, 'setup', input_files, copy_volume_mounts)

        # main container
        main_spec = {
            'command': job_spec['command'],
            'image': job_spec['image'],
            'name': 'main',
            # FIXME env
            'cpu': job_spec['resources']['cpu'],
            'volume_mounts': main_volume_mounts
        }
        containers['main'] = Container(self, 'main', main_spec)

        # FIXME
        # if output_files:
        containers['cleanup'] = copy_container(
            self, 'cleanup', output_files, copy_volume_mounts)

        self.containers = containers

    async def run(self, worker):
        io = None
        try:
            log.info(f'pod {self.name}: initializing')
            self.state = 'initializing'

            if self.mount_io:
                io = await docker_call_retry(docker.volumes.create, {'Name': 'io'})

            if self.secrets:
                for secret in self.secrets:
                    populate_secret_host_path(self.secret_host_path(secret), secret['data'])

            self.state = 'running'

            log.info(f'pod {self.name}: running setup')

            setup = self.containers['setup']
            await setup.run(worker)

            log.info(f'pod {self.name} setup: {setup.state}')

            if setup.state == 'succeeded':
                log.info(f'pod {self.name}: running main')

                main = self.containers['main']
                await main.run(worker)

                log.info(f'pod {self.name} main: {main.state}')

                log.info(f'pod {self.name}: running cleanup')

                cleanup = self.containers['cleanup']
                await cleanup.run(worker)

                log.info(f'pod {self.name} cleanup: {cleanup.state}')

                if main.state != 'succeeded':
                    self.state = main.state
                else:
                    if cleanup:
                        self.state = cleanup.state
                    else:
                        self.state = 'succeeded'
            else:
                self.state = setup.state

            log.info(f'pod {self.name}: uploading status')

            await worker.gcs_client.write_gs_file(
                LogStore.pod_status_path(self.output_directory),
                json.dumps(await self.status(), indent=4))
        except Exception:
            log.exception(f'while running pod {self.name}')

            self.state = 'error'
            self.error = traceback.format_exc()
        finally:
            log.info(f'pod {self.name}: marking complete')
            await worker.post_pod_complete(await self.status())

            log.info(f'pod {self.name}: cleaning up')
            try:
                shutil.rmtree(self.scratch, ignore_errors=True)
                if io:
                    await docker_call_retry(io.delete)
            except Exception:
                log.exception('while deleting volumes')

    async def get_log(self):
        return {name: await c.get_log() for name, c in self.containers.items()}

    # {
    #   name: str,
    #   batch_id: int,
    #   job_id: int,
    #   user: str,
    #   state: str,
    #   error: str, (optional)
    #   container_statuses: [Container.status]
    # }
    async def status(self):
        status = {
            'name': self.name,
            'batch_id': self.batch_id,
            'job_id': self.job_spec['job_id'],
            'user': self.user,
            'state': self.state
        }
        if self.error:
            status['error'] = self.error
        status['container_statuses'] = {
            name: await c.status() for name, c in self.containers.items()
        }
        return status


class Worker:
    def __init__(self, image, cores, deploy_config, token, ip_address):
        self.image = image
        self.cores_mcpu = cores * 1000
        self.deploy_config = deploy_config
        self.token = token
        self.free_cores_mcpu = self.cores_mcpu
        self.last_updated = time.time()
        self.pods = {}
        self.cpu_sem = WeightedSemaphore(self.cores_mcpu)
        self.ip_address = ip_address

        pool = concurrent.futures.ThreadPoolExecutor()

        self.gcs_client = GCS(pool)

    async def _create_pod(self, parameters):
        name = parameters['name']
        # FIXME think through logic to make sure this is sensible
        if name in self.pods:
            return

        batch_id = parameters['batch_id']
        user = parameters['user']
        job_spec = parameters['job_spec']
        output_directory = parameters['output_directory']

        pod = Pod(name, batch_id, user, job_spec, output_directory)
        self.pods[name] = pod
        await pod.run(self)

    async def create_pod(self, request):
        self.last_updated = time.time()
        parameters = await request.json()
        await asyncio.shield(self._create_pod(parameters))
        return web.Response()

    async def get_pod_log(self, request):
        pod_name = request.match_info['pod_name']
        pod = self.pods.get(pod_name)
        if not pod:
            raise web.HTTPNotFound(reason=f'unknown pod {pod_name}')
        return web.json_response(await pod.get_log())

    async def get_pod_status(self, request):
        pod_name = request.match_info['pod_name']
        pod = self.pods.get(pod_name)
        if not pod:
            raise web.HTTPNotFound(reason=f'unknown pod {pod_name}')
        return web.json_response(await pod.status())

    async def _delete_pod(self, request):
        pod_name = request.match_info['pod_name']
        if pod_name not in self.pods:
            raise web.HTTPNotFound(reason=f'unknown pod {pod_name}')
        del self.pods[pod_name]

    async def delete_pod(self, request):  # pylint: disable=unused-argument
        await asyncio.shield(self._delete_pod(request))
        return web.Response()

    async def healthcheck(self, request):  # pylint: disable=unused-argument
        return web.Response()

    async def run(self):
        app_runner = None
        site = None
        try:
            app = web.Application()
            app.add_routes([
                web.post('/api/v1alpha/pods/create', self.create_pod),
                web.get('/api/v1alpha/pods/{pod_name}/log', self.get_pod_log),
                web.get('/api/v1alpha/pods/{pod_name}/status', self.get_pod_status),
                web.post('/api/v1alpha/pods/{pod_name}/delete', self.delete_pod),
                web.get('/healthcheck', self.healthcheck)
            ])

            app_runner = web.AppRunner(app)
            await app_runner.setup()
            site = web.TCPSite(app_runner, '0.0.0.0', 5000)
            await site.start()

            await self.activate()

            last_ping = time.time() - self.last_updated
            while (self.pods and last_ping < MAX_IDLE_TIME_WITH_PODS) \
                    or last_ping < MAX_IDLE_TIME_WITHOUT_PODS:
                log.info(f'n_pods {len(self.pods)} free_cores {self.free_cores_mcpu / 1000} age {last_ping}')
                await asyncio.sleep(15)
                last_ping = time.time() - self.last_updated

            if self.pods:
                log.info(f'idle {MAX_IDLE_TIME_WITH_PODS} seconds with pods, exiting')
            else:
                log.info(f'idle {MAX_IDLE_TIME_WITHOUT_PODS} seconds with no pods, exiting')

            body = {'inst_token': self.token}
            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                await request_retry_transient_errors(
                    session, 'POST',
                    self.deploy_config.url('batch2-driver', '/api/v1alpha/instances/deactivate'),
                    json=body)
                log.info('deactivated')
        finally:
            log.info('shutting down')
            if site:
                await site.stop()
                log.info('stopped site')
            if app_runner:
                await app_runner.cleanup()
                log.info('cleaned up app runner')

    async def post_pod_complete(self, pod_status):
        body = {
            'inst_token': self.token,
            'status': pod_status
        }

        delay = 0.1
        while True:
            try:
                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                    await session.post(
                        self.deploy_config.url('batch2-driver', '/api/v1alpha/instances/pod_complete'),
                        json=body)
                    return
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                log.exception(f'failed to mark pod {pod_status["name"]} complete, retrying')
            # exponentially back off, up to (expected) max of 30s
            t = delay * random.random()
            await asyncio.sleep(t)
            delay = min(delay * 2, 60.0)

    async def activate(self):
        body = {
            'inst_token': self.token,
            'ip_address': self.ip_address
        }
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            await request_retry_transient_errors(
                session, 'POST',
                self.deploy_config.url('batch2-driver', '/api/v1alpha/instances/activate'),
                json=body)


cores = int(os.environ['CORES'])
namespace = os.environ['NAMESPACE']
inst_token = os.environ['INST_TOKEN']
ip_address = os.environ['INTERNAL_IP']
batch_worker_image = os.environ['BATCH_WORKER_IMAGE']

log.info(f'BATCH_WORKER_IMAGE={batch_worker_image}')

deploy_config = DeployConfig('gce', namespace, {})
worker = Worker(batch_worker_image, cores, deploy_config, inst_token, ip_address)

loop = asyncio.get_event_loop()
loop.run_until_complete(worker.run())
loop.run_until_complete(docker.close())
loop.run_until_complete(loop.shutdown_asyncgens())
loop.close()
log.info(f'closed')
sys.exit(0)
