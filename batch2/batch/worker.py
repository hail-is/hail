import os
import sys
from shlex import quote as shq
import time
import logging
import asyncio
import random
import json
import aiohttp
import base64
import uuid
import shutil
from aiohttp import web
import concurrent
import aiodocker
from aiodocker.exceptions import DockerError
from hailtop.utils import request_retry_transient_errors

# import uvloop

from hailtop.config import DeployConfig
from gear import configure_logging

from .utils import parse_cpu_in_mcpu, parse_image_tag
from .semaphore import NullWeightedSemaphore, WeightedSemaphore
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
            return f(*args, **kwargs)
        except DockerError as e:
            # 408 request timeout, 503 service unavailable
            if e.status == 408 or e.status == 503:
                pass
            else:
                raise
        except asyncio.TimeoutError:
            pass
        # exponentially back off, up to (expected) max of 30s
        t = delay * random.random()
        await asyncio.sleep(t)
        delay = min(delay * 2, 60.0)


class Substep:
    def __init__(self, name):
        self.name = name

        self._container = None
        self._status = {
            'name': name,
            'state': 'pending'
        }
        self._log = None

    def _container_config(self, spec):
        image = spec['image']
        config = {
            "AttachStdin": False,
            "AttachStdout": False,
            "AttachStderr": False,
            "Tty": False,
            'OpenStdin': False,
            'Cmd': spec['command'],
            'Image': image,
            'HostConfig': {'CpuPeriod': 100000,
                           'CpuQuota': cores_mcpu * 100}
        }

        volume_mounts = spec.get('volume_mounts')
        if volume_mounts:
            config['HostConfig']['Binds'] = volume_mounts

        return config

    def run(self, gcs_client, spec, log_directory):
        try:
            self._status['state'] = 'pulling'

            try:
                await docker_call_retry(docker.images.get, image)
            except DockerError as e:
                if e.status == 404:
                    await docker_call_retry(docker.images.pull, image)

            self._status['state'] = 'creating'

            config = self._container_config(self, spec)
            self._container = await docker_call_retry(docker.containers.create, config)

            async with worker.cpu_sem(cores_mcpu):
                self._status['state'] = 'starting'

                await docker_call_retry(self._container.start)

                self._status['state'] = 'running'

                await docker_call_retry(self._container.wait)

                cstatus = await self._container_status()
                self._status['container_status'] = cstatus

                await docker_call_retry(self._container.stop)
                await docker_call_retry(self._container.delete)
                self._container = None

            self._status['state'] = 'uploading'

            complete_status = dict(self._status)
            complete_status['state'] = 'succeeded' if cstatus['State']['ExitCode'] == 0 else 'failed'

            self._log = await self._container_logs()
            log_path = LogStore.container_log_path(self.log_directory, self.name)
            upload_log = gcs_client.write_gs_file(log_path, self._log)

            status_path = LogStore.container_status_path(self.log_directory, self.name)
            upload_status = gcs_client.write_gs_file(
                status_path, json.dumps(complete_status, indent=4))

            self._status = complete_status
        except Exception as e:
            self._status['state'] = 'error'
            self._status['error'] = traceback.format_exc()
        finally:
            if self._container:
                try:
                    await docker_call_retry(self._container.stop)
                    await docker_call_retry(self._container.delete)
                except Exception:  # pylint: disable=broad-except
                    log.exception('while deleting container')
                finally:
                    self._container = None

    async def _container_logs(self):
        logs = await docker_call_retry(self._container.log, stderr=True, stdout=True)
        return "".join(logs)

    async def log(self):
        if self._log:
            return self._log

        if self._container:
            return await self._container_logs()

        return None

    async def status(self):
        if 'error' in self._status:
            return self._status

        if self._container and not 'container_status' not in self._status:
            status = dict(self._status)
            status['container_status'] = await self._container_status()
            return status

        return self._status


class Volume:
    def __init__(self, name):
        self.name = name
        self.volume = None

    async def create(self):
        config = {
            'Name': self.name
        }
        self.volume = await docker_call_retry(docker.volumes.create, config)

    @property
    def volume_path(self):
        return self.name

    async def delete(self):
        if self.volume:
            await self.volume.delete()
            self.volume = None


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


def copy_substep(name, files, volume_mounts):
    sh_expression = copy(files)
    copy_spec = {
        'image': 'google/cloud-sdk:237.0.0-alpine',
        'name': name,
        'command': ['/bin/sh', '-c', sh_expression],
        'cpu': '500m',
        'volume_mounts': volume_mounts
    }
    return Substep(copy_spec)


class Step:
    def __init__(self, spec):
        self._state = 'pending'
        self._error = None

        # create volumes
        self.volumes = []

        copy_volume_mounts = []
        main_volume_mounts = []

        if job_spec.get('mount_docker_socket'):
            main_volume_mounts.append('/var/run/docker.sock:/var/run/docker.sock')

        if pvc_size or input_files or output_files:
            v = Volume('io')
            self.volumes.append(v)
            volume_mount = 'io:/io'
            main_volume_mounts.append(volume_mount)
            copy_volume_mounts.append(volume_mount)

        secrets = job_spec.get('secrets')
        if secrets:
            for secret in job_spec['secrets']:
                host_path = f'{self.scratch}/{secret["name"]}'
                populate_secret_host_path(host_path, secret['data'])
                volume_mount = f'{host_path}:{secret["mount_path"]}'
                main_volume_mounts.append(volume_mount)
                # this will be the user gsa-key
                if secret.get('mount_in_copy', False):
                    copy_volume_mounts.append(volume_mount)

        # create substeps
        substeps = {}

        if input_files:
            substep['setup'] = copy_substep('setup', input_files, copy_volume_mounts)

        # main substep
        main_spec = {
            'command': job_spec['command'],
            'image': job_spec['image'],
            'name': 'main',
            # FIXME env
            'cpu': job_spec['resources']['cpu'],
            'volume_mounts': main_volume_mounts
        }
        substeps['main'] = Step(main_spec)

        if output_files:
            substep['cleanup'] = copy_substep('cleanup', output_files, copy_volume_mounts)

        self._substeps = substeps

    async def _mark_complete(self, worker):
        body = {
            'inst_token': self.worker.token,
            'status': self.status()
        }

        try:
            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                await request_retry_transient_errors(
                    session, 'POST',
                    self.worker.deploy_config.url('batch2-driver', '/api/v1alpha/instances/pod_complete'),
                    json=body)
        except Exception:  # pylint: disable=broad-except
            log.exception('failed to mark {self.name} complete')

    def run(self, worker):
        try:
            self._state = 'running'

            setup = substeps.get('setup')
            if setup:
                await setup.run(worker)

                setup_state = setup.state()
                if setup_state != 'succeeded':
                    self._state = setup_state
                return

            main = substeps['main']
            await main.run(worker)
            main_state = main.state()

            cleanup = substeps.get('cleanup')
            if cleanup:
                await cleanup.run(worker)

            if main_state != 'succeeded':
                self._state = main_state
            else:
                if cleanup:
                    self._state = cleanup.state()
                else:
                    self._state = 'succeeded'

            await self._mark_complete(worker)
        except Exception:  # pylint: disable=broad-except
            self._state = 'error'
            self._error = traceback.format_exc()
        finally:
            try:
                for v in self.volumes:
                    await v.delete()
                # FIXME remove scratch tree
            except Exception:  # pylint: disable=broad-except
                log.exception('while deleting volumes')

    def log(self):
        return {name: ss.log() for name, ss in self.substeps.items()}

    def status(self):
        status = {
            'state': self._state
        }
        if self._error:
            status['error'] = self._error
        status['substep_statuses'] = {
            name: ss.status() for name, ss in self.substeps.items()
        }


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
        # FIXME
        ... = parameters.get(...)

        s = Step(...)
        self.steps[s.name] = s
        await s.run()

    async def create_pod(self, request):
        self.last_updated = time.time()
        parameters = await request.json()
        await asyncio.shield(self._create_pod(parameters))
        return web.Response()

    async def get_container_log(self, request):
        pod_name = request.match_info['pod_name']
        container_name = request.match_info['container_name']

        if pod_name not in self.pods:
            raise web.HTTPNotFound(reason='unknown pod name')
        bp = self.pods[pod_name]

        if container_name not in bp.containers:
            raise web.HTTPNotFound(reason='unknown container name')
        result = await bp.log(container_name)

        return web.json_response(result)

    async def get_container_status(self, request):
        pod_name = request.match_info['pod_name']
        container_name = request.match_info['container_name']

        if pod_name not in self.pods:
            raise web.HTTPNotFound(reason='unknown pod name')
        bp = self.pods[pod_name]

        if container_name not in bp.containers:
            raise web.HTTPNotFound(reason='unknown container name')
        result = bp.container_status(container_name)

        return web.json_response(result)

    async def get_pod(self, request):
        pod_name = request.match_info['pod_name']
        if pod_name not in self.pods:
            raise web.HTTPNotFound(reason='unknown pod name')
        bp = self.pods[pod_name]
        return web.json_response(bp.to_dict())

    async def _delete_pod(self, request):
        pod_name = request.match_info['pod_name']

        if pod_name not in self.pods:
            raise web.HTTPNotFound(reason='unknown pod name')
        bp = self.pods[pod_name]
        del self.pods[pod_name]

        asyncio.ensure_future(bp.delete())

    async def delete_pod(self, request):  # pylint: disable=unused-argument
        await asyncio.shield(self._delete_pod(request))
        return web.Response()

    async def list_pods(self, request):  # pylint: disable=unused-argument
        pods = [pod.to_dict() for _, pod in self.pods.items()]
        return web.json_response(pods)

    async def healthcheck(self, request):  # pylint: disable=unused-argument
        return web.Response()

    async def run(self):
        app_runner = None
        site = None
        try:
            app = web.Application()
            app.add_routes([
                web.post('/api/v1alpha/pods/create', self.create_pod),
                # FIXME only pod log, status, not container
                web.get('/api/v1alpha/pods/{pod_name}/containers/{container_name}/log', self.get_container_log),
                web.get('/api/v1alpha/pods/{pod_name}/containers/{container_name}/status', self.get_container_status),
                web.get('/api/v1alpha/pods/{pod_name}', self.get_pod),
                web.post('/api/v1alpha/pods/{pod_name}/delete', self.delete_pod),
                web.get('/api/v1alpha/pods', self.list_pods),
                web.get('/healthcheck', self.healthcheck)
            ])

            app_runner = web.AppRunner(app)
            await app_runner.setup()
            site = web.TCPSite(app_runner, '0.0.0.0', 5000)
            await site.start()

            await self.register()

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
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
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

    async def register(self):
        body = {
            'inst_token': self.token,
            'ip_address': self.ip_address
        }
        async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
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
