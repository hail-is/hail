import os
import json
import asyncio
import aiohttp
import logging
import google.oauth2.service_account
import sortedcontainers
import traceback
from aiohttp import web

from hailtop.config import get_deploy_config

from .google_compute import GServices
from .instance_pool import InstancePool
from .utils import AsyncWorkerPool, parse_cpu
from .globals import get_db, tasks
from .batch_configuration import BATCH_NAMESPACE


log = logging.getLogger('driver')

db = get_db()


class DriverException(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message

    def __str__(self):
        return f'{(self.status, self.message)}'


class Pod:
    @staticmethod
    def from_record(driver, record):
        spec = json.loads(record['spec'])
        status = json.loads(record['status']) if record['status'] else None

        inst = driver.inst_pool.token_inst.get(record['instance'])

        pod = Pod(
            driver=driver,
            name=record['name'],
            spec=spec,
            output_directory=record['output_directory'],
            cores=record['cores'],
            status=status,
            instance=inst
        )

        if inst:
            inst.schedule(pod)

        return pod

    @staticmethod
    async def create_pod(driver, name, spec, output_directory):
        container_cpu_requests = [container['resources']['requests']['cpu'] for container in spec['spec']['containers']]
        container_cores = [parse_cpu(cpu) for cpu in container_cpu_requests]
        if any([cores is None for cores in container_cores]):
            raise Exception(f'invalid value(s) for cpu: '
                            f'{[cpu for cpu, cores in zip(container_cpu_requests, container_cores) if cores is None]}')
        cores = max(container_cores)

        await db.pods.new_record(name=name, spec=json.dumps(spec), output_directory=output_directory,
                                 cores=cores, instance=None)

        return Pod(driver, name, spec, output_directory, cores)

    def __init__(self, driver, name, spec, output_directory, cores, instance=None, on_ready=False, status=None):
        self.driver = driver
        self.name = name
        self.spec = spec
        self.output_directory = output_directory
        self.cores = cores
        self.instance = instance
        self.on_ready = on_ready
        self._status = status
        self.deleted = False

        loop = asyncio.get_event_loop()
        self.lock = asyncio.Lock(loop=loop)

    async def config(self):
        future_secrets = []
        secret_names = []

        for volume in self.spec['spec']['volumes']:
            if volume['secret'] is not None:
                name = volume['secret']['secret_name']
                secret_names.append(name)
                future_secrets.append(self.driver.k8s.read_secret(name, namespace=BATCH_NAMESPACE))
        results = await asyncio.gather(*future_secrets)

        secrets = {}
        for name, (secret, err) in zip(secret_names, results):
            if err is not None:
                traceback.print_tb(err.__traceback__)
                log.info(f'could not get secret {name} due to {err}')
            secrets[name] = secret.data if not err else None

        return {
            'spec': self.spec,
            'secrets': secrets,
            'output_directory': self.output_directory
        }

    async def unschedule(self):
        if not self.instance:
            return

        log.info(f'unscheduling {self.name} cores {self.cores} from {self.instance}')
        self.instance.unschedule(self)
        self.instance = None
        await db.pods.update_record(self.name, instance=None)

    async def schedule(self, inst):
        log.info(f'scheduling {self.name} cores {self.cores} on {inst}')

        assert inst.active and not self.instance
        assert self.on_ready and not self._status and not self.deleted

        self.on_ready = False
        self.driver.ready_cores -= self.cores

        inst.schedule(self)

        self.instance = inst

        await db.pods.update_record(self.name, instance=inst.token)

    async def _put_on_ready(self):
        if self._status:
            log.info(f'{self.name} already complete, ignoring')
            return

        await self.unschedule()

        await self.driver.ready_queue.put(self)
        self.on_ready = True
        self.driver.ready_cores += self.cores
        self.driver.changed.set()

    async def put_on_ready(self):
        async with self.lock:
            await self._put_on_ready()

    async def create(self, inst):
        async with self.lock:
            if self.deleted:
                log.info(f'pod already deleted {self.name}')
                return

            log.info(f'creating {self.name} on instance {inst}')

            try:
                config = await self.config()  # FIXME: handle missing secrets!

                async with aiohttp.ClientSession(
                        raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.post(f'http://{inst.ip_address}:5000/api/v1alpha/pods/create', json=config) as resp:
                        inst.mark_as_healthy()
                        if resp.status == 200:
                            log.info(f'successfully created {self.name} on inst {inst.name}')
                            return None
                        else:
                            log.info(f'failed to create {self.name} on inst {inst.name} due to {resp}')
                            return
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as err:  # pylint: disable=broad-except
                log.info(f'failed to execute {self.name} on {inst} due to err {err}, rescheduling')
                inst.mark_as_unhealthy()
                await self._put_on_ready()  # IS this even necessary -- unschedule put back on queue? But if it goes back to running, never unscheduled. So, this is correct.

    async def delete(self):
        async with self.lock:
            log.info(f'deleting {self.name} from instance {self.instance}')
            self.deleted = True

            if self.on_ready:
                self.driver.ready_cores -= self.cores

            if self.instance:
                try:
                    async with aiohttp.ClientSession(
                            raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                        async with session.post(f'http://{self.instance.ip_address}:5000/api/v1alpha/pods/{self.name}/delete') as resp:
                            self.instance.mark_as_healthy()
                            if resp.status == 200:
                                log.info(f'successfully deleted {self.name}')
                            else:
                                log.info(f'failed to delete due to {resp}')
                                return
                except asyncio.CancelledError:  # pylint: disable=try-except-raise
                    raise
                except Exception as err:  # pylint: disable=broad-except
                    log.info(f'failed to delete {self.name} on {self.instance} due to err {err}, ignoring')
                    self.instance.mark_as_unhealthy()

            await self.unschedule()
            await db.pods.delete_record(self.name)

    async def read_pod_log(self, container):
        assert container in tasks
        log.info(f'reading pod log for {self.name}, {container} from instance {self.instance}')

        if self.instance is None:
            return None, None

        try:
            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f'http://{self.instance.ip_address}:5000/api/v1alpha/pods/{self.name}/containers/{container}/log') as resp:
                    self.instance.mark_as_healthy()
                    return resp.json(), None
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as err:  # pylint: disable=broad-except
            log.info(f'failed to read pod log {self.name}, {container} on {self.instance} due to err {err}, ignoring')
            self.instance.mark_as_unhealthy()
            return None, err

    async def read_container_status(self, container):
        assert container in tasks
        log.info(f'reading container status for {self.name}, {container} from instance {self.instance}')

        if self.instance is None:
            return None

        try:
            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f'http://{self.instance.ip_address}:5000/api/v1alpha/pods/{self.name}/containers/{container}/status') as resp:
                    self.instance.mark_as_healthy()
                    return resp.json(), None
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as err:  # pylint: disable=broad-except
            log.info(f'failed to read container status {self.name}, {container} on {self.instance} due to err {err}, ignoring')
            self.instance.mark_as_unhealthy()
            return None, err

    def status(self):
        if self._status is None:
            return {
                'metadata': self.spec['metadata'],
                'status': {
                    'containerStatuses': None,
                    'phase': 'Pending'
                }
            }
        else:
            return self._status

    def __str__(self):
        return self.name


class Driver:
    def __init__(self, k8s, batch_bucket, batch_gsa_key=None, worker_type='standard', worker_cores=1,
                 worker_disk_size_gb=10, pool_size=1, max_instances=2):
        self.k8s = k8s
        self.batch_bucket = batch_bucket
        self.pods = None  # populated in run
        self.complete_queue = asyncio.Queue()
        self.ready_queue = asyncio.Queue(maxsize=1000)
        self.ready = sortedcontainers.SortedSet(key=lambda pod: pod.cores)
        self.ready_cores = 0
        self.changed = asyncio.Event()

        self.pool = None  # created in run

        deploy_config = get_deploy_config()

        self.base_url = deploy_config.base_url('batch2')

        self.inst_pool = InstancePool(self, worker_type, worker_cores,
                                      worker_disk_size_gb, pool_size, max_instances)

        if batch_gsa_key is None:
            batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(batch_gsa_key)

        self.gservices = GServices(self.inst_pool.machine_name_prefix, credentials)

    async def get_secret(self, name):
        secret = self.k8s.read_secret(name)
        return secret.data

    async def activate_worker(self, request):
        body = await request.json()
        inst_token = body['inst_token']
        ip_address = body['ip_address']

        inst = self.inst_pool.token_inst.get(inst_token)
        if not inst:
            log.warning(f'/activate_worker from unknown inst {inst_token}')
            raise web.HTTPNotFound()

        log.info(f'activating instance {inst}')
        await inst.activate(ip_address)
        inst.mark_as_healthy()
        return web.Response()

    async def deactivate_worker(self, request):
        body = await request.json()
        inst_token = body['inst_token']

        inst = self.inst_pool.token_inst.get(inst_token)
        if not inst:
            log.warning(f'/deactivate_worker from unknown instance {inst_token}')
            raise web.HTTPNotFound()

        log.info(f'received /deactivate_worker from instance {inst}')
        await inst.deactivate()
        inst.mark_as_healthy()
        return web.Response()

    async def pod_complete(self, request):
        body = await request.json()
        inst_token = body['inst_token']
        status = body['status']

        inst = self.inst_pool.token_inst.get(inst_token)
        if not inst:
            log.warning(f'pod_complete from unknown instance {inst_token}')
            raise web.HTTPNotFound()
        inst.mark_as_healthy()

        pod_name = status['metadata']['name']
        pod = self.pods.get(pod_name)
        if pod is None:
            log.warning(f'pod_complete from unknown pod {pod_name}')
            return web.HTTPNotFound()
        else:
            log.info(f'pod_complete from pod {pod_name}')

        pod._status = status

        await db.pods.update_record(pod_name, status=json.dumps(status))
        await self.complete_queue.put(status)
        return web.Response()

    async def create_pod(self, spec, output_directory):
        name = spec['metadata']['name']

        if name in self.pods:
            return DriverException(409, f'pod {name} already exists')

        try:
            pod = await Pod.create_pod(self, name, spec, output_directory)
        except Exception as err:
            return DriverException(400, f'invalid pod spec given: {err}')  # FIXME: what error code should this be?

        self.pods[name] = pod
        asyncio.ensure_future(pod.put_on_ready())

    async def delete_pod(self, name):
        pod = self.pods.get(name)
        if pod is None:
            return DriverException(409, f'pod {name} does not exist')
        await self.pool.call(pod.delete)
        del self.pods[name]

    async def read_pod_log(self, name, container):
        assert container in tasks
        pod = self.pods.get(name)
        if pod is None:
            return None, DriverException(409, f'pod {name} does not exist')
        return await pod.read_pod_log(container), None

    async def read_container_status(self, name, container):
        assert container in tasks
        pod = self.pods.get(name)
        if pod is None:
            return None, DriverException(409, f'pod {name} does not exist')
        return await pod.read_container_status(container), None

    def list_pods(self):
        return [pod.status() for _, pod in self.pods.items()]

    async def schedule(self):
        log.info('scheduler started')

        self.changed.clear()
        should_wait = False
        while True:
            if should_wait:
                await self.changed.wait()
                self.changed.clear()

            while len(self.ready) < 50 and not self.ready_queue.empty():
                pod = self.ready_queue.get_nowait()
                if not pod.deleted:
                    self.ready.add(pod)
                else:
                    pod.on_ready = False
                    log.info(f'skipping pod {pod.name} from ready; already deleted')

            should_wait = True
            if self.inst_pool.instances_by_free_cores and self.ready:
                inst = self.inst_pool.instances_by_free_cores[-1]
                i = self.ready.bisect_key_right(inst.free_cores)
                if i > 0:
                    pod = self.ready[i - 1]
                    assert pod.cores <= inst.free_cores
                    self.ready.remove(pod)
                    should_wait = False
                    if not pod.deleted:
                        await pod.schedule(inst)
                        await self.pool.call(pod.create, inst)
                    else:
                        log.info(f'not scheduling pod {pod.name}; already deleted')

    async def initialize(self):
        await self.inst_pool.initialize()

        self.pool = AsyncWorkerPool(100)

        def _pod(record):
            pod = Pod.from_record(self, record)
            return pod.name, pod

        records = await db.pods.get_all_records()
        self.pods = dict([_pod(record) for record in records])

        for pod in list(self.pods.values()):
            if not pod.instance and not pod._status:
                asyncio.ensure_future(pod.put_on_ready())

    async def run(self):
        await self.inst_pool.start()
        await self.schedule()
