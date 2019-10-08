import os
import json
import asyncio
import aiohttp
from aiohttp import web
import logging
import google.oauth2.service_account
import sortedcontainers
import traceback

from hailtop.config import get_deploy_config
from hailtop.utils import AsyncWorkerPool

from ..google_compute import GServices
from ..utils import parse_cpu
from ..globals import tasks

from .instance_pool import InstancePool

log = logging.getLogger('driver')


class DriverException(Exception):
    def __init__(self, status, message):
        super().__init__()
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

        await driver.db.pods.new_record(name=name, spec=json.dumps(spec), output_directory=output_directory,
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
                future_secrets.append(self.driver.k8s.read_secret(name))
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

    async def mark_complete(self, status):
        self._status = status
        asyncio.ensure_future(self.driver.db.pods.update_record(self.name, status=json.dumps(status)))

    def mark_deleted(self):
        assert not self.deleted
        self.deleted = True
        self.remove_from_ready()

    async def unschedule(self):
        if not self.instance:
            return

        log.info(f'unscheduling {self.name} cores {self.cores} from {self.instance}')
        self.instance.unschedule(self)
        self.instance = None
        await self.driver.db.pods.update_record(self.name, instance=None)

    async def schedule(self, inst):
        async with self.lock:
            assert not self.instance

            self.remove_from_ready()

            if self.deleted:
                log.info(f'not scheduling {self.name} on {inst.name}; pod already deleted')
                return False

            if self._status:
                log.info(f'not scheduling {self.name} on {inst.name}; pod already complete')
                return False

            if not inst.active:
                log.info(f'not scheduling {self.name} on {inst.name}; instance not active')
                asyncio.ensure_future(self.put_on_ready())
                return False

            if not inst.healthy:
                log.info(f'not scheduling {self.name} on {inst.name}; instance not healthy')
                asyncio.ensure_future(self.put_on_ready())
                return False

            log.info(f'scheduling {self.name} cores {self.cores} on {inst}')

            inst.schedule(self)

            self.instance = inst

            # FIXME: is there a way to eliminate this blocking the scheduler?
            await self.driver.db.pods.update_record(self.name, instance=inst.token)
            return True

    async def put_on_ready(self):
        # FIXME: does this need a lock?
        assert not self.on_ready

        if self._status:
            log.info(f'{self.name} already complete, ignoring put on ready')
            return

        if self.deleted:
            log.info(f'{self.name} already deleted, ignoring put on ready')
            return

        await self.unschedule()

        await self.driver.ready_queue.put(self)
        self.on_ready = True
        self.driver.ready_cores += self.cores
        self.driver.changed.set()

    def remove_from_ready(self):
        if self.on_ready:
            self.on_ready = False
            self.driver.ready_cores -= self.cores

    async def _request(self, f):
        try:
            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with f(session) as resp:
                    if self.instance:
                        self.instance.mark_as_healthy()
                    return resp, None
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception as err:  # pylint: disable=broad-except
            if self.instance:
                self.instance.mark_as_unhealthy()
            return None, err

    async def _post(self, url, json=None):
        return await self._request(lambda session: session.post(url, json=json))

    async def _get(self, url):
        return await self._request(lambda session: session.get(url))

    async def create(self):
        async with self.lock:
            assert not self.on_ready

            config = await self.config()

            if self.deleted:
                log.info(f'pod already deleted {self.name}')
                return

            if not self.instance:
                log.info(f'instance was deactivated before {self.name} could be created; rescheduling')
                asyncio.ensure_future(self.put_on_ready())
                return

            inst = self.instance
            url = f'http://{inst.ip_address}:5000/api/v1alpha/pods/create'
            resp, err = await self._post(url, config)
            if resp:
                if resp.status == 200:
                    log.info(f'created {self.name} on inst {inst}')
                    return None
                log.info(f'failed to create {self.name} on inst {inst} due to {resp}')
                return
            assert err
            log.info(f'failed to create {self.name} on inst {inst} due to {err}, putting back on ready queue')
            asyncio.ensure_future(self.put_on_ready())

    async def delete(self):
        assert self.deleted

        async with self.lock:
            inst = self.instance
            if inst:
                url = f'http://{inst.ip_address}:5000/api/v1alpha/pods/{self.name}/delete'
                resp, err = await self._post(url)
                if resp:
                    if resp.status == 200:
                        log.info(f'deleted {self.name} from inst {inst}')
                    else:
                        log.info(f'failed to delete {self.name} from inst {inst} due to {resp}')
                else:
                    assert err
                    log.info(f'failed to delete {self.name} on inst {inst} due to err {err}, ignoring')

            await self.unschedule()
            asyncio.ensure_future(self.driver.db.pods.delete_record(self.name))

    async def read_pod_log(self, container):
        assert container in tasks
        log.info(f'reading container log for {self.name}, {container} from instance {self.instance}')

        if self.instance is None:
            return None, None

        inst = self.instance
        url = f'http://{inst.ip_address}:5000/api/v1alpha/pods/{self.name}/containers/{container}/log'
        resp, err = self._get(url)
        if resp:
            return resp.json(), None
        assert err
        log.info(f'failed to read container log {self.name}, {container} on {inst} due to err {err}, ignoring')
        return None, err

    async def read_container_status(self, container):
        assert container in tasks
        log.info(f'reading container status for {self.name}, {container} from instance {self.instance}')

        if self.instance is None:
            return None, None

        inst = self.instance
        url = f'http://{inst.ip_address}:5000/api/v1alpha/pods/{self.name}/containers/{container}/status'
        resp, err = self._get(url)
        if resp:
            return resp.json(), None
        assert err
        log.info(f'failed to read container status {self.name}, {container} on {inst} due to err {err}, ignoring')
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
        return self._status

    def __str__(self):
        return self.name


class Driver:
    def __init__(self, db, k8s, batch_bucket, batch_gsa_key=None):
        self.db = db
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

        self.inst_pool = InstancePool(self)

        if batch_gsa_key is None:
            batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(batch_gsa_key)

        self.gservices = GServices(self.inst_pool.machine_name_prefix, credentials)

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
            log.warning(f'pod_complete from unknown pod {pod_name}, instance {inst_token}')
            return web.HTTPNotFound()
        log.info(f'pod_complete from pod {pod_name}, instance {inst_token}')
        await pod.mark_complete(status)
        await self.complete_queue.put(status)
        return web.Response()

    async def create_pod(self, spec, output_directory):
        name = spec['metadata']['name']

        if name in self.pods:
            return DriverException(409, f'pod {name} already exists')

        try:
            pod = await Pod.create_pod(self, name, spec, output_directory)
        except Exception as err:  # pylint: disable=broad-except
            return DriverException(400, f'unknown error creating pod: {err}')  # FIXME: what error code should this be?

        self.pods[name] = pod
        asyncio.ensure_future(pod.put_on_ready())

    async def delete_pod(self, name):
        pod = self.pods.get(name)
        if pod is None:
            return DriverException(409, f'pod {name} does not exist')
        pod.mark_deleted()
        await self.pool.call(pod.delete)
        del self.pods[name]  # this must be after delete finishes successfully in case pod marks complete before delete call

    async def read_pod_log(self, name, container):
        assert container in tasks
        pod = self.pods.get(name)
        if pod is None:
            return None, DriverException(409, f'pod {name} does not exist')
        return await pod.read_pod_log(container)

    async def read_container_status(self, name, container):
        assert container in tasks
        pod = self.pods.get(name)
        if pod is None:
            return None, DriverException(409, f'pod {name} does not exist')
        return await pod.read_container_status(container)

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
                    log.info(f'skipping pod {pod} from ready queue, already deleted')
                    pod.remove_from_ready()

            should_wait = True
            if self.inst_pool.instances_by_free_cores and self.ready:
                inst = self.inst_pool.instances_by_free_cores[-1]
                i = self.ready.bisect_key_right(inst.free_cores)
                if i > 0:
                    pod = self.ready[i - 1]
                    assert pod.cores <= inst.free_cores
                    self.ready.remove(pod)
                    should_wait = False
                    scheduled = await pod.schedule(inst)  # This cannot go in the pool!
                    if scheduled:
                        await self.pool.call(pod.create)

    async def initialize(self):
        await self.inst_pool.initialize()

        self.pool = AsyncWorkerPool(100)

        def _pod(record):
            pod = Pod.from_record(self, record)
            return pod.name, pod

        records = await self.db.pods.get_all_records()
        self.pods = dict(_pod(record) for record in records)

        for pod in self.pods.values():
            if not pod.instance and not pod._status:
                asyncio.ensure_future(pod.put_on_ready())

    async def run(self):
        await self.inst_pool.start()
        await self.schedule()
