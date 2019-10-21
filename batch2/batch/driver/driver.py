import os
import json
import asyncio
import aiohttp
from aiohttp import web
import logging
import google.oauth2.service_account
import sortedcontainers

from hailtop.config import get_deploy_config
from hailtop.utils import AsyncWorkerPool, request_retry_transient_errors

from ..google_compute import GServices
from ..utils import parse_cpu_in_mcpu

from .instance_pool import InstancePool

log = logging.getLogger('driver')

BATCH_JOB_DEFAULT_CPU = os.environ.get('HAIL_BATCH_JOB_DEFAULT_CPU', '1')
BATCH_JOB_DEFAULT_MEMORY = os.environ.get('HAIL_BATCH_JOB_DEFAULT_MEMORY', '3.75G')


class Pod:
    @staticmethod
    def from_record(driver, record):
        batch_id = record['batch_id']
        job_spec = json.loads(record['job_spec'])
        userdata = json.loads(record['userdata'])
        status = json.loads(record['status']) if record['status'] else None

        inst = driver.inst_pool.token_inst.get(record['instance'])

        pod = Pod(
            driver=driver,
            name=record['name'],
            batch_id=batch_id,
            job_spec=job_spec,
            userdata=userdata,
            output_directory=record['output_directory'],
            cores_mcpu=record['cores_mcpu'],
            status=status,
            instance=inst)

        if inst:
            inst.schedule(pod)

        return pod

    @staticmethod
    async def create_pod(driver, name, batch_id, job_spec, userdata, output_directory):
        cpu = None
        resources = job_spec.get('resources')
        if resources:
            cpu = resources.get('cpu')
        if not cpu:
            cpu = BATCH_JOB_DEFAULT_CPU
        cores_mcpu = parse_cpu_in_mcpu(cpu)

        await driver.db.pods.new_record(name=name, batch_id=batch_id, job_spec=json.dumps(job_spec),
                                        userdata=json.dumps(userdata), output_directory=output_directory,
                                        cores_mcpu=cores_mcpu, instance=None)

        return Pod(driver, name, batch_id, job_spec, userdata, output_directory, cores_mcpu)

    def __init__(self, driver, name, batch_id, job_spec, userdata, output_directory, cores_mcpu, instance=None, on_ready=False, status=None):
        self.driver = driver
        self.name = name
        self.batch_id = batch_id
        self.job_spec = job_spec
        self.userdata = userdata
        self.output_directory = output_directory
        self.cores_mcpu = cores_mcpu
        self.instance = instance
        self.on_ready = on_ready
        self._status = status
        self.deleted = False

        loop = asyncio.get_event_loop()
        self.lock = asyncio.Lock(loop=loop)

    async def config(self):
        job_spec = dict(self.job_spec)

        # copy secrets
        if 'secrets' in job_spec:
            job_spec['secrets'] = [dict(secret) for secret in job_spec['secrets']]

        # update resources with defaults
        if 'resources' not in job_spec:
            job_spec['resources'] = {}
        resources = job_spec['resources']
        if 'cpu' not in resources:
            resources['cpu'] = BATCH_JOB_DEFAULT_CPU
        if 'memory' not in resources:
            resources['memory'] = BATCH_JOB_DEFAULT_MEMORY

        # add user's gsa-key
        secrets = job_spec.get('secrets', [])
        secrets = list(secrets)
        secrets.append({
            'namespace': 'batch-pods',  # FIXME unused
            'name': self.userdata['gsa_key_secret_name'],
            'mount_path': '/gsa-key',
            'mount_in_copy': True
        })
        job_spec['secrets'] = secrets

        secret_futures = []
        for secret in secrets:
            # FIXME need access control to verify user is allowed to access secret
            secret_futures.append(self.driver.k8s.read_secret(secret['name']))
            k8s_secrets = await asyncio.gather(*secret_futures)

        for secret, k8s_secret in zip(secrets, k8s_secrets):
            if k8s_secret:
                secret['data'] = k8s_secret.data

        return {
            'name': self.name,
            'batch_id': self.batch_id,
            'user': self.userdata['username'],
            'job_spec': job_spec,
            'output_directory': self.output_directory
        }

    async def mark_complete(self, status):
        self._status = status
        self.driver.db.pods.update_record(self.name, status=json.dumps(status))

    def mark_deleted(self):
        assert not self.deleted
        self.deleted = True
        self.remove_from_ready()

    async def unschedule(self):
        if not self.instance:
            return

        log.info(f'unscheduling {self.name} with {self.cores_mcpu / 1000} cores from {self.instance}')

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

            log.info(f'scheduling {self.name} with {self.cores_mcpu / 1000} cores on {inst}')

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
        self.driver.ready_cores_mcpu += self.cores_mcpu
        self.driver.changed.set()

    def remove_from_ready(self):
        if self.on_ready:
            self.on_ready = False
            self.driver.ready_cores_mcpu -= self.cores_mcpu

    async def _request(self, method, url, **kwargs):
        try:
            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                resp = await request_retry_transient_errors(session, method, url, **kwargs)
                if self.instance:
                    self.instance.mark_as_healthy()
                return resp
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception:
            if self.instance:
                self.instance.mark_as_unhealthy()
            raise

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
            try:
                await self._request('POST', url, json=config)
                log.info(f'created {self.name} on inst {inst}')
            except Exception:
                log.exception(f'failed to create {self.name} on inst {inst}, putting back on ready queue')
                asyncio.ensure_future(self.put_on_ready())

    async def delete(self):
        assert self.deleted

        async with self.lock:
            inst = self.instance
            if inst:
                url = f'http://{inst.ip_address}:5000/api/v1alpha/pods/{self.name}/delete'
                try:
                    await self._request('POST', url)
                    log.info(f'deleted {self.name} from inst {inst}')
                except Exception:
                    log.exception(f'failed to delete {self.name} on inst {inst} due to exception, ignoring')

            await self.unschedule()
            await self.driver.db.pods.delete_record(self.name)

    async def read_pod_logs(self):
        log.info(f'reading pod {self.name} logs from instance {self.instance}')

        if self.instance is None:
            return None

        inst = self.instance
        url = f'http://{inst.ip_address}:5000/api/v1alpha/pods/{self.name}/logs'
        resp = self._request('GET', url)
        return await resp.json()

    async def read_pod_status(self):
        log.info(f'reading status for pod {self.name} from instance {self.instance}')

        if self.instance is None:
            return None

        inst = self.instance
        url = f'http://{inst.ip_address}:5000/api/v1alpha/pods/{self.name}/status'
        resp = self._request('GET', url)
        return await resp.json()

    def status(self):
        if self._status is None:
            # don't know yet
            return {
                'name': self.name,
                'batch_id': self.batch_id,
                'job_id': self.job_spec['job_id'],
                'user': self.userdata['username'],
                'state': 'pending'
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
        self.ready = sortedcontainers.SortedSet(key=lambda pod: pod.cores_mcpu)
        self.ready_cores_mcpu = 0
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

        pod_name = status['name']
        pod = self.pods.get(pod_name)
        if pod is None:
            log.warning(f'pod_complete from unknown pod {pod_name}, instance {inst_token}')
            return web.HTTPNotFound()
        log.info(f'pod_complete from pod {pod_name}, instance {inst_token}')
        await pod.mark_complete(status)
        await self.complete_queue.put(status)
        return web.Response()

    async def create_pod(self, name, batch_id, job_spec, userdata, output_directory):
        assert name not in self.pods
        pod = await Pod.create_pod(self, name, batch_id, job_spec, userdata, output_directory)
        self.pods[name] = pod
        await pod.put_on_ready()

    async def delete_pod(self, name):
        pod = self.pods[name]
        pod.mark_deleted()
        await self.pool.call(pod.delete)
        # this must be after delete finishes successfully in case pod marks complete before delete call
        del self.pods[name]

    async def read_pod_logs(self, name):
        return await self.pods[name].read_pod_logs()

    async def read_pod_status(self, name):
        return await self.pods[name].read_pod_status()

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
                i = self.ready.bisect_key_right(inst.free_cores_mcpu)
                if i > 0:
                    pod = self.ready[i - 1]
                    assert pod.cores_mcpu <= inst.free_cores_mcpu
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

        async def _put_on_ready():
            for pod in self.pods.values():
                if not pod.instance and not pod._status:
                    await pod.put_on_ready()

        asyncio.ensure_future(_put_on_ready())

    async def run(self):
        await self.inst_pool.start()
        await self.schedule()
