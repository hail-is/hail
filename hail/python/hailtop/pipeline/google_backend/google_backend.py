import collections
import os
import time
import secrets
import logging
import socket
import threading
import concurrent
import urllib.parse
import asyncio
from shlex import quote as shq
import aiohttp
from aiohttp import web
import sortedcontainers
import uvloop

import googleapiclient.discovery
import google.cloud.logging

from ..backend import Backend
from ..resource import InputResourceFile, TaskResourceFile
from ..utils import PipelineException

uvloop.install()

PROJECT = os.environ['PROJECT']
ZONE = os.environ['ZONE']


def new_token(n=5):
    return ''.join([secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(n)])


class UTCFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging():
    fmt = UTCFormatter(
        # NB: no space after levename because WARNING is so long
        '%(levelname)s\t| %(asctime)s.%(msecs)dZ | %(filename)s\t| %(funcName)s:%(lineno)d\t| '
        '%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')

    fh = logging.FileHandler('pipeline.log')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)
    root_log.addHandler(fh)
    # root_log.addHandler(sh)


configure_logging()
log = logging.getLogger('pipeline')


class ATimer:
    def __init__(self, name):
        self.name = name
        self.start = None

    async def __aenter__(self):
        self.start = time.clock()

    async def __aexit__(self, exc_type, exc, tb):
        end = time.clock()
        log.info(f'time {self.name} {exc} {end - self.start}')


class AsyncWorkerPool:
    def __init__(self, parallelism):
        self.queue = asyncio.Queue(maxsize=100)

        for _ in range(parallelism):
            asyncio.ensure_future(self._worker())

    async def _worker(self):
        while True:
            try:
                f, args, kwargs = await self.queue.get()
                await f(*args, **kwargs)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception(f'worker pool caught exception')

    async def call(self, f, *args, **kwargs):
        await self.queue.put((f, args, kwargs))


async def anext(ait):
    return await ait.__anext__()


class EntryIterator:
    def __init__(self, gservices):
        self.gservices = gservices
        self.mark = time.time()
        self.entries = None
        self.latest = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            if not self.entries:
                self.entries = await self.gservices.list_entries()
            try:
                entry = await anext(self.entries)
                timestamp = entry.timestamp.timestamp()
                if not self.latest:
                    self.latest = timestamp
                # might miss events with duplicate times
                # solution is to track resourceId
                if timestamp <= self.mark:
                    raise StopAsyncIteration
                return entry
            except StopAsyncIteration:
                if self.latest and self.mark < self.latest:
                    self.mark = self.latest
                self.entries = None
                self.latest = None


class PagedIterator:
    def __init__(self, gservices, pages):
        self.gservices = gservices
        self.pages = pages
        self.page = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            if self.page is None:
                await asyncio.sleep(5)
                try:
                    self.page = next(self.pages)
                except StopIteration:
                    raise StopAsyncIteration
            try:
                return next(self.page)
            except StopIteration:
                self.page = None


class GClients:
    def __init__(self):
        self.compute_client = googleapiclient.discovery.build('compute', 'v1')


class GServices:
    def __init__(self, machine_name_prefix):
        self.machine_name_prefix = machine_name_prefix
        self.logging_client = google.cloud.logging.Client()
        self.local_clients = threading.local()
        self.loop = asyncio.get_event_loop()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=40)

        self.filter = f'''
logName="projects/{PROJECT}/logs/compute.googleapis.com%2Factivity_log" AND
resource.type=gce_instance AND
jsonPayload.resource.name:"{self.machine_name_prefix}" AND
jsonPayload.event_subtype=("compute.instances.preempted" OR "compute.instances.delete")
'''
        log.info(f'filter {self.filter}')

    def get_clients(self):
        clients = getattr(self.local_clients, 'clients', None)
        if clients is None:
            clients = GClients()
            self.local_clients.clients = clients
        return clients

    async def run_in_pool(self, f, *args, **kwargs):
        return await self.loop.run_in_executor(self.thread_pool, lambda: f(*args, **kwargs))

    # logging
    async def list_entries(self):
        entries = self.logging_client.list_entries(filter_=self.filter, order_by=google.cloud.logging.DESCENDING)
        return PagedIterator(self, entries.pages)

    async def stream_entries(self):
        return EntryIterator(self)

    # compute
    async def get_instance(self, instance):
        def get():
            clients = self.get_clients()
            return clients.compute_client.instances().get(project=PROJECT, zone=ZONE, instance=instance).execute()  # pylint: disable=no-member

        return await self.run_in_pool(get)

    async def create_instance(self, body):
        def create():
            clients = self.get_clients()
            return clients.compute_client.instances().insert(project=PROJECT, zone=ZONE, body=body).execute()  # pylint: disable=no-member
        return await self.run_in_pool(create)

    async def delete_instance(self, instance):
        def delete():
            clients = self.get_clients()
            return clients.compute_client.instances().delete(project=PROJECT, zone=ZONE, instance=instance).execute()  # pylint: disable=no-member
        return await self.run_in_pool(delete)


class GTask:
    def __init__(self, t, task_token):
        self.task = t
        self.token = task_token
        self.parents = set()
        self.children = set()
        self.n_pending_parents = len(t._dependencies)
        self.state = None
        self.attempt_token = None

        self.on_ready = False
        self.active_inst = None

        if t._cpu:
            self.cores = t._cpu
        else:
            self.cores = 1

        self.fail_count = 0

    def unschedule(self):
        if not self.active_inst:
            return

        inst = self.active_inst
        inst_pool = inst.inst_pool

        inst.tasks.remove(self)

        assert not inst.pending and inst.active
        inst_pool.instances_by_free_cores.remove(inst)
        inst.free_cores += self.cores
        inst_pool.free_cores += self.cores
        inst_pool.instances_by_free_cores.add(inst)
        inst_pool.runner.changed.set()

        self.active_inst = None

    def schedule(self, inst, runner):
        assert inst.active
        assert not self.active_inst

        # all mine
        self.unschedule()
        self.remove_from_ready(runner)

        inst.tasks.add(self)

        # inst.active
        inst.inst_pool.instances_by_free_cores.remove(inst)
        inst.free_cores -= self.cores
        inst.inst_pool.instances_by_free_cores.add(inst)
        inst.inst_pool.free_cores -= self.cores
        # can't create more scheduling opportunities, don't set changed

        self.active_inst = inst

    def remove_from_ready(self, runner):
        if not self.on_ready:
            return
        self.on_ready = False
        runner.ready_cores -= self.cores

    async def put_on_ready(self, runner):
        if self.on_ready:
            return
        self.unschedule()
        self.on_ready = True
        runner.ready_cores += self.cores
        await runner.ready_queue.put(self)
        runner.changed.set()
        log.info(f'put {self} on ready')

    async def notify_children(self, runner):
        for c in self.children:
            n = c.n_pending_parents
            assert n > 0
            n -= 1
            c.n_pending_parents = n
            if n == 0:
                log.info(f'{c} parents finished')
                if any(p.state != 'OK' for p in c.parents):
                    await c.set_state(runner, 'SKIPPED', None)
                else:
                    await c.put_on_ready(runner)

    async def set_state(self, runner, state, attempt_token):
        if self.state:
            return

        log.info(f'set_state {self} state {state} attempt_token {attempt_token}')

        self.unschedule()
        self.remove_from_ready(runner)

        self.state = state
        self.attempt_token = attempt_token

        runner.n_pending_tasks -= 1
        log.info(f'n_pending_tasks {runner.n_pending_tasks}')
        if runner.n_pending_tasks == 0:
            log.info('all tasks complete')
            # signal scheduler that we're done
            await runner.ready_queue.put(None)
            runner.changed.set()

        print(f'{runner.n_pending_tasks} / {len(runner.tasks)} pending')

        asyncio.ensure_future(self.notify_children(runner))

    def __str__(self):
        return f'task {self.task.name} {self.token}'


class Instance:
    def __init__(self, inst_pool, inst_token):
        self.inst_pool = inst_pool
        self.tasks = set()
        self.token = inst_token

        self.free_cores = inst_pool.worker_capacity

        # state: pending, active, deactivated (and/or deleted)
        self.pending = True
        self.inst_pool.n_pending_instances += 1
        self.inst_pool.free_cores += inst_pool.worker_capacity

        self.active = False
        self.deleted = False

        self.last_updated = time.time()

        print(f'{self.inst_pool.n_pending_instances} pending {self.inst_pool.n_active_instances} active workers')

    def machine_name(self):
        return self.inst_pool.token_machine_name(self.token)

    def activate(self):
        if self.active:
            return
        if self.deleted:
            return

        if self.pending:
            self.pending = False
            self.inst_pool.n_pending_instances -= 1
            self.inst_pool.free_cores -= self.inst_pool.worker_capacity

        self.active = True
        self.inst_pool.n_active_instances += 1
        self.inst_pool.instances_by_free_cores.add(self)
        self.inst_pool.free_cores += self.inst_pool.worker_capacity
        self.inst_pool.runner.changed.set()

        print(f'{self.inst_pool.n_pending_instances} pending {self.inst_pool.n_active_instances} active workers')

    async def deactivate(self):
        if self.pending:
            self.pending = False
            self.inst_pool.n_pending_instances -= 1
            self.inst_pool.free_cores -= self.inst_pool.worker_capacity
            assert not self.active

            print(f'{self.inst_pool.n_pending_instances} pending {self.inst_pool.n_active_instances} active workers')
            return

        if not self.active:
            return

        task_list = list(self.tasks)
        for t in task_list:
            t.unschedule()

        self.active = False
        self.inst_pool.instances_by_free_cores.remove(self)
        self.inst_pool.n_active_instances -= 1
        self.inst_pool.free_cores -= self.inst_pool.worker_capacity

        for t in task_list:
            await t.put_on_ready(self.inst_pool.runner)
        assert not self.tasks

        print(f'{self.inst_pool.n_pending_instances} pending {self.inst_pool.n_active_instances} active workers')

    def update_timestamp(self):
        if self in self.inst_pool.instances:
            self.inst_pool.instances.remove(self)
            self.last_updated = time.time()
            self.inst_pool.instances.add(self)

    async def remove(self):
        await self.deactivate()
        self.inst_pool.instances.remove(self)
        if self.token in self.inst_pool.token_inst:
            del self.inst_pool.token_inst[self.token]

    async def handle_call_delete_event(self):
        await self.deactivate()
        self.deleted = True
        self.update_timestamp()

    async def delete(self):
        if self.deleted:
            return
        await self.deactivate()
        await self.inst_pool.runner.gservices.delete_instance(self.machine_name())
        log.info(f'deleted machine {self.machine_name()}')
        self.deleted = True

    async def handle_preempt_event(self):
        await self.delete()
        self.update_timestamp()

    async def heal(self):
        try:
            spec = await self.inst_pool.runner.gservices.get_instance(self.machine_name())
        except googleapiclient.errors.HttpError as e:
            if e.resp['status'] == '404':
                await self.remove()
                return

        status = spec['status']
        log.info(f'heal: machine {self.machine_name()} status {status}')

        # preempted goes into terminated state
        if status == 'TERMINATED' and self.deleted:
            await self.remove()
            return

        if status in ('TERMINATED', 'STOPPING'):
            await self.deactivate()

        if status == 'TERMINATED' and not self.deleted:
            await self.delete()

        self.update_timestamp()

    def __str__(self):
        return f'inst {self.token}'


class InstancePool:
    def __init__(self, runner, service_account, worker_type, worker_cores, worker_disk_size_gb, pool_size, max_instances):
        self.runner = runner
        self.service_account = service_account
        self.worker_type = worker_type
        self.worker_cores = worker_cores
        # 2x overschedule
        self.worker_capacity = 2 * worker_cores
        self.worker_disk_size_gb = worker_disk_size_gb
        self.pool_size = pool_size
        self.max_instances = max_instances

        self.token = new_token()
        self.machine_name_prefix = f'pipeline-{self.token}-worker-'
        self.instances = sortedcontainers.SortedSet(key=lambda inst: inst.last_updated)

        # for active instances only
        self.instances_by_free_cores = sortedcontainers.SortedSet(key=lambda inst: inst.free_cores)

        self.n_pending_instances = 0
        self.n_active_instances = 0

        # for pending and active
        self.free_cores = 0

        self.token_inst = {}

    def token_machine_name(self, inst_token):
        return f'{self.machine_name_prefix}{inst_token}'

    async def start(self):
        log.info('starting instance pool')
        asyncio.ensure_future(self.control_loop())
        asyncio.ensure_future(self.event_loop())
        asyncio.ensure_future(self.heal_loop())
        log.info('instance pool started')

    async def create_instance(self):
        while True:
            inst_token = new_token()
            if inst_token not in self.token_inst:
                break
        # reserve
        self.token_inst[inst_token] = None

        log.info(f'creating instance {inst_token}')

        machine_name = self.token_machine_name(inst_token)
        config = {
            'name': machine_name,
            'machineType': f'projects/{PROJECT}/zones/{ZONE}/machineTypes/n1-{self.worker_type}-{self.worker_cores}',
            'labels': {
                'role': 'pipeline_worker',
                'inst_token': inst_token
            },

            'disks': [{
                'boot': True,
                'autoDelete': True,
                'diskSizeGb': self.worker_disk_size_gb,
                'initializeParams': {
                    'sourceImage': 'projects/broad-ctsa/global/images/pipeline',
                }
            }],

            'networkInterfaces': [{
                'network': 'global/networks/default',
                'networkTier': 'PREMIUM',
                'accessConfigs': [{
                    'type': 'ONE_TO_ONE_NAT',
                    'name': 'external-nat'
                }]
            }],

            'scheduling': {
                'automaticRestart': False,
                'onHostMaintenance': "TERMINATE",
                'preemptible': True
            },

            'serviceAccounts': [{
                'email': self.service_account,
                'scopes': [
                    'https://www.googleapis.com/auth/cloud-platform'
                ]
            }],

            # Metadata is readable from the instance and allows you to
            # pass configuration from deployment scripts to instances.
            'metadata': {
                'items': [{
                    'key': 'driver_base_url',
                    'value': self.runner.base_url
                }, {
                    'key': 'inst_token',
                    'value': inst_token
                }, {
                    'key': 'scratch',
                    'value': self.runner.scratch_dir
                }, {
                    'key': 'startup-script-url',
                    'value': 'gs://hail-common/dev2/pipeline/worker-startup.sh'
                }]
            }
        }

        await self.runner.gservices.create_instance(config)
        log.info(f'created machine {machine_name}')

        inst = Instance(self, inst_token)
        self.token_inst[inst_token] = inst
        self.instances.add(inst)

        log.info(f'created {inst}')

        return inst

    async def handle_event(self, event):
        if not event.payload:
            log.warning(f'event has no payload')
            return

        payload = event.payload
        version = payload['version']
        if version != '1.2':
            log.warning('unknown event verison {version}')
            return

        resource_type = event.resource.type
        if resource_type != 'gce_instance':
            log.warning(f'unknown event resource type {resource_type}')
            return

        event_type = payload['event_type']
        event_subtype = payload['event_subtype']
        resource = payload['resource']
        name = resource['name']

        log.info(f'event {version} {resource_type} {event_type} {event_subtype} {name}')

        if not name.startswith(self.machine_name_prefix):
            log.warning(f'event for unknown machine {name}')
            return

        inst_token = name[len(self.machine_name_prefix):]
        inst = self.token_inst.get(inst_token)
        if not inst:
            log.warning(f'event for unknown instance {inst_token}')
            return

        if event_subtype == 'compute.instances.preempted':
            log.info(f'event handler: handle preempt {inst}')
            await inst.handle_preempt_event()
        elif event_subtype == 'compute.instances.delete':
            if event_type == 'GCE_OPERATION_DONE':
                log.info(f'event handler: remove {inst}')
                await inst.remove()
            elif event_type == 'GCE_API_CALL':
                log.info(f'event handler: handle call delete {inst}')
                await inst.handle_call_delete_event()
            else:
                log.warning(f'unknown event type {event_type}')
        else:
            log.warning(f'unknown event subtype {event_subtype}')

    async def event_loop(self):
        while True:
            try:
                async for event in await self.runner.gservices.stream_entries():
                    await self.handle_event(event)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception('event loop failed due to exception')
            await asyncio.sleep(15)

    async def heal_loop(self):
        while True:
            try:
                if self.instances:
                    # 0 is the smallest (oldest)
                    inst = self.instances[0]
                    inst_age = time.time() - inst.last_updated
                    if inst_age > 60:
                        log.info(f'heal: oldest {inst} age {inst_age}s')
                        await inst.heal()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception('instance pool heal loop: caught exception')

            await asyncio.sleep(1)

    async def control_loop(self):
        while True:
            try:
                log.info(f'n_pending_instances {self.n_pending_instances}'
                         f' n_active_instances {self.n_active_instances}'
                         f' pool_size {self.pool_size}'
                         f' n_instances {len(self.instances)}'
                         f' max_instances {self.max_instances}'
                         f' free_cores {self.free_cores}'
                         f' ready_cores {self.runner.ready_cores}')

                if self.runner.ready_cores > 0:
                    instances_needed = (self.runner.ready_cores - self.free_cores + self.worker_capacity - 1) // self.worker_capacity
                    instances_needed = min(instances_needed,
                                           self.pool_size - (self.n_pending_instances + self.n_active_instances),
                                           self.max_instances - len(self.instances),
                                           # 20 queries/s; our GCE long-run quota
                                           300)
                    if instances_needed > 0:
                        log.info(f'creating {instances_needed} new instances')
                        # parallelism will be bounded by thread pool
                        await asyncio.gather(*[self.create_instance() for _ in range(instances_needed)])
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception('instance pool control loop: caught exception')

            await asyncio.sleep(15)


class GRunner:
    def gs_input_path(self, resource):
        if isinstance(resource, InputResourceFile):
            return resource._input_path

        assert isinstance(resource, TaskResourceFile)
        t = self.ptask_task[resource._source]
        return resource._get_path(f'{self.scratch_dir}/{t.token}/{t.attempt_token}')

    def gs_output_paths(self, resource, task_token, attempt_token):
        assert isinstance(resource, TaskResourceFile)
        output_paths = [resource._get_path(f'{self.scratch_dir}/{task_token}/{attempt_token}')]
        if resource._output_paths:
            for p in resource._output_paths:
                output_paths.append(p)
        return output_paths

    def __init__(self, pipeline, verbose, service_account, scratch_dir, worker_type, worker_cores, worker_disk_size_gb, pool_size, max_instances, port):
        self.pipeline = pipeline
        self.verbose = verbose

        self.scratch_dir = scratch_dir

        parsed_scratch_dir = urllib.parse.urlparse(self.scratch_dir)

        self.scratch_dir_bucket_name = parsed_scratch_dir.netloc

        self.scratch_dir_path = parsed_scratch_dir.path
        while self.scratch_dir_path and self.scratch_dir_path[0] == '/':
            self.scratch_dir_path = self.scratch_dir_path[1:]

        self.port = port

        hostname = socket.gethostname()
        self.base_url = f'http://{hostname}:{port}'

        if worker_type == 'standard':
            m = 3.75
        elif worker_type == 'highmem':
            m = 6.5
        else:
            assert worker_type == 'highcpu', worker_type
            m = 0.9
        self.worker_mem_per_core_in_gb = 0.9 * m

        self.inst_pool = InstancePool(self, service_account, worker_type, worker_cores, worker_disk_size_gb, pool_size, max_instances)
        self.gservices = GServices(self.inst_pool.machine_name_prefix)
        self.pool = None  # created in run

        self.loop = asyncio.get_event_loop()

        self.n_pending_tasks = len(pipeline._tasks)

        self.changed = asyncio.Event()
        self.ready_queue = asyncio.Queue(maxsize=1000)
        self.ready = sortedcontainers.SortedSet(key=lambda inst: inst.cores)
        self.ready_cores = 0

        self.tasks = []
        self.token_task = {}
        self.ptask_task = {}
        for pt in pipeline._tasks:
            while True:
                # 36**7 / 12000000.0 ~ 6.5K
                task_token = new_token(7)
                if task_token not in self.token_task:
                    break
            t = GTask(pt, task_token)
            self.token_task[t.token] = t
            self.tasks.append(t)
            self.ptask_task[pt] = t

        for t in self.tasks:
            for pp in t.task._dependencies:
                p = self.ptask_task[pp]
                t.parents.add(p)
                p.children.add(t)

        self.app = web.Application()
        self.app.add_routes([
            web.post('/activate_worker', self.handle_activate_worker),
            web.post('/deactivate_worker', self.handle_deactivate_worker),
            web.post('/task_complete', self.handle_task_complete),
            web.post('/pool/size', self.handle_pool_size)
        ])

    async def handle_pool_size(self, request):
        return await asyncio.shield(self.handle_pool_size2(request))

    async def handle_pool_size2(self, request):
        body = await request.json()
        pool_size = body['pool_size']
        old_pool_size = self.inst_pool.pool_size
        self.inst_pool.pool_size = pool_size
        print(f'pool_size changed: {old_pool_size} => {pool_size}')
        return web.Response()

    async def handle_activate_worker(self, request):
        return await asyncio.shield(self.handle_activate_worker2(request))

    async def handle_activate_worker2(self, request):
        body = await request.json()
        inst_token = body['inst_token']

        inst = self.inst_pool.token_inst.get(inst_token)
        if not inst:
            log.warning(f'/activate_worker from unknown inst {inst_token}')
            raise web.HTTPNotFound()

        log.info(f'activating {inst}')
        inst.activate()

        return web.Response()

    async def handle_deactivate_worker(self, request):
        return await asyncio.shield(self.handle_deactivate_worker2(request))

    async def handle_deactivate_worker2(self, request):
        body = await request.json()
        inst_token = body['inst_token']

        inst = self.inst_pool.token_inst.get(inst_token)
        if not inst:
            log.warning(f'/deactivate_worker from unknown inst {inst_token}')
            raise web.HTTPNotFound()

        log.info(f'deactivating {inst}')
        await inst.deactivate()

        return web.Response()

    async def handle_task_complete(self, request):
        return await asyncio.shield(self.handle_task_complete2(request))

    async def handle_task_complete2(self, request):
        status = await request.json()

        inst_token = status['inst_token']
        task_token = status['task_token']
        attempt_token = status['attempt_token']
        t = self.token_task.get(task_token)
        if not t:
            log.warning(f'received /task_complete for unknown task {task_token}')
            raise web.HTTPNotFound()

        log.info(f'{t} complete status {status}')

        if all(status.get(name, 0) == 0 for name in ['input', 'main', 'output']):
            state = 'OK'
        else:
            state = 'BAD'
            log.error(f'{t} failed logs in {self.scratch_dir}/{task_token}/{attempt_token}')

        await t.set_state(self, state, attempt_token)

        inst = self.inst_pool.token_inst.get(inst_token)
        if not inst:
            inst.fail_count = 0

        return web.Response()

    async def execute_task(self, t, inst):
        try:
            config = self.get_task_config(t)
            req_body = {'task': config}

            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.post(f'http://{inst.machine_name()}:5000/execute_task', json=req_body):
                    log.info(f'submitted {t} attempt {config["attempt_token"]} on {inst}')
                    inst.fail_count = 0
                    inst.update_timestamp()
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception:  # pylint: disable=broad-except
            log.exception(f'failed to execute {t} on {inst} {inst.fail_count}, rescheduling"')
            await t.put_on_ready(self)
            inst.fail_count += 1
            if inst.fail_count >= 3:
                log.info(f'deleting failing instance {inst} fail_count {inst.fail_count}')
                await inst.delete()

    def get_task_config(self, t):
        attempt_token = new_token()

        pt = t.task

        if pt._inputs:
            inputs_cmd = 'set -ex; ' + (' && '.join([
                f'gsutil -m cp -r {shq(self.gs_input_path(i))} {shq(i._get_path("/shared"))}'
                for i in pt._inputs
            ]))
        else:
            inputs_cmd = None

        bash_flags = 'set -e' + ('x' if self.verbose else '') + '; '
        defs = ''.join([r._declare('/shared') + '; ' for r in pt._mentioned])
        make_local_tmpdir = f'mkdir -p /shared/{pt._uid}'
        cmd = bash_flags + defs + make_local_tmpdir + ' && (' + ' && '.join(pt._command) + ')'

        if pt._outupts:
            outputs = pt._internal_outputs.union(pt._external_outputs)
            outputs_cmd = 'set -ex; ' + (' && '.join([
                f'gsutil -m cp -r {shq(o._get_path("/shared"))} {shq(output_path)}'
                for o in outputs for output_path in self.gs_output_paths(o, t.token, attempt_token)
            ]))
        else:
            outputs_cmd = None

        assert pt._image
        config = {
            'scratch_dir': self.scratch_dir,
            'task_name': pt.name,
            'task_token': t.token,
            'cores': t.cores,
            'mem_in_gb': t.cores * self.worker_mem_per_core_in_gb,
            'attempt_token': attempt_token,
            'inputs_cmd': inputs_cmd,
            'image': pt._image,
            'command': cmd,
            'outputs_cmd': outputs_cmd
        }
        return config

    async def schedule(self):
        log.info('scheduler started')

        self.changed.clear()
        should_wait = False
        while self.n_pending_tasks > 0:
            if should_wait:
                await self.changed.wait()
                self.changed.clear()

            if not self.ready:
                t = await self.ready_queue.get()
                if not t:
                    return
                self.ready.add(t)
            while len(self.ready) < 50 and not self.ready_queue.empty():
                t = self.ready_queue.get_nowait()
                if not t:
                    return
                self.ready.add(t)

            should_wait = True
            if self.inst_pool.instances_by_free_cores and self.ready:
                inst = self.inst_pool.instances_by_free_cores[-1]
                i = self.ready.bisect_key_right(inst.free_cores)
                if i > 0:
                    t = self.ready[i - 1]
                    assert t.cores <= inst.free_cores
                    self.ready.remove(t)
                    should_wait = False
                    if not t.state:
                        assert not t.active_inst

                        log.info(f'scheduling {t} cores {t.cores} on {inst} free_cores {inst.free_cores} ready {len(self.ready)} qsize {self.pool.queue.qsize()}')
                        t.schedule(inst, self)
                        asyncio.ensure_future(self.pool.call(self.execute_task, t, inst))

    async def enqueue_roots(self):
        for t in self.tasks:
            if not t.parents:
                await t.put_on_ready(self)

    async def run(self):
        log.info(f'running pipeline...')
        print('running pipeline...')

        app_runner = None
        site = None
        try:
            app_runner = web.AppRunner(self.app)
            await app_runner.setup()
            site = web.TCPSite(app_runner, '0.0.0.0', self.port)
            await site.start()

            await self.inst_pool.start()

            self.pool = AsyncWorkerPool(100)

            asyncio.ensure_future(self.enqueue_roots())

            await self.schedule()
        finally:
            if site:
                await site.stop()
            if app_runner:
                await app_runner.cleanup()

        c = collections.Counter([t.state for t in self.tasks])

        n_ok = c.get('OK', 0)
        n_bad = c.get('BAD', 0)
        n_skipped = c.get('SKIPPED', 0)

        log.info(f'pipeline finished: OK {n_ok} BAD {n_bad} SKIPPED {n_skipped}')

        if n_bad > 0:
            raise PipelineException('pipeline failed')

        print('INFO: pipeline succeeded!')


class GoogleBackend(Backend):
    def __init__(self, service_account, scratch_dir, worker_cores, worker_disk_size_gb, pool_size, max_instances, *, worker_type='standard', port=5000):
        assert worker_type in ('standard', 'highcpu', 'highmem'), worker_type
        self.service_account = service_account
        self.scratch_dir = scratch_dir
        self.worker_type = worker_type
        self.worker_cores = worker_cores
        self.worker_disk_size_gb = worker_disk_size_gb
        self.pool_size = pool_size
        self.max_instances = max_instances
        self.port = port

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        if dry_run:
            print('do stuff')
            return

        runner = GRunner(pipeline, verbose, self.service_account, self.scratch_dir, self.worker_type, self.worker_cores, self.worker_disk_size_gb, self.pool_size, self.max_instances, self.port)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.run_until_complete(loop.shutdown_asyncgens())
