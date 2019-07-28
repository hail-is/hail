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

import googleapiclient.discovery
import google.cloud.logging

from .backend import Backend
from .resource import InputResourceFile, TaskResourceFile
from .utils import PipelineException

HOSTNAME = socket.gethostname()

PROJECT = os.environ['PROJECT']
ZONE = os.environ['ZONE']

def new_token(n=5):
    return ''.join([secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(n)])

class UTCFormatter(logging.Formatter):
    converter = time.gmtime

def configure_logging():
    fmt = UTCFormatter(
        # NB: no space after levename because WARNING is so long
        '%(levelname)s\t| %(asctime)s | %(filename)s\t| %(funcName)s:%(lineno)d\t| '
        '%(message)s',
        # FIXME microseconds
        datefmt='%Y-%m-%dT%H:%M:%SZ')

    fh = logging.FileHandler('pipeline.log')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    root_log = logging.getLogger()

    root_log.addHandler(fh)
    root_log.addHandler(sh)

    root_log.setLevel(logging.DEBUG)

configure_logging()
log = logging.getLogger('pipeline')


class AsyncWorkerPool:
    def __init__(self, parallelism):
        self.queue = asyncio.Queue(maxsize=50)

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
        log.debug(f'initial mark {self.mark}')
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
                log.debug(f'got entry timestamp {timestamp}')
                if not self.latest:
                    log.debug('new latest')
                    self.latest = timestamp
                if timestamp < self.mark:
                    log.debug('timestamp older than mark')
                    raise StopAsyncIteration
                return entry
            except StopAsyncIteration:
                if self.latest and self.mark < self.latest:
                    log.debug(f'mark {self.latest} => {self.mark}')
                    self.mark = self.latest
                log.debug('end of list entries stream')
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
                    log.debug('getting new page...')
                    self.page = next(self.pages)
                    log.debug('got new page')
                except StopIteration:
                    log.debug('end of pages')
                    raise StopAsyncIteration
            try:
                return next(self.page)
            except StopIteration:
                log.debug('end of page')
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
            self.cores = int(t._cpu)
        else:
            self.cores = 1
        assert self.cores in (1, 2, 4, 8, 16)

    def unschedule(self):
        if not self.active_inst:
            return

        inst = self.active_inst
        inst_pool = inst.inst_pool

        inst.tasks.remove(self)

        assert not inst.pending
        if inst.active:
            inst_pool.instances_by_free_cores.remove(inst)
            inst.free_cores += self.cores
            inst_pool.instances_by_free_cores.add(inst)
            inst_pool.free_cores += self.cores
            inst_pool.changed.set()

        self.active_inst = None

    def schedule(self, inst, runner):
        assert inst.active
        assert not self.active_inst

        # all mine
        self.unschedule()
        self.remove_from_ready(runner)

        inst.tasks.add(self)

        # inst.active:
        inst.inst_pool.instances_by_free_cores.remove(inst)
        inst.free_cores -= self.cores
        inst.inst_pool.instances_by_free_cores.add(inst)
        inst.inst_pool.free_cores -= self.cores
        inst.inst_pool.changed.set()

        self.active_inst = inst

    def remove_from_ready(self, runner):
        if not self.on_ready:
            return
        self.on_ready = False
        runner.ready_cores -= self.cores

    def put_on_ready(self, runner):
        if self.on_ready:
            return
        self.unschedule()
        self.on_ready = True
        runner.ready_cores += self.cores
        runner.ready.put_nowait(self)
        log.info(f'put {self} on ready')

    def notify_children(self, runner):
        for c in self.children:
            n = c.n_pending_parents
            assert n > 0
            n -= 1
            c.n_pending_parents = n
            log.info(f'{c} now waiting on {n} parents')
            if n == 0:
                if any(p.state != 'OK' for p in self.parents):
                    c.set_state(runner, 'SKIPPED', None)
                else:
                    c.put_on_ready(runner)

    def set_state(self, runner, state, attempt_token):
        if self.state:
            return

        log.info(f'set_state {self} state {state} attempt_token {attempt_token}')

        # can potentially set state of task on ready queue
        self.unschedule()

        self.state = state
        self.attempt_token = attempt_token

        runner.n_pending_tasks -= 1
        log.info(f'n_pending_tasks {runner.n_pending_tasks}')
        if runner.n_pending_tasks == 0:
            log.info('all tasks complete, put None task')
            runner.ready.put_nowait(None)

        self.notify_children(runner)

    def __str__(self):
        return f'task {self.task.name} {self.token}'


class Instance:
    def __init__(self, inst_pool, inst_token):
        self.inst_pool = inst_pool
        self.tasks = set()
        self.token = inst_token

        # for active and pending
        self.free_cores = inst_pool.worker_cores

        # state: pending, active, deactivated (and/or deleted)
        self.pending = True
        self.inst_pool.n_pending_instances += 1
        self.inst_pool.free_cores += self.free_cores

        self.active = False
        self.deleted = False

        self.last_updated = time.time()

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

        self.active = True
        self.inst_pool.n_active_instances += 1
        self.inst_pool.instances_by_free_cores.add(self)
        self.inst_pool.changed.set()

    def deactivate(self):
        if self.pending:
            self.pending = False
            self.inst_pool.n_pending_instances -= 1
            self.inst_pool.free_cores -= self.inst_pool.worker_cores
            assert not self.active
            return

        if not self.active:
            return
        self.active = False
        self.inst_pool.n_active_instances -= 1
        self.inst_pool.instances_by_free_cores.remove(self)
        self.inst_pool.free_cores -= self.inst_pool.worker_cores
        for t in self.tasks:
            assert t.active_inst == self
            t.put_on_ready(self.inst_pool.runner)

    def update_timestamp(self):
        if self in self.inst_pool.instances:
            self.inst_pool.instances.remove(self)
            self.last_updated = time.time()
            self.inst_pool.instances.add(self)

    def remove(self):
        self.deactivate()
        self.inst_pool.instances.remove(self)
        if self.token in self.inst_pool.token_inst:
            del self.inst_pool.token_inst[self.token]

    def handle_call_delete_event(self):
        self.deactivate()
        self.deleted = True
        self.update_timestamp()

    async def delete(self):
        if self.deleted:
            return
        self.deactivate()
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
                self.remove()
                return

        status = spec['status']
        log.info(f'heal: machine {self.machine_name()} status {status}')

        # preempted goes into terminated state
        if status == 'TERMINATED' and self.deleted:
            self.remove()
            return

        if status in ('TERMINATED', 'STOPPING'):
            self.deactivate()

        if status == 'TERMINATED' and not self.deleted:
            await self.delete()

        self.update_timestamp()

    def __str__(self):
        return f'inst {self.token}'


class InstancePool:
    def __init__(self, runner, worker_cores, worker_disk_size_gb, pool_size, max_instances):
        self.runner = runner
        self.worker_cores = worker_cores
        self.worker_disk_size_gb = worker_disk_size_gb
        self.pool_size = pool_size
        self.max_instances = max_instances

        self.token = new_token()
        self.machine_name_prefix = f'pipeline-{self.token}-worker-'
        self.instances = sortedcontainers.SortedSet(key=lambda inst: inst.last_updated)

        # for active instances only
        self.instances_by_free_cores = sortedcontainers.SortedSet(key=lambda inst: inst.free_cores)

        self.changed = asyncio.Event()
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
            # FIXME resize
            'machineType': f'projects/{PROJECT}/zones/{ZONE}/machineTypes/n1-standard-{self.worker_cores}',
            'labels': {
                'role': 'pipeline_worker',
                'inst_token': inst_token
            },

            'disks': [{
                'boot': True,
                'autoDelete': True,
                # FIXME resize
                'diskSizeGb': self.worker_disk_size_gb,
                'initializeParams': {
                    'sourceImage': 'projects/broad-ctsa/global/images/cs-hack',
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
                'email': '842871226259-compute@developer.gserviceaccount.com',
                'scopes': [
                    'https://www.googleapis.com/auth/cloud-platform'
                ]
            }],

            # Metadata is readable from the instance and allows you to
            # pass configuration from deployment scripts to instances.
            'metadata': {
                'items': [{
                    'key': 'driver',
                    'value': HOSTNAME
                }, {
                    'key': 'inst_token',
                    'value': inst_token
                }, {
                    'key': 'startup-script-url',
                    'value': 'gs://hail-cseed/cs-hack/worker-startup.sh'
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
            await inst.handle_preempt_event()
        elif event_subtype == 'compute.instances.delete':
            if event_type == 'GCE_OPERATION_DONE':
                inst.remove()
            elif event_type == 'GCE_API_CALL':
                inst.handle_call_delete_event()
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
                    log.debug(f'heal: oldest {inst} age {inst_age}s')
                    if inst_age > 60:
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
                while ((self.n_pending_instances + self.n_active_instances) < self.pool_size and
                       len(self.instances) < self.max_instances and
                       self.free_cores < self.runner.ready_cores):
                    await self.create_instance()
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
        t = self.task_gtask[resource._source]
        return resource._get_path(f'{self.scratch_dir}/{t.token}/{t.attempt_token}')

    def gs_output_paths(self, resource, task_token, attempt_token):
        assert isinstance(resource, TaskResourceFile)
        output_paths = [resource._get_path(f'{self.scratch_dir}/{task_token}/{attempt_token}')]
        if resource._output_paths:
            for p in resource._output_paths:
                output_paths.append(p)
        return output_paths

    def __init__(self, pipeline, verbose, scratch_dir, worker_cores, worker_disk_size_gb, pool_size, max_instances):
        self.pipeline = pipeline
        self.verbose = verbose

        self.scratch_dir = scratch_dir

        parsed_scratch_dir = urllib.parse.urlparse(self.scratch_dir)

        self.scratch_dir_bucket_name = parsed_scratch_dir.netloc

        self.scratch_dir_path = parsed_scratch_dir.path
        while self.scratch_dir_path and self.scratch_dir_path[0] == '/':
            self.scratch_dir_path = self.scratch_dir_path[1:]

        self.inst_pool = InstancePool(self, worker_cores, worker_disk_size_gb, pool_size, max_instances)
        self.gservices = GServices(self.inst_pool.machine_name_prefix)
        self.pool = None  # created in run

        self.changed = asyncio.Event()

        self.n_pending_tasks = len(pipeline._tasks)

        self.ready = asyncio.Queue()
        self.ready_cores = 0

        self.tasks = []
        self.token_task = {}
        self.task_gtask = {}
        for pt in pipeline._tasks:
            while True:
                # 36**7 / 12000000.0 ~ 6.5K
                task_token = new_token(7)
                if task_token not in self.token_task:
                    break
            t = GTask(pt, task_token)
            self.token_task[t.token] = t
            self.tasks.append(t)
            self.task_gtask[pt] = t

        for t in self.tasks:
            for pp in t.task._dependencies:
                p = self.task_gtask[pp]
                t.parents.add(p)
                p.children.add(t)

        self.app = web.Application()
        self.app.add_routes([
            web.post('/activate_worker', self.handle_activate_worker),
            web.post('/deactivate_worker', self.handle_deactivate_worker),
            web.post('/task_complete', self.handle_task_complete)
        ])

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
        inst.deactivate()

        return web.Response()

    async def handle_task_complete(self, request):
        return await asyncio.shield(self.handle_task_complete2(request))

    async def handle_task_complete2(self, request):
        status = await request.json()

        task_token = status['task_token']
        attempt_token = status['attempt_token']
        t = self.token_task.get(task_token)
        if not t:
            log.warning('received /task_complete for unknown task {task_token}')

        log.info(f'{t} complete status {status}')

        if all(status.get(name, 0) == 0 for name in ['input', 'main', 'output']):
            state = 'OK'
        else:
            state = 'BAD'
            log.error(f'{t} failed logs in {self.scratch_dir}/{task_token}/{attempt_token}')

        t.set_state(self, state, attempt_token)
        self.changed.set()

        return web.Response()

    async def execute_task(self, t):
        try:
            # get instance
            while True:
                if self.inst_pool.instances_by_free_cores:
                    inst = self.inst_pool.instances_by_free_cores[-1]
                    if t.cores <= inst.free_cores:
                        log.info(f'scheduling {t} cores {t.cores} on {inst} free_cores {inst.free_cores}')
                        break
                await self.inst_pool.changed.wait()
                self.inst_pool.changed.clear()

            t.schedule(inst, self)

            config = self.get_task_config(t)

            async with aiohttp.ClientSession(
                    raise_for_status=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                req_body = {'task': config}
                async with session.post(f'http://{inst.machine_name()}:5000/execute_task', json=req_body) as resp:
                    await resp.json()
                    # FIXME update inst tasks
                    inst.update_timestamp()

            log.info(f'executed {t} attempt {config["attempt_token"]} on {inst}')
        except asyncio.CancelledError:  # pylint: disable=try-except-raise
            raise
        except Exception:  # pylint: disable=broad-except
            log.exception(f'failed to execute {t}, rescheduling"')
            t.put_on_ready(self)

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
            'attempt_token': attempt_token,
            'inputs_cmd': inputs_cmd,
            'image': pt._image,
            'command': cmd,
            'outputs_cmd': outputs_cmd
        }
        return config

    async def run(self):
        log.info(f'running pipeline...')

        app_runner = None
        site = None
        try:
            app_runner = web.AppRunner(self.app)
            await app_runner.setup()
            site = web.TCPSite(app_runner, '0.0.0.0', 5000)
            await site.start()

            await self.inst_pool.start()

            self.pool = AsyncWorkerPool(100)

            for t in self.tasks:
                if not t.parents:
                    t.put_on_ready(self)

            while True:
                # wait for a task
                t = await self.ready.get()
                if not t:
                    break
                if t.state:
                    continue

                log.info(f'executing {t}')

                assert not t.active_inst
                assert not t.state

                await self.pool.call(self.execute_task, t)
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
    def __init__(self, scratch_dir, worker_cores, worker_disk_size_gb, pool_size, max_instances):
        self.scratch_dir = scratch_dir
        self.worker_cores = worker_cores
        self.worker_disk_size_gb = worker_disk_size_gb
        self.pool_size = pool_size
        self.max_instances = max_instances

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        if dry_run:
            print('do stuff')
            return

        runner = GRunner(pipeline, verbose, self.scratch_dir, self.worker_cores, self.worker_disk_size_gb, self.pool_size, self.max_instances)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.run_until_complete(loop.shutdown_asyncgens())
