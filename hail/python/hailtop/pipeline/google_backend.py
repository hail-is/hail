import collections
import os
import re
import time
import random
import secrets
import logging
import json
import urllib.parse
import asyncio
import concurrent
from shlex import quote as shq
from aiohttp import web
import sortedcontainers

import googleapiclient.discovery
import google.cloud.storage
import google.cloud.logging

from .backend import Backend
from .resource import InputResourceFile, TaskResourceFile
from .utils import PipelineException

PROJECT = os.environ['PROJECT']
ZONE = os.environ['ZONE']

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('pipeline')

async def anext(ait):
    return await ait.__anext__()


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
                # this can potentially drop entries with identical timestamp
                # fix is to track insertIds
                if timestamp <= self.mark:
                    raise StopIteration
                return entry
            except StopIteration:
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
                self.page = await self.gservices.run_in_pool(next, self.pages)
            try:
                return next(self.page)
            except StopIteration:
                self.page = None


class GServices:
    def __init__(self):
        self.logging_client = google.cloud.logging.Client()
        self.storage_client = google.cloud.storage.Client()
        self.compute_client = googleapiclient.discovery.build('compute', 'v1')
        self.loop = asyncio.get_event_loop()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)

    async def run_in_pool(self, f, *args, **kwargs):
        return await self.loop.run_in_executor(self.thread_pool, lambda: f(*args, **kwargs))

    # storage
    async def get_bucket(self, name):
        return await self.run_in_pool(self.storage_client.get_bucket, name)

    async def upload_from_string(self, bucket, path, data):
        blob = await self.run_in_pool(bucket.blob, path)
        await self.run_in_pool(blob.upload_from_string, data)

    # logging
    async def list_entries(self):
        filter = f'logName:projects/{PROJECT}/logs/compute.googleapis.com%2Factivity_log'
        await asyncio.sleep(5)
        entries = await self.run_in_pool(
            self.logging_client.list_entries, filter_=filter, order_by=google.cloud.logging.DESCENDING)
        return PagedIterator(self, entries.pages)

    async def stream_entries(self):
        return EntryIterator(self)

    # compute
    async def get_instance(self, instance):
        return await self.run_in_pool(
            self.compute_client.instances().get(project=PROJECT, zone=ZONE, instance=instance).execute)  # pylint: disable=no-member

    async def create_instance(self, body):
        return await self.run_in_pool(
            self.compute_client.instances().insert(project=PROJECT, zone=ZONE, body=body).execute)  # pylint: disable=no-member

    async def delete_instance(self, instance):
        return await self.run_in_pool(
            self.compute_client.instances().delete(project=PROJECT, zone=ZONE, instance=instance).execute)  # pylint: disable=no-member


def round_up(x):
    i = int(x)
    if x > i:
        i += 1
    return i

MIN_STORAGE_GB = 20

def storage_size_in_gb(storage):
    pattern = '(?P<value>[0-9\\.]+)(?P<unit>[KMGTP]i?B?)?'
    m = re.match(pattern, storage)

    if not m:
        raise ValueError(f'could not convert to size: {storage}')
    value = float(m['value'])

    unit = m['unit']
    if unit:
        end = -1
        if unit[end] == 'B':
            end -= 1
        if unit[end] == 'i':
            base = 1024
        else:
            base = 1000
        exponents = {'K': 1, 'M': 2, 'G': 3, 'T': 4}
        e = exponents[unit[0]]
        multiplier = base**e
    else:
        multiplier = 1

    storage_gb = round_up((value * multiplier) / (1000 ** 3))
    if storage_gb < MIN_STORAGE_GB:
        storage_gb = MIN_STORAGE_GB

    return storage_gb

class GTask:
    def __init__(self, t):
        self.task = t
        self.token = secrets.token_hex(16)
        self.parents = set()
        self.children = set()
        self.n_pending_parents = len(t._dependencies)
        self.state = None
        self.complete_inst_token = None
        self.active_inst = None

        if t._cpu:
            self.cores = int(t._cpu)
        else:
            self.cores = 1
        assert self.cores in (1, 2, 4, 8, 16)

        if t._storage:
            self.storage_gb = storage_size_in_gb(t._storage)
        else:
            self.storage_gb = MIN_STORAGE_GB

    async def notify_children(self, runner):
        for c in self.children:
            n = c.n_pending_parents
            assert n > 0
            n -= 1
            c.n_pending_parents = n
            log.info(f'{c} now waiting on {n} parents')
            if n == 0:
                await runner.pool.call(runner.launch, c)

    async def set_state(self, runner, state, complete_inst_token):
        if self.state:
            return

        log.info(f'set_state {self} state {state} complete_inst_token {complete_inst_token}')

        self.state = state
        self.complete_inst_token = complete_inst_token

        # detach
        if self.active_inst:
            self.active_inst.task = None
            self.active_inst = None

        runner.n_pending -= 1
        runner.changed.set()
        if runner.n_pending == 0:
            log.info('all tasks complete, waiting for instances to drain')

        await self.notify_children(runner)

    def __str__(self):
        return f'task {self.task.name} {self.token}'


class Instance:
    def __init__(self, runner, t, inst_token):
        self.runner = runner

        if not t.active_inst:
            t.active_inst = self
            self.task = t
        else:
            self.task = None

        self.token = inst_token
        self.last_updated = time.time()

        runner.token_inst[inst_token] = self
        runner.instances.add(self)

    async def detach(self):
        if self.task and self.task.active_inst is self:
            t = self.task
            self.task = None
            t.active_inst = None
            await self.runner.pool.call(self.runner.launch, t)

    def update_timestamp(self):
        if self in self.runner.instances:
            self.runner.instances.remove(self)
            self.last_updated = time.time()
            self.runner.instances.add(self)

    async def mark_deleted(self):
        await self.detach()
        self.runner.instances.remove(self)
        if self.token in self.runner.token_inst:
            del self.runner.token_inst[self.token]
        self.runner.inst_semaphore.release()
        self.runner.changed.set()

    async def mark_preempted(self):
        await self.detach()
        await self.runner.gservices.delete_instance(f'pipeline-{self.token}')
        log.info('deleted preempted {self}')

    async def mark_complete(self):
        self.update_timestamp()

    async def heal(self):
        try:
            spec = await self.runner.gservices.get_instance(f'pipeline-{self.token}')
        except googleapiclient.errors.HttpError as e:
            if e.resp['status'] == '404':
                await self.mark_deleted()
                return

        status = spec['status']

        if status in ('TERMINATED', 'STOPPING'):
            await self.detach()

        if not self.task:
            await self.runner.gservices.delete_instance(f'pipeline-{self.token}')
            log.info('heal: deleted {self}')

        self.update_timestamp()

    def __str__(self):
        return f'inst {self.token} for {self.task}'

class GRunner:
    def gs_input_path(self, resource):
        if isinstance(resource, InputResourceFile):
            return resource._input_path

        assert isinstance(resource, TaskResourceFile)
        complete_inst_token = self.task_gtask[resource._source].complete_inst_token
        return resource._get_path(f'{self.scratch_dir}/{complete_inst_token}')

    def gs_output_paths(self, resource, inst_token):
        assert isinstance(resource, TaskResourceFile)
        output_paths = [resource._get_path(f'{self.scratch_dir}/{inst_token}')]
        if resource._output_paths:
            for p in resource._output_paths:
                output_paths.append(p)
        return output_paths

    def __init__(self, pipeline, verbose, scratch_dir):
        self.pipeline = pipeline
        self.verbose = verbose

        self.gservices = GServices()

        self.pool = AsyncWorkerPool(16)

        self.scratch_dir = scratch_dir
        parsed_scratch_dir = urllib.parse.urlparse(self.scratch_dir)

        self.scratch_dir_bucket_name = parsed_scratch_dir.netloc
        self.scratch_dir_bucket = None # filled in by async_init

        self.scratch_dir_path = parsed_scratch_dir.path
        while self.scratch_dir_path and self.scratch_dir_path[0] == '/':
            self.scratch_dir_path = self.scratch_dir_path[1:]

        self.inst_semaphore = asyncio.Semaphore(10)
        self.changed = asyncio.Event()

        self.n_pending = len(pipeline._tasks)

        self.tasks = []
        self.token_task = {}
        self.task_gtask = {}
        for pt in pipeline._tasks:
            t = GTask(pt)
            self.token_task[t.token] = t
            self.tasks.append(t)
            self.task_gtask[pt] = t

        for t in self.tasks:
            for pp in t.task._dependencies:
                p = self.task_gtask[pp]
                t.parents.add(p)
                p.children.add(t)

        self.token_inst = {}
        self.instances = sortedcontainers.SortedSet(key=lambda inst: inst.last_updated)

        self.app = web.Application()
        self.app.add_routes([
            web.post('/status', self.handle_status)
        ])

    async def async_init(self):
        self.scratch_dir_bucket = await self.gservices.get_bucket(self.scratch_dir_bucket_name)

    async def handle_status(self, request):
        status = await request.json()
        await self.mark_complete(status)
        return web.Response()

    async def set_state(self, t, state, complete_inst_token):
        await t.set_state(self, state, complete_inst_token)
        self.changed.set()

    async def mark_complete(self, status):
        task_token = status['task_token']
        complete_inst_token = status['inst_token']
        t = self.token_task[task_token]

        if all(status.get(name, 0) == 0 for name in ['input', 'main', 'output']):
            state = 'OK'
        else:
            state = 'BAD'
            log.error(f'{t} failed status {status} logs in {self.scratch_dir}/{complete_inst_token}')

        await self.set_state(t, state, complete_inst_token)

        complete_inst = self.token_inst.get(complete_inst_token)
        if complete_inst:
            await complete_inst.mark_complete()

    async def launch2(self, t):
        pt = t.task
        inst_token = secrets.token_hex(16)

        inputs_cmd = ' && '.join([
            f'gsutil -m cp -r {shq(self.gs_input_path(i))} {shq(i._get_path("/shared"))}'
            for i in pt._inputs
        ]) if pt._inputs else None

        bash_flags = 'set -e' + ('x' if self.verbose else '') + '; '
        defs = ''.join([r._declare('/shared') + '; ' for r in pt._mentioned])
        make_local_tmpdir = f'mkdir -p /shared/{pt._uid}'
        cmd = bash_flags + defs + make_local_tmpdir + ' && (' + ' && '.join(pt._command) + ')'

        outputs = pt._internal_outputs.union(pt._external_outputs)
        outputs_cmd = ' && '.join([
            f'gsutil -m cp -r {shq(o._get_path("/shared"))} {shq(output_path)}'
            for o in outputs for output_path in self.gs_output_paths(o, inst_token)
        ]) if pt._outputs else None

        assert pt._image
        config = {
            'master': 'cs-hack-master',
            'uid': pt.uid,
            'name': pt.name,
            'inst_token': inst_token,
            'task_token': t.token,
            'scratch_dir': self.scratch_dir,
            'inputs_cmd': inputs_cmd,
            'image': pt._image,
            'command': cmd,
            'outputs_cmd': outputs_cmd
        }

        await self.gservices.upload_from_string(
            self.scratch_dir_bucket,
            f'{self.scratch_dir_path}/{inst_token}/config.json',
            json.dumps(config))
        log.info(f'uploaded {t} {inst_token} config.json')

        config = {
            'name': f'pipeline-{inst_token}',
            'machineType': f'projects/{PROJECT}/zones/{ZONE}/machineTypes/n1-standard-{t.cores}',
            'labels': {
                'role': 'pipeline_worker',
                'inst_token': inst_token
            },

            # Specify the boot disk and the image to use as a source.
            'disks': [{
                'boot': True,
                'autoDelete': True,
                'diskSizeGb': str(t.storage_gb),
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
                # FIXME
                'email': '842871226259-compute@developer.gserviceaccount.com',
                'scopes': [
                    'https://www.googleapis.com/auth/cloud-platform'
                ]
            }],

            # Metadata is readable from the instance and allows you to
            # pass configuration from deployment scripts to instances.
            'metadata': {
                'items': [{
                    'key': 'master',
                    'value': 'cs-hack-master'
                }, {
                    'key': 'inst_dir',
                    'value': f'{self.scratch_dir}/{inst_token}'
                }, {
                    'key': 'startup-script-url',
                    'value': 'gs://hail-cseed/cs-hack/task-startup.sh'
                }]
            }
        }

        spec = await self.gservices.create_instance(config)
        log.info(f'created instance pipeline-{inst_token} for {t}: {spec}')

        return Instance(self, t, inst_token)

    async def launch(self, t):
        if any(p.state != 'OK' for p in t.parents):
            await self.set_state(t, 'SKIPPED', None)
            return

        delay = 1
        while True:
            if t.state:
                return
            if t.active_inst:
                return

            await self.inst_semaphore.acquire()
            try:
                await self.launch2(t)
                return
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception('launch {t} failed due to exception, will retry')
                self.inst_semaphore.release()
                await asyncio.sleep(delay * random.uniform(1, 1.25))
                delay = min(delay * 2, 180)

    async def handle_event(self, event):
        if not event.payload:
            log.warning(f'event has no payload')
            return

        payload = event.payload
        version = payload['version']
        if version != '1.2':
            log.warning('unknown log event verison {version}')
            return

        event_type = payload['event_type']
        event_subtype = payload['event_subtype']
        resource = payload['resource']
        name = resource['name']

        log.info(f'event {version} {event_type} {event_subtype} {name}')

        if event_type == 'GCE_OPERATION_DONE':
            if event_subtype in ('compute.instances.delete', 'compute.instances.preempted'):
                if name.startswith('pipeline-'):
                    inst_token = name[9:]
                    inst = self.token_inst.get(inst_token)
                    if inst:
                        if event_subtype == 'compute.instances.delete':
                            await inst.mark_deleted()
                            log.info(f'{inst} marked deleted')
                        elif event_subtype == 'compute.instances.preempted':
                            await inst.mark_preempted()
                            log.info(f'{inst} marked preempted')
                    else:
                        log.warning(f'event for unknown instance {name}')

    async def event_loop(self):
        while True:
            try:
                async for event in await self.gservices.stream_entries():
                    await self.handle_event(event)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception('event loop failed due to exception')

    async def heal(self):
        while True:
            try:
                if self.instances:
                    # 0 is the smalltest (oldest)
                    inst = self.instances[0]
                    if time.time() - inst.last_updated > 60:
                        await inst.heal()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception('heal failed due to exception')
            await asyncio.sleep(1)

    async def run(self):
        await self.async_init()

        log.info(f'running pipeline...')

        app_runner = None
        site = None
        try:
            app_runner = web.AppRunner(self.app)
            await app_runner.setup()
            site = web.TCPSite(app_runner, '0.0.0.0', 5000)
            await site.start()

            asyncio.ensure_future(self.event_loop())
            asyncio.ensure_future(self.heal())

            for t in self.tasks:
                if not t.parents:
                    await self.pool.call(self.launch, t)

            while self.n_pending != 0 or self.instances:
                await self.changed.wait()
                log.info(f'changed n_pending {self.n_pending} n_instances {len(self.instances)}')
                self.changed.clear()
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
    def __init__(self, scratch_dir):
        self.scratch_dir = scratch_dir

    def _run(self, pipeline, dry_run, verbose, delete_scratch_on_exit):
        if dry_run:
            print('do stuff')
            return

        runner = GRunner(pipeline, verbose, self.scratch_dir)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(runner.run())
        loop.run_until_complete(loop.shutdown_asyncgens())
