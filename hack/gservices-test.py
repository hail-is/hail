import collections
import os
import re
import time
import secrets
import json
import logging
import asyncio
import concurrent
from aiohttp import web
from shlex import quote as shq

import googleapiclient.discovery
import google.cloud.storage
import google.cloud.logging

import warnings
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")

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
                f(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(f'worker pool caught exception')

    async def call(f, *args, **kwargs):
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
        return await self.run_in_pool(self.storage_client.get_bucket, 'hail-cseed')

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
            self.compute_client.instances().get(project=PROJECT, zone=ZONE, instance=instance).execute)

    async def create_instance(self, body):
        return await self.run_in_pool(
            self.compute_client.instances().insert(project=PROJECT, zone=ZONE, body=body).execute)

    async def delete_instance(self, instance):
        return await self.run_in_pool(
            self.compute_client.instances().delete(project=PROJECT, zone=ZONE, instance=instance).execute)

async def events_loop(gservices):
    async for event in await gservices.stream_entries():
        log.info(f'event {event.timestamp} {event.payload}')

config = {
    'name': 'cs-test-9',
    'machineType': 'projects/broad-ctsa/zones/us-central1-a/machineTypes/n1-standard-1',

    'labels': {
        'app': 'hail_pipeline',
        'role': 'pipeline_worker',
        'pipeline_token': 'foobar'
    },

    # Specify the boot disk and the image to use as a source.
    'disks': [{
        'boot': True,
        'autoDelete': True,
        'diskSizeGb': "20",
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
            'key': 'master',
            'value': 'cs-hack-master'
        }, {
            'key': 'token',
            'value': 'foobar'
        }, {
            'key': 'startup-script-url',
            'value': 'gs://hail-cseed/cs-hack/task-startup.sh'
        }]
    }
}

async def main():
    gservices = GServices()

    # works
    # hail_cseed = await gservices.get_bucket('hail-cseed')
    # await gservices.upload_from_string(hail_cseed, 'baz/quux', 'quam')

    asyncio.ensure_future(events_loop(gservices))

    await asyncio.sleep(5)

    spec = await gservices.create_instance(config)
    print(f'created instance: {spec}')

    await asyncio.sleep(60_000)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.run_until_complete(loop.shutdown_asyncgens())
