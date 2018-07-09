import json
import time
import random
import requests
import batch.api as api

class Job(object):
    def __init__(self, client, id):
        self.client = client
        self.id = id
        self._status = None

    def is_complete(self):
        if self._status:
            state = self._status['state']
            if state == 'Complete' or state == 'Cancelled':
                return True
        return False

    def status(self):
        self._status = self.client._get_job(self.id)
        return self._status

    def wait(self):
        i = 0
        while True:
            self.status() # update
            if self.is_complete():
                return self._status
            j = random.randrange(2 ** i)
            time.sleep(0.100 * j)
            # max 5.12s
            if i < 9:
                i = i + 1

    def cancel(self):
        self.client._cancel_job(self.id)

class Batch(object):
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def create_job(self, name, image, command=None, args=None, env=None):
        return self.client._create_job(name, image, command, args, env, self.name)

    def status(self):
        return self.client._get_batch(self.name)

    def wait(self):
        i = 0
        while True:
            status = self.status()
            if status['jobs'].get('Created', 0) == 0:
                return status
            j = random.randrange(2 ** i)
            time.sleep(0.100 * j)
            # max 5.12s
            if i < 9:
                i = i + 1

class BatchClient(object):
    def __init__(self, url='http://batch'):
        self.url = url

    def _create_job(self, name, image, command, args, env, batch):
        j = api.create_job(self.url, name, image, command, args, env, batch)
        return Job(self, j['id'])

    def _get_job(self, id):
        return api.get_job(self.url, id)

    def _cancel_job(self, id):
        api.cancel_job(self.url, id)

    def _get_batch(self, batch):
        return api.get_batch(self.url, batch)

    def get_job(self, id):
        # make sure job exists
        j = api.get_job(self.url, id)
        return Job(self, j['id'])

    def create_job(self, name, image, command=None, args=None, env=None):
        return self._create_job(name, image, command, args, env, None)

    def create_batch(self, name):
        return Batch(self, name)
