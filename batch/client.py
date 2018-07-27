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
    def __init__(self, client, id):
        self.client = client
        self.id = id

    def create_job(self, image, command=None, args=None, env=None, ports=None,
                   attributes=None, callback=None):
        return self.client._create_job(image, command, args, env, ports, attributes, self.id, callback)

    def status(self):
        return self.client._get_batch(self.id)

    def wait(self):
        i = 0
        while True:
            status = self.status()
            if status['jobs']['Created'] == 0:
                return status
            j = random.randrange(2 ** i)
            time.sleep(0.100 * j)
            # max 5.12s
            if i < 9:
                i = i + 1

class BatchClient(object):
    def __init__(self, url=None):
        if not url:
            url = 'http://batch'
        self.url = url

    def _create_job(self, image, command, args, env, ports, resources, volumes, attributes, batch_id, callback):
        if env:
            env = [{'name': k, 'value': v} for (k, v) in env.items()]
        else:
            env = []
        env.append({
            'name': 'POD_IP',
            'valueFrom': {
                'fieldRef': {'fieldPath': 'status.podIP'}
            }
        })

        container = {
            'image': image,
            'name': 'default'
        }
        if command:
            container['command'] = command
        if args:
            container['args'] = args
        if env:
            container['env'] = env
        if ports:
            container['ports'] = [{
                'containerPort': p,
                'protocol': 'TCP'
            } for p in ports]
        if resources:
            container['resources'] = resources
        if volumes:
            container['volumeMounts'] = [v['volume_mount'] for v in volumes]
        spec = {
            'containers': [container],
            'restartPolicy': 'Never'
        }
        if volumes:
            spec['volumes'] = [v['volume'] for v in volumes]

        j = api.create_job(self.url, spec, attributes, batch_id, callback)
        return Job(self, j['id'])

    def _get_job(self, id):
        return api.get_job(self.url, id)

    def _cancel_job(self, id):
        api.cancel_job(self.url, id)

    def _get_batch(self, batch_id):
        return api.get_batch(self.url, batch_id)

    def get_job(self, id):
        # make sure job exists
        j = api.get_job(self.url, id)
        return Job(self, j['id'])

    def create_job(self,
                   image,
                   command=None,
                   args=None,
                   env=None,
                   ports=None,
                   resources=None,
                   volumes=None,
                   attributes=None,
                   callback=None):
        return self._create_job(image, command, args, env, ports, resources, volumes, attributes, None, callback)

    def create_batch(self, attributes=None):
        b = api.create_batch(self.url, attributes)
        return Batch(self, b['id'])
