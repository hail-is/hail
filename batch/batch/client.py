import os
import math
import time
import random
import abc
import requests

import hailjwt as hj

from .requests_helper import raise_on_failure, filter_params


class Job:
    @staticmethod
    def exit_code(job_status):
        if 'exit_code' not in job_status or job_status['exit_code'] is None:
            return None

        exit_codes = job_status['exit_code']
        exit_codes = [exit_codes[task] for task in ['input', 'main', 'output'] if task in exit_codes]

        i = 0
        while i < len(exit_codes):
            ec = exit_codes[i]
            if ec is None:
                return None
            if ec > 0:
                return ec
            i += 1
        return 0

    def __init__(self, client, id, attributes=None, parent_ids=None, _status=None):
        if parent_ids is None:
            parent_ids = []
        if attributes is None:
            attributes = {}

        self.client = client
        self.id = id
        self.attributes = attributes
        self.parent_ids = parent_ids
        self._status = _status

    async def is_complete(self):
        if self._status:
            state = self._status['state']
            if state in ('Complete', 'Cancelled'):
                return True
        await self.status()
        state = self._status['state']
        return state in ('Complete', 'Cancelled')

    async def status(self):
        self._status = await self.client._get('/jobs/{}'.format(self.id))
        return self._status

    async def wait(self):
        i = 0
        while True:
            if await self.is_complete():
                return self._status
            j = random.randrange(math.floor(1.1 ** i))
            time.sleep(0.100 * j)
            # max 4.45s
            if i < 64:
                i = i + 1

    async def log(self):
        return await self.client._get('/jobs/{}/log'.format(self.id))


class Batch:
    def __init__(self, client, id, attributes):
        self.client = client
        self.id = id
        self.attributes = attributes

    async def create_job(self, image, command=None, args=None, env=None, ports=None,
                         resources=None, tolerations=None, volumes=None, security_context=None,
                         service_account_name=None, attributes=None, callback=None, parent_ids=None,
                         input_files=None, output_files=None, always_run=False):
        if parent_ids is None:
            parent_ids = []

        if env:
            env = [{'name': k, 'value': v} for (k, v) in env.items()]
        else:
            env = []
        env.extend([{
            'name': 'POD_IP',
            'valueFrom': {
                'fieldRef': {'fieldPath': 'status.podIP'}
            }
        }, {
            'name': 'POD_NAME',
            'valueFrom': {
                'fieldRef': {'fieldPath': 'metadata.name'}
            }
        }])

        container = {
            'image': image,
            'name': 'main'
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
        if tolerations:
            spec['tolerations'] = tolerations
        if security_context:
            spec['securityContext'] = security_context
        if service_account_name:
            spec['serviceAccountName'] = service_account_name

        doc = {
            'spec': spec,
            'parent_ids': parent_ids,
            'always_run': always_run,
            'batch_id': self.id
        }
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        if input_files:
            doc['input_files'] = input_files
        if output_files:
            doc['output_files'] = output_files

        j = await self.client._post('/jobs/create', json=doc)

        return Job(self.client,
                   j['id'],
                   attributes=j.get('attributes'),
                   parent_ids=j.get('parent_ids', []))

    async def close(self):
        await self.client._patch('/batches/{}/close'.format(self.id))

    async def cancel(self):
        await self.client._patch('/batches/{}/cancel'.format(self.id))

    async def status(self):
        return await self.client._get('/batches/{}'.format(self.id))

    async def wait(self):
        i = 0
        while True:
            status = await self.status()
            if status['complete']:
                return status
            j = random.randrange(math.floor(1.1 ** i))
            time.sleep(0.100 * j)
            # max 4.45s
            if i < 64:
                i = i + 1

    async def delete(self):
        await self.client._delete('/batches/{}/delete'.format(self.id))


class BatchClient:
    __metaclass__ = abc.ABCMeta

    def __init__(self, url=None, token_file=None, token=None, headers=None):
        if not url:
            url = 'http://batch.default'
        self.url = url
        if token is None:
            token_file = (token_file or
                          os.environ.get('HAIL_TOKEN_FILE') or
                          os.path.expanduser('~/.hail/token'))
            if not os.path.exists(token_file):
                raise ValueError(
                    f'Cannot create a client without a token. No file was '
                    f'found at {token_file}')
            with open(token_file) as f:
                token = f.read()
        userdata = hj.JWTClient.unsafe_decode(token)
        assert "bucket_name" in userdata, userdata
        self.bucket = userdata["bucket_name"]
        self.cookies = {'user': token}
        self.headers = headers

    @abc.abstractmethod
    def _get(self, path, params=None):
        pass

    @abc.abstractmethod
    def _post(self, path, json=None):
        pass

    @abc.abstractmethod
    def _patch(self, path):
        pass

    @abc.abstractmethod
    def _delete(self, path):
        pass

    async def _refresh_k8s_state(self):
        await self._post('/refresh_k8s_state')

    async def list_batches(self, complete=None, success=None, attributes=None):
        params = filter_params(complete, success, attributes)
        batches = await self._get('/batches', params=params)
        return [Batch(self,
                      j['id'],
                      attributes=j.get('attributes'))
                for j in batches]

    async def get_job(self, id):
        j = await self._get('/jobs/{}'.format(id))
        return Job(self,
                   j['id'],
                   attributes=j.get('attributes'),
                   parent_ids=j.get('parent_ids', []),
                   _status=j)

    async def get_batch(self, id):
        j = await self._get(f'/batches/{id}')
        return Batch(self,
                     j['id'],
                     attributes=j.get('attributes'))

    async def create_batch(self, attributes=None, callback=None, ttl=None):
        doc = {}
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        if ttl:
            doc['ttl'] = ttl
        j = await self._post('/batches/create', json=doc)
        return Batch(self, j['id'], j.get('attributes'))


class AsyncBatchClient(BatchClient):
    def __init__(self, session, url=None, token_file=None, token=None, headers=None):
        super().__init__(url=url, token_file=token_file, token=token, headers=headers)
        self._session = session

    async def _get(self, path, params=None):
        response = await self._session.get(
            self.url + path, params=params, cookies=self.cookies, headers=self.headers)
        return response.json()

    async def _post(self, path, json=None):
        response = await self._session.post(
            self.url + path, json=json, cookies=self.cookies, headers=self.headers)
        return await response.json()

    async def _patch(self, path):
        await self._session.patch(
            self.url + path, cookies=self.cookies, headers=self.headers)

    async def _delete(self, path):
        await self._session.delete(
            self.url + path, cookies=self.cookies, headers=self.headers)

    async def close(self):
        await self._session.close()
        self._session = None


class SyncBatchClient(BatchClient):
    def __init__(self, url=None, timeout=None, token_file=None, token=None, headers=None):
        super().__init__(url=url, token_file=token_file, token=token, headers=headers)
        self.timeout = timeout

    def _http(self, verb, url, json=None, timeout=None, cookies=None, headers=None, json_response=True, params=None):
        if timeout is None:
            timeout = self.timeout
        if cookies is None:
            cookies = self.cookies
        if headers is None:
            headers = self.headers
        if json is not None:
            response = verb(url, json=json, timeout=timeout, cookies=cookies, headers=headers, params=params)
        else:
            response = verb(url, timeout=timeout, cookies=cookies, headers=headers, params=params)
        raise_on_failure(response)
        if json_response:
            return response.json()
        return response

    def _get(self, path, params=None):
        return self._http(requests.get, self.url + path, params=params)

    def _post(self, path, json=None):
        return self._http(requests.post, self.url + path, json=json)

    def _patch(self, path):
        return self._http(requests.patch, self.url + path, json_response=False)

    def _delete(self, path):
        return self._http(requests.delete, self.url + path, json_response=False)
