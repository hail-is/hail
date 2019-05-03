import os
import math
import time
import random

import hailjwt as hj

from .requests_helper import filter_params


class Job:
    def __init__(self, client, batch_id, job_id, attributes=None, parent_ids=None, _status=None):
        if parent_ids is None:
            parent_ids = []
        if attributes is None:
            attributes = {}

        self.client = client
        self.batch_id = batch_id
        self.job_id = job_id
        self.attributes = attributes
        self.parent_ids = parent_ids
        self._status = _status

    @property
    def id(self):
        return self.batch_id, self.job_id

    @property
    def _batch_is_open(self):
        return self.batch_id is None

    async def is_complete(self):
        if self._batch_is_open:
            return False

        if self._status:
            state = self._status['state']
            if state in ('Complete', 'Cancelled'):
                return True
        await self.status()
        state = self._status['state']
        return state in ('Complete', 'Cancelled')

    async def status(self):
        if self._batch_is_open:
            raise ValueError("Job is not running yet")

        self._status = await self.client._get('/batches/{}/jobs/{}'.format(*self.id))
        assert self._status['batch_id'] == self.batch_id and \
               self._status['job_id'] == self.job_id
        self._status['id'] = (self._status['batch_id'], self._status['job_id'])
        return self._status

    async def wait(self):
        if self._batch_is_open:
            raise ValueError("Job is not running yet")

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
        if self._batch_is_open:
            raise ValueError("Job is not running yet")

        return await self.client._get('/batches/{}/jobs/{}/log'.format(*self.id))


class Batch:
    def __init__(self, client, id, attributes, _doc=None, _is_open=False):
        self.client = client
        self.id = id
        self.attributes = attributes
        self._doc = _doc
        self._is_open = _is_open
        self._job_idx = 0
        self._job_docs = []
        self._jobs = []

    def create_job(self, image, command=None, args=None, env=None, ports=None,
                   resources=None, tolerations=None, volumes=None, security_context=None,
                   service_account_name=None, attributes=None, callback=None, parent_ids=None,
                   input_files=None, output_files=None, always_run=False):
        self._job_idx += 1

        if not self._is_open:
            raise ValueError("Cannot add a job to a batch that has already been run")

        if parent_ids is None:
            parent_ids = []

        def _get_pid(pid):
            if isinstance(pid, tuple):
                batch_id, job_id = pid
            else:
                assert isinstance(pid, int)
                batch_id = None
                job_id = pid

            return batch_id, job_id

        def _is_valid_pid(pid):
            batch_id, job_id = pid
            return batch_id is None and 0 < job_id < self._job_idx

        parent_ids = [_get_pid(pid) for pid in parent_ids]
        if not all([_is_valid_pid(pid) for pid in parent_ids]):
            raise ValueError("Found the following invalid parent ids:\n{}".format(
                "\n  ".join([str(pid) for pid in parent_ids if not _is_valid_pid(pid)])))

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
            'job_id': self._job_idx,
            'spec': spec,
            'parent_ids': [pid[1] for pid in parent_ids],
            'always_run': always_run,
        }
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        if input_files:
            doc['input_files'] = input_files
        if output_files:
            doc['output_files'] = output_files

        self._job_docs.append(doc)

        j = Job(self.client,
                None,
                self._job_idx,
                attributes=attributes,
                parent_ids=parent_ids)

        self._jobs.append(j)

        return j

    async def cancel(self):
        if self._is_open:
            raise ValueError("Batch is not running yet")
        await self.client._patch('/batches/{}/cancel'.format(self.id))

    async def delete(self):
        if self._is_open:
            raise ValueError("Batch is not running yet")
        await self.client._delete('/batches/{}/delete'.format(self.id))

    async def status(self):
        if self._is_open:
            raise ValueError("Batch is not running yet")
        return await self.client._get('/batches/{}'.format(self.id))

    async def wait(self):
        if self._is_open:
            raise ValueError("Batch is not running yet")

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

    async def get_job(self, id):
        if self._is_open:
            raise ValueError("Batch is not running yet")

        if isinstance(id, tuple):
            batch_id, job_id = id
            if batch_id != self.id:
                raise ValueError(f"Invalid batch id {batch_id} for batch {self.id}")
        else:
            assert isinstance(id, int)
            job_id = id

        if not 0 < job_id <= self._job_idx:
            raise ValueError(f"Invalid job id {id} for batch {self.id} with {self._job_idx} jobs")

        j = await self.client._get('/batches/{}/jobs/{}'.format(self.id, job_id))
        return Job(self.client,
                   j['batch_id'],
                   j['job_id'],
                   attributes=j.get('attributes'),
                   parent_ids=j.get('parent_ids', []),
                   _status=j)

    async def run(self):
        assert self._doc is not None and self._is_open
        self._doc['jobs'] = self._job_docs
        b = await self.client._post('/batches/create', json=self._doc)
        self.id = b['id']
        self._is_open = False
        self._doc = None
        self._job_docs = []
        self._job_idx = 0

        for job in self._jobs:
            job.batch_id = self.id


class BatchClient:
    def __init__(self, session, url=None, token_file=None, token=None, headers=None):
        if not url:
            url = 'http://batch.default'
        self.url = url
        self._session = session
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

    async def _get(self, path, params=None):
        response = await self._session.get(
            self.url + path, params=params, cookies=self.cookies, headers=self.headers)
        return await response.json()

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

    async def _refresh_k8s_state(self):
        await self._post('/refresh_k8s_state')

    async def list_batches(self, complete=None, success=None, attributes=None):
        params = filter_params(complete, success, attributes)
        batches = await self._get('/batches', params=params)
        return [Batch(self,
                      j['id'],
                      attributes=j.get('attributes'))
                for j in batches]

    async def get_batch(self, id):
        b = await self._get(f'/batches/{id}')
        return Batch(self,
                     b['id'],
                     attributes=b.get('attributes'))

    def create_batch(self, attributes=None, callback=None):
        doc = {}
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        return Batch(self,
                     id=None,
                     attributes=attributes,
                     _doc=doc,
                     _is_open=True)

    async def close(self):
        await self._session.close()
        self._session = None
