import math
import random
import asyncio
import aiohttp
from asyncinit import asyncinit

from hailtop.config import get_deploy_config
from hailtop.auth import async_get_userinfo, service_auth_headers

from .globals import complete_states

job_array_size = 50
max_job_submit_attempts = 3


def filter_params(complete, success, attributes):
    params = None
    if complete is not None:
        if not params:
            params = {}
        params['complete'] = '1' if complete else '0'
    if success is not None:
        if not params:
            params = {}
        params['success'] = '1' if success else '0'
    if attributes is not None:
        if not params:
            params = {}
        for n, v in attributes.items():
            params[f'a:{n}'] = v
    return params


class Job:
    @staticmethod
    def exit_code(job_status):
        if 'exit_code' not in job_status or job_status['exit_code'] is None:
            return None

        exit_codes = job_status['exit_code']
        exit_codes = [exit_codes[task] for task in ['setup', 'main', 'cleanup'] if task in exit_codes]

        i = 0
        while i < len(exit_codes):
            ec = exit_codes[i]
            if ec is None:
                return None
            if ec > 0:
                return ec
            i += 1
        return 0

    @staticmethod
    def total_duration(job_status):
        if 'duration' not in job_status or job_status['duration'] is None:
            return None

        durations = job_status['duration']
        durations = [durations[task] for task in ['setup', 'main', 'cleanup'] if task in durations]

        i = 0
        duration = 0
        while i < len(durations):
            d = durations[i]
            if d is None:
                return None
            duration += d
            i += 1
        return duration

    @staticmethod
    def unsubmitted_job(batch_builder, job_id, attributes=None, parent_ids=None):
        assert isinstance(batch_builder, BatchBuilder)
        _job = UnsubmittedJob(batch_builder, job_id, attributes, parent_ids)
        return Job(_job)

    @staticmethod
    def submitted_job(batch, job_id, attributes=None, parent_ids=None, _status=None):
        assert isinstance(batch, Batch)
        _job = SubmittedJob(batch, job_id, attributes, parent_ids, _status)
        return Job(_job)

    def __init__(self, job):
        self._job = job

    @property
    def batch_id(self):
        return self._job.batch_id

    @property
    def job_id(self):
        return self._job.job_id

    @property
    def id(self):
        return self._job.id

    @property
    def attributes(self):
        return self._job.attributes

    @property
    def parent_ids(self):
        return self._job.parent_ids

    async def is_complete(self):
        return await self._job.is_complete()

    async def status(self):
        return await self._job.status()

    @property
    def _status(self):
        return self._job._status

    async def wait(self):
        return await self._job.wait()

    async def log(self):
        return await self._job.log()

    async def pod_status(self):
        return await self._job.pod_status()


class UnsubmittedJob:
    def _submit(self, batch):
        return SubmittedJob(batch, self._job_id, self.attributes, self.parent_ids)

    def __init__(self, batch_builder, job_id, attributes=None, parent_ids=None):
        if parent_ids is None:
            parent_ids = []
        if attributes is None:
            attributes = {}

        self._batch_builder = batch_builder
        self._job_id = job_id
        self.attributes = attributes
        self.parent_ids = parent_ids

    @property
    def batch_id(self):
        raise ValueError("cannot get the batch_id of an unsubmitted job")

    @property
    def job_id(self):
        raise ValueError("cannot get the job_id of an unsubmitted job")

    @property
    def id(self):
        raise ValueError("cannot get the id of an unsubmitted job")

    async def is_complete(self):
        raise ValueError("cannot determine if an unsubmitted job is complete")

    async def status(self):
        raise ValueError("cannot get the status of an unsubmitted job")

    @property
    def _status(self):
        raise ValueError("cannot get the _status of an unsubmitted job")

    async def wait(self):
        raise ValueError("cannot wait on an unsubmitted job")

    async def log(self):
        raise ValueError("cannot get the log of an unsubmitted job")

    async def pod_status(self):
        raise ValueError("cannot get the pod status of an unsubmitted job")


class SubmittedJob:
    def __init__(self, batch, job_id, attributes=None, parent_ids=None, _status=None):
        if parent_ids is None:
            parent_ids = []
        if attributes is None:
            attributes = {}

        self._batch = batch
        self.batch_id = batch.id
        self.job_id = job_id
        self.id = (self.batch_id, self.job_id)
        self.attributes = attributes
        self.parent_ids = parent_ids
        self._status = _status

    async def is_complete(self):
        if self._status:
            state = self._status['state']
            if state in complete_states:
                return True
        await self.status()
        state = self._status['state']
        return state in complete_states

    async def status(self):
        self._status = await self._batch._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}')
        return self._status

    async def wait(self):
        i = 0
        while True:
            if await self.is_complete():
                return self._status
            j = random.randrange(math.floor(1.1 ** i))
            await asyncio.sleep(0.100 * j)
            # max 44.5s
            if i < 64:
                i = i + 1

    async def log(self):
        return await self._batch._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/log')

    async def pod_status(self):
        return await self._batch._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/pod_status')


class Batch:
    def __init__(self, client, id, attributes):
        self._client = client
        self.id = id
        self.attributes = attributes

    async def cancel(self):
        await self._client._patch(f'/api/v1alpha/batches/{self.id}/cancel')

    async def status(self, limit=None, offset=None):
        params = None
        if limit is not None:
            if not params:
                params = {}
            params['limit'] = str(limit)
        if offset is not None:
            if limit is None:
                raise ValueError("cannot define 'offset' without a 'limit'")
            params['offset'] = str(offset)
        return await self._client._get(f'/api/v1alpha/batches/{self.id}', params=params)

    async def wait(self):
        i = 0
        while True:
            status = await self.status(limit=0)
            if status['complete']:
                return await self.status()
            j = random.randrange(math.floor(1.1 ** i))
            await asyncio.sleep(0.100 * j)
            # max 44.5s
            if i < 64:
                i = i + 1

    async def delete(self):
        await self._client._delete(f'/api/v1alpha/batches/{self.id}')


class BatchBuilder:
    def __init__(self, client, attributes, callback):
        self._client = client
        self._job_idx = 0
        self._job_docs = []
        self._jobs = []
        self._submitted = False
        self.attributes = attributes
        self.callback = callback

    def create_job(self, image, command=None, args=None, env=None, ports=None,
                   resources=None, tolerations=None, volumes=None, security_context=None,
                   service_account_name=None, attributes=None, callback=None, parents=None,
                   input_files=None, output_files=None, always_run=False, pvc_size=None):
        if self._submitted:
            raise ValueError("cannot create a job in an already submitted batch")

        self._job_idx += 1

        if parents is None:
            parents = []

        parent_ids = []
        foreign_batches = []
        invalid_job_ids = []
        for parent in parents:
            job = parent._job
            if isinstance(job, UnsubmittedJob):
                if job._batch_builder != self:
                    foreign_batches.append(job)
                elif not 0 < job._job_id < self._job_idx:
                    invalid_job_ids.append(job)
                else:
                    parent_ids.append(job._job_id)
            else:
                foreign_batches.append(job)

        error_msg = []
        if len(foreign_batches) != 0:
            error_msg.append('Found {} parents from another batch:\n{}'.format(str(len(foreign_batches)),
                                                                               "\n".join([str(j) for j in foreign_batches])))
        if len(invalid_job_ids) != 0:
            error_msg.append('Found {} parents with invalid job ids:\n{}'.format(str(len(invalid_job_ids)),
                                                                                 "\n".join([str(j) for j in invalid_job_ids])))
        if error_msg:
            raise ValueError("\n".join(error_msg))

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
            'job_id': self._job_idx
        }
        if attributes:
            doc['attributes'] = attributes
        if callback:
            doc['callback'] = callback
        if input_files:
            doc['input_files'] = input_files
        if output_files:
            doc['output_files'] = output_files
        if pvc_size:
            doc['pvc_size'] = pvc_size

        self._job_docs.append(doc)

        j = Job.unsubmitted_job(self, self._job_idx, attributes, parent_ids)
        self._jobs.append(j)
        return j

    async def _submit_job_with_retry(self, batch_id, docs):
        n_attempts = 0
        saved_err = None
        while n_attempts < max_job_submit_attempts:
            try:
                return await self._client._post(f'/api/v1alpha/batches/{batch_id}/jobs/create', json={'jobs': docs})
            except Exception as err:  # pylint: disable=W0703
                saved_err = err
                n_attempts += 1
                await asyncio.sleep(1)
        raise saved_err

    async def submit(self):
        if self._submitted:
            raise ValueError("cannot submit an already submitted batch")
        self._submitted = True

        batch_doc = {}
        if self.attributes:
            batch_doc['attributes'] = self.attributes
        if self.callback:
            batch_doc['callback'] = self.callback

        batch = None
        try:
            b = await self._client._post('/api/v1alpha/batches/create', json=batch_doc)
            batch = Batch(self._client, b['id'], b.get('attributes'))

            docs = []
            n = 0
            for jdoc in self._job_docs:
                n += 1
                docs.append(jdoc)
                if n == job_array_size:
                    await self._submit_job_with_retry(batch.id, docs)
                    n = 0
                    docs = []

            if docs:
                await self._submit_job_with_retry(batch.id, docs)

            await self._client._patch(f'/api/v1alpha/batches/{batch.id}/close')
        except Exception as err:  # pylint: disable=W0703
            if batch:
                await batch.cancel()
            raise err

        for j in self._jobs:
            j._job = j._job._submit(batch)

        self._job_docs = []
        self._jobs = []
        self._job_idx = 0

        return batch


@asyncinit
class BatchClient:
    async def __init__(self, deploy_config=None, session=None, headers=None, _token=None):
        if not deploy_config:
            deploy_config = get_deploy_config()

        self.url = deploy_config.base_url('batch')

        if session is None:
            session = aiohttp.ClientSession(raise_for_status=True,
                                            timeout=aiohttp.ClientTimeout(total=60))
        self._session = session

        userinfo = await async_get_userinfo(deploy_config)
        self.bucket = userinfo['bucket_name']

        h = {}
        if headers:
            h.update(headers)
        if _token:
            h['Authorization'] = f'Bearer {_token}'
        else:
            h.update(service_auth_headers(deploy_config, 'batch'))
        self._headers = h

    async def _get(self, path, params=None):
        response = await self._session.get(
            self.url + path, params=params, headers=self._headers)
        return await response.json()

    async def _post(self, path, json=None):
        response = await self._session.post(
            self.url + path, json=json, headers=self._headers)
        return await response.json()

    async def _patch(self, path):
        await self._session.patch(
            self.url + path, headers=self._headers)

    async def _delete(self, path):
        await self._session.delete(
            self.url + path, headers=self._headers)

    async def _refresh_k8s_state(self):
        await self._post('/refresh_k8s_state')

    async def list_batches(self, complete=None, success=None, attributes=None):
        params = filter_params(complete, success, attributes)
        batches = await self._get('/api/v1alpha/batches', params=params)
        return [Batch(self,
                      b['id'],
                      attributes=b.get('attributes'))
                for b in batches]

    async def get_job(self, batch_id, job_id):
        b = await self.get_batch(batch_id)
        j = await self._get(f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
        return Job.submitted_job(
            b,
            j['job_id'],
            attributes=j.get('attributes'),
            parent_ids=j.get('parent_ids', []),
            _status=j)

    async def get_batch(self, id):
        b = await self._get(f'/api/v1alpha/batches/{id}')
        return Batch(self,
                     b['id'],
                     attributes=b.get('attributes'))

    def create_batch(self, attributes=None, callback=None):
        return BatchBuilder(self, attributes, callback)

    async def close(self):
        await self._session.close()
        self._session = None
