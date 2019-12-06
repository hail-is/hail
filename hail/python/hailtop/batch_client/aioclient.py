import math
import random
import logging
import json
import functools
import asyncio
import aiohttp
from asyncinit import asyncinit

from hailtop.config import get_deploy_config
from hailtop.auth import async_get_userinfo, service_auth_headers
from hailtop.utils import bounded_gather, request_retry_transient_errors

from .globals import tasks, complete_states

log = logging.getLogger('batch_client.aioclient')


class Job:
    @staticmethod
    def _get_error(job_status, task):
        status = job_status.get('status')
        if not status:
            return None

        # don't return status error

        container_statuses = status.get('container_statuses')
        if not container_statuses:
            return None

        container_status = container_statuses.get(task)
        if not container_status:
            return None

        error = container_status.get('error')
        if error:
            return error

        docker_container_status = container_status.get('container_status')
        if not docker_container_status:
            return None

        return docker_container_status.get('error')

    @staticmethod
    def _get_out_of_memory(job_status, task):
        status = job_status.get('status')
        if not status:
            return None

        container_statuses = status.get('container_statuses')
        if not container_statuses:
            return None

        container_status = container_statuses.get(task)
        if not container_status:
            return None

        docker_container_status = container_status.get('container_status')
        if not docker_container_status:
            return None

        return docker_container_status['out_of_memory']

    @staticmethod
    def _get_container_status_exit_code(container_status):
        docker_container_status = container_status.get('container_status')
        if not docker_container_status:
            return None

        return docker_container_status.get('exit_code')

    @staticmethod
    def _get_exit_code(job_status, task):
        status = job_status.get('status')
        if not status:
            return None

        container_statuses = status.get('container_statuses')
        if not container_statuses:
            return None

        container_status = container_statuses.get(task)
        if not container_status:
            return None

        return Job._get_container_status_exit_code(container_status)

    @staticmethod
    def _get_exit_codes(job_status):
        status = job_status.get('status')
        if not status:
            return None

        container_statuses = status.get('container_statuses')
        if not container_statuses:
            return None

        return {
            task: Job._get_container_status_exit_code(container_status)
            for task, container_status in container_statuses.items()
        }

    @staticmethod
    def exit_code(job_status):
        exit_codes = Job._get_exit_codes(job_status)
        if exit_codes is None:
            return None

        exit_codes = [
            exit_codes[task]
            for task in tasks
            if task in exit_codes
        ]

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
    def total_duration_msecs(job_status):
        status = job_status.get('status')
        if not status:
            return None

        container_statuses = status.get('container_statuses')
        if not container_statuses:
            return None

        def _get_duration(container_status):
            if not container_status:
                return None

            timing = container_status.get('timing')
            if not timing:
                return None

            runtime = timing.get('runtime')
            if not runtime:
                return None

            return runtime.get('duration')

        durations = [
            _get_duration(container_status)
            for task, container_status in container_statuses.items()
        ]

        if any(d is None for d in durations):
            return None
        return sum(durations)

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
        resp = await self._batch._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}')
        self._status = await resp.json()
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
        resp = await self._batch._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/log')
        return await resp.json()


class Batch:
    def __init__(self, client, id, attributes):
        self._client = client
        self.id = id
        self.attributes = attributes

    async def cancel(self):
        await self._client._patch(f'/api/v1alpha/batches/{self.id}/cancel')

    async def jobs(self):
        last_job_id = None
        while True:
            params = {}
            if last_job_id is not None:
                params['last_job_id'] = last_job_id
            resp = await self._client._get(f'/api/v1alpha/batches/{self.id}/jobs', params=params)
            body = await resp.json()
            for job in body['jobs']:
                yield job
            last_job_id = body.get('last_job_id')
            if last_job_id is None:
                break

    async def status(self):
        resp = await self._client._get(f'/api/v1alpha/batches/{self.id}')
        return await resp.json()

    async def wait(self):
        i = 0
        while True:
            status = await self.status()
            if status['complete']:
                return status
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
        self._job_specs = []
        self._jobs = []
        self._submitted = False
        self.attributes = attributes
        self.callback = callback

    def create_job(self, image, command, env=None, mount_docker_socket=False,
                   resources=None, secrets=None,
                   service_account=None, attributes=None, parents=None,
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

        job_spec = {
            'always_run': always_run,
            'command': command,
            'image': image,
            'job_id': self._job_idx,
            'mount_docker_socket': mount_docker_socket,
            'parent_ids': parent_ids
        }

        if env:
            job_spec['env'] = [{'name': k, 'value': v} for (k, v) in env.items()]
        if resources:
            job_spec['resources'] = resources
        if secrets:
            job_spec['secrets'] = secrets
        if service_account:
            job_spec['service_account'] = service_account

        if attributes:
            job_spec['attributes'] = attributes
        if input_files:
            job_spec['input_files'] = [{"from": src, "to": dst} for (src, dst) in input_files]
        if output_files:
            job_spec['output_files'] = [{"from": src, "to": dst} for (src, dst) in output_files]
        if pvc_size:
            job_spec['pvc_size'] = pvc_size

        self._job_specs.append(job_spec)

        j = Job.unsubmitted_job(self, self._job_idx, attributes, parent_ids)
        self._jobs.append(j)
        return j

    async def _submit_jobs(self, batch_id, byte_job_specs):
        assert len(byte_job_specs) > 0

        b = bytearray()
        b.append(ord('['))

        i = 0
        while i < len(byte_job_specs):
            spec = byte_job_specs[i]
            if i > 0:
                b.append(ord(','))
            b.extend(spec)
            i += 1

        b.append(ord(']'))

        await self._client._post(
            f'/api/v1alpha/batches/{batch_id}/jobs/create',
            data=aiohttp.BytesPayload(
                b, content_type='application/json', encoding='utf-8'))

    async def submit(self):
        if self._submitted:
            raise ValueError("cannot submit an already submitted batch")
        self._submitted = True

        batch_spec = {'billing_project': self._client.billing_project, 'n_jobs': len(self._job_specs)}
        if self.attributes:
            batch_spec['attributes'] = self.attributes
        if self.callback:
            batch_spec['callback'] = self.callback

        b_resp = await self._client._post('/api/v1alpha/batches/create', json=batch_spec)
        b = await b_resp.json()
        log.info(f'created batch {b["id"]}')
        batch = Batch(self._client, b['id'], self.attributes)

        byte_job_specs = [json.dumps(job_spec).encode('utf-8') for job_spec in self._job_specs]

        groups = []
        group = []
        group_size = 0
        for spec in byte_job_specs:
            n = len(spec)
            if group_size + n < 1000000 and len(group) < 1000:
                group.append(spec)
                group_size += n
            else:
                groups.append(group)
                group = [spec]
                group_size = n
        if group:
            groups.append(group)

        await bounded_gather(*[functools.partial(self._submit_jobs, batch.id, group)
                               for group in groups],
                             parallelism=2)

        await self._client._patch(f'/api/v1alpha/batches/{batch.id}/close')
        log.info(f'closed batch {b["id"]}')

        for j in self._jobs:
            j._job = j._job._submit(batch)

        self._job_specs = []
        self._jobs = []
        self._job_idx = 0

        return batch


@asyncinit
class BatchClient:
    async def __init__(self, billing_project, deploy_config=None, session=None,
                       headers=None, _token=None):
        self.billing_project = billing_project

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
        return await request_retry_transient_errors(
            self._session, 'GET',
            self.url + path, params=params, headers=self._headers)

    async def _post(self, path, data=None, json=None):
        return await request_retry_transient_errors(
            self._session, 'POST',
            self.url + path, data=data, json=json, headers=self._headers)

    async def _patch(self, path):
        return await request_retry_transient_errors(
            self._session, 'PATCH',
            self.url + path, headers=self._headers)

    async def _delete(self, path):
        return await request_retry_transient_errors(
            self._session, 'DELETE',
            self.url + path, headers=self._headers)

    async def list_batches(self, q=None):
        last_batch_id = None
        while True:
            params = {}
            if q is not None:
                params['q'] = q
            if last_batch_id is not None:
                params['last_batch_id'] = last_batch_id

            resp = await self._get('/api/v1alpha/batches', params=params)
            body = await resp.json()

            for batch in body['batches']:
                yield Batch(self, batch['id'], attributes=batch.get('attributes'))
            last_batch_id = body.get('last_batch_id')
            if last_batch_id is None:
                break

    async def get_job(self, batch_id, job_id):
        b = await self.get_batch(batch_id)
        j_resp = await self._get(f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
        j = await j_resp.json()
        return Job.submitted_job(
            b,
            j['job_id'],
            attributes=j.get('attributes'),
            parent_ids=j.get('parent_ids', []),
            _status=j)

    async def get_batch(self, id):
        b_resp = await self._get(f'/api/v1alpha/batches/{id}')
        b = await b_resp.json()
        return Batch(self,
                     b['id'],
                     attributes=b.get('attributes'))

    def create_batch(self, attributes=None, callback=None):
        return BatchBuilder(self, attributes, callback)

    async def close(self):
        await self._session.close()
        self._session = None
