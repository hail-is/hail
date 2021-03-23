from typing import Optional, Dict, Any
import math
import random
import logging
import json
import functools
import asyncio
import aiohttp
import secrets

from hailtop.config import get_deploy_config, DeployConfig
from hailtop.auth import service_auth_headers
from hailtop.utils import bounded_gather, request_retry_transient_errors, tqdm, TQDM_DEFAULT_DISABLE
from hailtop.httpx import client_session

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
        error = container_status.get('error')
        if error is not None:
            return None

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

        error = status.get('error')
        if error is not None:
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
    def unsubmitted_job(batch_builder, job_id):
        assert isinstance(batch_builder, BatchBuilder)
        _job = UnsubmittedJob(batch_builder, job_id)
        return Job(_job)

    @staticmethod
    def submitted_job(batch, job_id, _status=None):
        assert isinstance(batch, Batch)
        _job = SubmittedJob(batch, job_id, _status)
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

    async def attributes(self):
        return await self._job.attributes()

    async def is_complete(self):
        return await self._job.is_complete()

    # {
    #   batch_id: int
    #   job_id: int
    #   user: str
    #   billing_project: str
    #   name: optional(str)
    #   state: str (Ready, Running, Success, Error, Failure, Cancelled)
    #   exit_code: optional(int)
    #   duration: optional(int) (msecs)
    #   msec_mcpu: int
    #   cost: float
    # }
    async def status(self):
        return await self._job.status()

    @property
    def _status(self):
        return self._job._status

    async def wait(self):
        return await self._job.wait()

    async def log(self):
        return await self._job.log()

    async def attempts(self):
        return await self._job.attempts()


class UnsubmittedJob:
    def _submit(self, batch):
        return SubmittedJob(batch, self._job_id)

    def __init__(self, batch_builder, job_id):
        self._batch_builder = batch_builder
        self._job_id = job_id

    @property
    def batch_id(self):
        raise ValueError("cannot get the batch_id of an unsubmitted job")

    @property
    def job_id(self):
        raise ValueError("cannot get the job_id of an unsubmitted job")

    @property
    def id(self):
        raise ValueError("cannot get the id of an unsubmitted job")

    async def attributes(self):
        raise ValueError("cannot get the attributes of an unsubmitted job")

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

    async def attempts(self):
        raise ValueError("cannot get the attempts of an unsubmitted job")


class SubmittedJob:
    def __init__(self, batch, job_id, _status=None):
        self._batch = batch
        self.batch_id = batch.id
        self.job_id = job_id
        self.id = (self.batch_id, self.job_id)
        self._status = _status

    async def attributes(self):
        if not self._status:
            await self.status()
        return self._status['attributes']

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

    async def attempts(self):
        resp = await self._batch._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/attempts')
        return await resp.json()


class Batch:
    def __init__(self, client, id, attributes, n_jobs, token, last_known_status=None):
        self._client = client
        self.id = id
        self.attributes = attributes
        self.n_jobs = n_jobs
        self.token = token
        self._last_known_status = last_known_status

    async def cancel(self):
        await self._client._patch(f'/api/v1alpha/batches/{self.id}/cancel')

    async def jobs(self, q=None):
        last_job_id = None
        while True:
            params = {}
            if q is not None:
                params['q'] = q
            if last_job_id is not None:
                params['last_job_id'] = last_job_id
            resp = await self._client._get(f'/api/v1alpha/batches/{self.id}/jobs', params=params)
            body = await resp.json()
            for job in body['jobs']:
                yield job
            last_job_id = body.get('last_job_id')
            if last_job_id is None:
                break

    async def get_job_log(self, job_id: int) -> Optional[Dict[str, Any]]:
        return await self._client.get_job_log(self.id, job_id)

    # {
    #   id: int
    #   user: str
    #   billing_project: str
    #   token: str
    #   state: str, (open, failure, cancelled, success, running)
    #   complete: bool
    #   closed: bool
    #   n_jobs: int
    #   n_completed: int
    #   n_succeeded: int
    #   n_failed: int
    #   n_cancelled: int
    #   time_created: optional(str), (date)
    #   time_closed: optional(str), (date)
    #   time_completed: optional(str), (date)
    #   duration: optional(str)
    #   attributes: optional(dict(str, str))
    #   msec_mcpu: int
    #   cost: float
    # }
    async def status(self):
        resp = await self._client._get(f'/api/v1alpha/batches/{self.id}')
        self._last_known_status = await resp.json()
        return self._last_known_status

    async def last_known_status(self):
        if self._last_known_status is None:
            return await self.status()  # updates _last_known_status
        return self._last_known_status

    async def wait(self, *, disable_progress_bar=TQDM_DEFAULT_DISABLE):
        i = 0
        with tqdm(total=self.n_jobs,
                  disable=disable_progress_bar,
                  desc='completed jobs') as pbar:
            while True:
                status = await self.status()
                pbar.update(status['n_completed'] - pbar.n)
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
    def __init__(self, client, attributes, callback, token=None):
        self._client = client
        self._job_idx = 0
        self._job_specs = []
        self._jobs = []
        self._submitted = False
        self.attributes = attributes
        self.callback = callback

        if token is None:
            token = secrets.token_urlsafe(32)
        self.token = token

    def create_job(self, image, command, env=None, mount_docker_socket=False,
                   port=None, resources=None, secrets=None,
                   service_account=None, attributes=None, parents=None,
                   input_files=None, output_files=None, always_run=False,
                   timeout=None, gcsfuse=None, requester_pays_project=None,
                   mount_tokens=False, network: Optional[str] = None):
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
            'job_id': self._job_idx,
            'parent_ids': parent_ids,
            'process': {
                'command': command,
                'image': image,
                'mount_docker_socket': mount_docker_socket,
                'type': 'docker'
            }
        }

        if env:
            job_spec['env'] = [{'name': k, 'value': v} for (k, v) in env.items()]
        if port is not None:
            job_spec['port'] = port
        if resources:
            job_spec['resources'] = resources
        if secrets:
            job_spec['secrets'] = secrets
        if service_account:
            job_spec['service_account'] = service_account
        if timeout:
            job_spec['timeout'] = timeout

        if attributes:
            job_spec['attributes'] = attributes
        if input_files:
            job_spec['input_files'] = [{"from": src, "to": dst} for (src, dst) in input_files]
        if output_files:
            job_spec['output_files'] = [{"from": src, "to": dst} for (src, dst) in output_files]
        if gcsfuse:
            job_spec['gcsfuse'] = [{"bucket": bucket, "mount_path": mount_path, "read_only": read_only}
                                   for (bucket, mount_path, read_only) in gcsfuse]
        if requester_pays_project:
            job_spec['requester_pays_project'] = requester_pays_project
        if mount_tokens:
            job_spec['mount_tokens'] = mount_tokens
        if network:
            job_spec['network'] = network

        self._job_specs.append(job_spec)

        j = Job.unsubmitted_job(self, self._job_idx)
        self._jobs.append(j)
        return j

    async def _submit_jobs(self, batch_id, byte_job_specs, n_jobs, pbar):
        assert len(byte_job_specs) > 0, byte_job_specs

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
        pbar.update(n_jobs)

    async def _create(self):
        n_jobs = len(self._job_specs)
        batch_spec = {'billing_project': self._client.billing_project,
                      'n_jobs': n_jobs,
                      'token': self.token}
        if self.attributes:
            batch_spec['attributes'] = self.attributes
        if self.callback:
            batch_spec['callback'] = self.callback
        batch_json = await (await self._client._post('/api/v1alpha/batches/create',
                                                     json=batch_spec)).json()
        return Batch(self._client, batch_json['id'], self.attributes, n_jobs, self.token)

    MAX_BUNCH_BYTESIZE = 1024 * 1024
    MAX_BUNCH_SIZE = 1024

    async def submit(self,
                     max_bunch_bytesize=MAX_BUNCH_BYTESIZE,
                     max_bunch_size=MAX_BUNCH_SIZE,
                     disable_progress_bar=TQDM_DEFAULT_DISABLE):
        assert max_bunch_bytesize > 0
        assert max_bunch_size > 0
        if self._submitted:
            raise ValueError("cannot submit an already submitted batch")
        batch = await self._create()
        id = batch.id
        log.info(f'created batch {id}')
        byte_job_specs = [json.dumps(job_spec).encode('utf-8')
                          for job_spec in self._job_specs]
        byte_job_specs_bunches = []
        bunch_sizes = []
        bunch = []
        bunch_n_bytes = 0
        bunch_n_jobs = 0
        for spec in byte_job_specs:
            n_bytes = len(spec)
            assert n_bytes < max_bunch_bytesize, (
                f'every job spec must be less than max_bunch_bytesize,'
                f' { max_bunch_bytesize }B, but {spec} is larger')
            if bunch_n_bytes + n_bytes < max_bunch_bytesize and len(bunch) < max_bunch_size:
                bunch.append(spec)
                bunch_n_bytes += n_bytes
                bunch_n_jobs += 1
            else:
                byte_job_specs_bunches.append(bunch)
                bunch_sizes.append(bunch_n_jobs)
                bunch = [spec]
                bunch_n_bytes = n_bytes
                bunch_n_jobs = 1
        if bunch:
            byte_job_specs_bunches.append(bunch)
            bunch_sizes.append(bunch_n_jobs)

        with tqdm(total=len(self._job_specs),
                  disable=disable_progress_bar,
                  desc='jobs submitted to queue') as pbar:
            await bounded_gather(
                *[functools.partial(self._submit_jobs, id, bunch, size, pbar)
                  for bunch, size in zip(byte_job_specs_bunches, bunch_sizes)],
                parallelism=6)

        await self._client._patch(f'/api/v1alpha/batches/{id}/close')
        log.info(f'closed batch {id}')

        for j in self._jobs:
            j._job = j._job._submit(batch)

        self._job_specs = []
        self._jobs = []
        self._job_idx = 0

        self._submitted = True
        return batch


class BatchClient:
    def __init__(self,
                 billing_project: str,
                 deploy_config: Optional[DeployConfig] = None,
                 session: Optional[aiohttp.ClientSession] = None,
                 headers: Optional[Dict[str, str]] = None,
                 _token: Optional[str] = None,
                 token_file: Optional[str] = None):
        self.billing_project = billing_project

        if not deploy_config:
            deploy_config = get_deploy_config()

        self.url = deploy_config.base_url('batch')

        if session is None:
            session = client_session()
        self._session = session

        h: Dict[str, str] = {}
        if headers:
            h.update(headers)
        if _token:
            h['Authorization'] = f'Bearer {_token}'
        else:
            h.update(service_auth_headers(deploy_config, 'batch', token_file=token_file))
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

    async def list_batches(self, q=None, last_batch_id=None, limit=2**64):
        n = 0
        while True:
            params = {}
            if q is not None:
                params['q'] = q
            if last_batch_id is not None:
                params['last_batch_id'] = last_batch_id

            resp = await self._get('/api/v1alpha/batches', params=params)
            body = await resp.json()

            for batch in body['batches']:
                if n >= limit:
                    return
                n += 1
                yield Batch(self, batch['id'], attributes=batch.get('attributes'),
                            n_jobs=int(batch['n_jobs']), token=batch['token'], last_known_status=batch)
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
            _status=j)

    async def get_job_log(self, batch_id, job_id) -> Optional[Dict[str, Any]]:
        resp = await self._get(
            f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
        return await resp.json()

    async def get_job_attempts(self, batch_id, job_id):
        resp = await self._get(
            f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/attempts')
        return await resp.json()

    async def get_batch(self, id):
        b_resp = await self._get(f'/api/v1alpha/batches/{id}')
        b = await b_resp.json()
        return Batch(self,
                     b['id'],
                     attributes=b.get('attributes'),
                     n_jobs=int(b['n_jobs']),
                     token=b['token'],
                     last_known_status=b)

    def create_batch(self, attributes=None, callback=None, token=None):
        return BatchBuilder(self, attributes, callback, token)

    async def get_billing_project(self, billing_project):
        bp_resp = await self._get(f'/api/v1alpha/billing_projects/{billing_project}')
        return await bp_resp.json()

    async def list_billing_projects(self):
        bp_resp = await self._get('/api/v1alpha/billing_projects')
        return await bp_resp.json()

    async def create_billing_project(self, project):
        bp_resp = await self._post(f'/api/v1alpha/billing_projects/{project}/create')
        return await bp_resp.json()

    async def add_user(self, user, project):
        resp = await self._post(f'/api/v1alpha/billing_projects/{project}/users/{user}/add')
        return await resp.json()

    async def remove_user(self, user, project):
        resp = await self._post(f'/api/v1alpha/billing_projects/{project}/users/{user}/remove')
        return await resp.json()

    async def close_billing_project(self, project):
        bp_resp = await self._post(f'/api/v1alpha/billing_projects/{project}/close')
        return await bp_resp.json()

    async def reopen_billing_project(self, project):
        bp_resp = await self._post(f'/api/v1alpha/billing_projects/{project}/reopen')
        return await bp_resp.json()

    async def delete_billing_project(self, project):
        bp_resp = await self._post(f'/api/v1alpha/billing_projects/{project}/delete')
        return await bp_resp.json()

    async def edit_billing_limit(self, project, limit):
        bp_resp = await self._post(f'/api/v1alpha/billing_limits/{project}/edit',
                                   json={'limit': limit})
        return await bp_resp.json()

    async def close(self):
        await self._session.close()
        self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
