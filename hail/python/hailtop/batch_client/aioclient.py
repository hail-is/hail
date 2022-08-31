from typing import Optional, Dict, Any, List, Tuple, Union
import math
import random
import logging
import json
import functools
import asyncio
import aiohttp

from hailtop.config import get_deploy_config, DeployConfig
from hailtop.auth import service_auth_headers
from hailtop.utils import bounded_gather, request_retry_transient_errors, secret_alnum_string, tqdm, TqdmDisableOption
from hailtop import httpx

from .globals import tasks, complete_states

log = logging.getLogger('batch_client.aioclient')

MAX_BUNCH_BYTESIZE = 1024 * 1024
MAX_BUNCH_SIZE = 1024


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

        return container_status.get('error')

    @staticmethod
    def _get_out_of_memory(job_status, task):
        status = job_status.get('status')
        if not status:
            return None

        container_statuses = status.get('container_statuses')
        if not container_statuses:
            return None

        task_status = container_statuses.get(task)
        if not task_status:
            return None

        container_status = task_status.get('container_status')
        if not container_status:
            return None

        return container_status['out_of_memory']

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

        exit_codes = [exit_codes[task] for task in tasks if task in exit_codes]

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

        durations = [_get_duration(container_status) for _, container_status in container_statuses.items()]

        if any(d is None for d in durations):
            return None
        return sum(durations)

    @staticmethod
    def unsubmitted_job(batch_builder: 'BatchBuilder', spec: 'JobSpec'):
        return Job(UnsubmittedJob(batch_builder, spec))

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

    @property
    def _spec(self):
        return self._job._spec

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


class JobSpec:
    def __init__(self,
                 relative_job_id: int,
                 process: dict,
                 *,
                 env: Optional[Dict[str, str]] = None,
                 port: Optional[int] = None,
                 resources: Optional[dict] = None,
                 secrets: Optional[dict] = None,
                 service_account: Optional[str] = None,
                 attributes: Optional[Dict[str, str]] = None,
                 parents: Optional[List[Job]] = None,
                 input_files: Optional[List[Tuple[str, str]]] = None,
                 output_files: Optional[List[Tuple[str, str]]] = None,
                 always_run: bool = False,
                 timeout: Optional[Union[int, float]] = None,
                 cloudfuse: Optional[List[Tuple[str, str, bool]]] = None,
                 requester_pays_project: Optional[str] = None,
                 mount_tokens: bool = False,
                 network: Optional[str] = None,
                 unconfined: bool = False,
                 user_code: Optional[str] = None
                 ):
        self.relative_job_id = relative_job_id
        self.process = process
        self.env = env
        self.port = port
        self.resources = resources
        self.secrets = secrets
        self.service_account = service_account
        self.attributes = attributes
        self.parents = parents or []
        self.input_files = input_files
        self.output_files = output_files
        self.always_run = always_run
        self.timeout = timeout
        self.cloudfuse = cloudfuse
        self.requester_pays_project = requester_pays_project
        self.mount_tokens = mount_tokens
        self.network = network
        self.unconfined = unconfined
        self.user_code = user_code

    def to_json(self):
        absolute_parent_ids = []
        relative_parent_ids = []
        for parent in self.parents:
            job = parent._job
            if isinstance(job, UnsubmittedJob):
                relative_parent_ids.append(job._spec.relative_job_id)
            else:
                assert isinstance(job, SubmittedJob)
                absolute_parent_ids.append(job.job_id)

        job_spec = {
            'always_run': self.always_run,
            'relative_job_id': self.relative_job_id,
            'absolute_parent_ids': absolute_parent_ids,
            'relative_parent_ids': relative_parent_ids,
            'process': self.process,
        }

        if self.env:
            job_spec['env'] = [{'name': k, 'value': v} for (k, v) in self.env.items()]
        if self.port is not None:
            job_spec['port'] = self.port
        if self.resources:
            job_spec['resources'] = self.resources
        if self.secrets:
            job_spec['secrets'] = self.secrets
        if self.service_account:
            job_spec['service_account'] = self.service_account
        if self.timeout:
            job_spec['timeout'] = self.timeout

        if self.attributes:
            job_spec['attributes'] = self.attributes
        if self.input_files:
            job_spec['input_files'] = [{"from": src, "to": dst} for (src, dst) in self.input_files]
        if self.output_files:
            job_spec['output_files'] = [{"from": src, "to": dst} for (src, dst) in self.output_files]
        if self.cloudfuse:
            job_spec['cloudfuse'] = [{"bucket": bucket, "mount_path": mount_path, "read_only": read_only}
                                     for (bucket, mount_path, read_only) in self.cloudfuse]
        if self.requester_pays_project:
            job_spec['requester_pays_project'] = self.requester_pays_project
        if self.mount_tokens:
            job_spec['mount_tokens'] = self.mount_tokens
        if self.network:
            job_spec['network'] = self.network
        if self.unconfined:
            job_spec['unconfined'] = self.unconfined
        if self.user_code:
            job_spec['user_code'] = self.user_code

        return job_spec


class UnsubmittedJob:
    def _submit(self, batch: 'Batch', job_id: int):
        return SubmittedJob(batch, job_id)

    def __init__(self, batch_builder: 'BatchBuilder', spec: JobSpec):
        self._batch_builder = batch_builder
        self._spec = spec

    @property
    def batch_id(self):  # pylint: disable=no-self-use
        raise ValueError("cannot get the batch_id of an unsubmitted job")

    @property
    def job_id(self):  # pylint: disable=no-self-use
        raise ValueError("cannot get the job_id of an unsubmitted job")

    @property
    def id(self):  # pylint: disable=no-self-use
        raise ValueError("cannot get the id of an unsubmitted job")

    async def attributes(self):  # pylint: disable=no-self-use
        raise ValueError("cannot get the attributes of an unsubmitted job")

    async def is_complete(self):  # pylint: disable=no-self-use
        raise ValueError("cannot determine if an unsubmitted job is complete")

    async def status(self):  # pylint: disable=no-self-use
        raise ValueError("cannot get the status of an unsubmitted job")

    @property
    def _status(self):  # pylint: disable=no-self-use
        raise ValueError("cannot get the _status of an unsubmitted job")

    async def wait(self):  # pylint: disable=no-self-use
        raise ValueError("cannot wait on an unsubmitted job")

    async def log(self):  # pylint: disable=no-self-use
        raise ValueError("cannot get the log of an unsubmitted job")

    async def attempts(self):  # pylint: disable=no-self-use
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


class BatchSubmissionInfo:
    def __init__(self, used_fast_create: Optional[bool] = None, used_fast_update: Dict[int, bool] = None):
        self.used_fast_create = used_fast_create
        self.used_fast_update = used_fast_update or {}


class Batch:
    def __init__(self,
                 client: 'BatchClient',
                 id: int,
                 attributes: Optional[Dict[str, str]],
                 token: str,
                 *,
                 last_known_status: bool = None,
                 current_update_id: Optional[int] = None,
                 n_jobs_in_current_update: int = 0,
                 submission_info: Optional[BatchSubmissionInfo] = None):
        self._client = client
        self.id: int = id
        self.attributes: Dict[str, str] = attributes or {}
        self.token = token
        self._current_update_id = current_update_id
        self._n_jobs_in_current_update = n_jobs_in_current_update
        self._last_known_status = last_known_status
        self.submission_info = submission_info or BatchSubmissionInfo()

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

    async def get_job(self, job_id: int) -> Job:
        return await self._client.get_job(self.id, job_id)

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

    # TODO I think there's a decision here for what the semantics should be
    # in a world of multiple concurrent batch updates.
    # Should this be wait on the whole batch to reach completion or just
    # the most recently-submitted update?
    async def wait(self, *, disable_progress_bar=TqdmDisableOption.default):
        i = 0
        with tqdm(total=self._n_jobs_in_current_update, disable=disable_progress_bar, desc=f'Batch {self.id}: completed jobs') as pbar:
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

    async def debug_info(self):
        batch_status = await self.status()
        jobs = []
        async for j_status in self.jobs():
            id = j_status['job_id']
            log, job = await asyncio.gather(self.get_job_log(id), self.get_job(id))
            jobs.append({'log': log, 'status': job._status})
        return {'status': batch_status, 'jobs': jobs}

    async def delete(self):
        await self._client._delete(f'/api/v1alpha/batches/{self.id}')


class BatchBuilder:
    def __init__(self, client, *, attributes=None, callback=None, token=None, cancel_after_n_failures=None,
                 batch=None, update_token=None):
        self._client = client
        self._job_idx = 0
        self._unsubmitted_jobs: List[Job] = []
        self._submitted = False
        self.attributes = attributes
        self.callback = callback
        self._batch = batch

        if token is None:
            token = secret_alnum_string(32)
        self.token = token

        self._update_token = update_token

        self._cancel_after_n_failures = cancel_after_n_failures

    def create_job(self, image: str, command: List[str], *, mount_docker_socket: bool = False, **kwargs) -> Job:
        return self._create_job(
            {'command': command, 'image': image, 'mount_docker_socket': mount_docker_socket, 'type': 'docker'}, **kwargs
        )

    def create_jvm_job(self, jar_spec: Dict[str, str], argv: List[str], **kwargs) -> Job:
        return self._create_job({'type': 'jvm', 'jar_spec': jar_spec, 'command': argv}, **kwargs)

    def _create_job(self,
                    process: dict,
                    *,
                    env: Optional[Dict[str, str]] = None,
                    port: Optional[int] = None,
                    resources: Optional[dict] = None,
                    secrets: Optional[dict] = None,
                    service_account: Optional[str] = None,
                    attributes: Optional[Dict[str, str]] = None,
                    parents: Optional[List[Job]] = None,
                    input_files: Optional[List[Tuple[str, str]]] = None,
                    output_files: Optional[List[Tuple[str, str]]] = None,
                    always_run: bool = False,
                    timeout: Optional[Union[int, float]] = None,
                    cloudfuse: Optional[List[Tuple[str, str, bool]]] = None,
                    requester_pays_project: Optional[str] = None,
                    mount_tokens: bool = False,
                    network: Optional[str] = None,
                    unconfined: bool = False,
                    user_code: Optional[str] = None) -> Job:
        rel_job_id = len(self._unsubmitted_jobs) + 1

        parents = parents or []

        foreign_jobs = []
        invalid_job_ids = []
        parents = parents or []
        for parent in parents:
            job = parent._job
            if isinstance(job, UnsubmittedJob):
                if job._batch_builder != self:
                    foreign_jobs.append(job)
                elif not 0 < job._spec.relative_job_id <= len(self._unsubmitted_jobs):
                    invalid_jobs.append(job._spec.relative_job_id)
            else:
                if self._batch is None or job._batch != self._batch:
                    foreign_jobs.append(job)

        error_msg = []
        if len(foreign_jobs) != 0:
            error_msg.append(
                'Found {} parents from another batch:\n{}'.format(
                    str(len(foreign_jobs)), "\n".join([str(j) for j in foreign_jobs])
                )
            )
        if len(invalid_job_ids) != 0:
            error_msg.append(
                'Found {} parents with invalid job ids:\n{}'.format(
                    str(len(invalid_job_ids)), "\n".join([str(j) for j in invalid_job_ids])
                )
            )
        if error_msg:
            raise ValueError("\n".join(error_msg))

        spec = JobSpec(
            rel_job_id, process, env=env, port=port, resources=resources, secrets=secrets, service_account=service_account,
            attributes=attributes, parents=parents, input_files=input_files, output_files=output_files,
            always_run=always_run, timeout=timeout, cloudfuse=cloudfuse, requester_pays_project=requester_pays_project,
            mount_tokens=mount_tokens, network=network, unconfined=unconfined, user_code=user_code
        )
        j = Job.unsubmitted_job(self, spec)
        self._unsubmitted_jobs.append(j)
        return j

    async def _create_fast(self, byte_job_specs: List[bytes], pbar) -> Batch:
        n_jobs = len(self._unsubmitted_jobs)
        b = bytearray()
        b.extend(b'{"bunch":')
        b.append(ord('['))
        for i, spec in enumerate(byte_job_specs):
            if i > 0:
                b.append(ord(','))
            b.extend(spec)
        b.append(ord(']'))
        b.extend(b',"batch":')
        b.extend(json.dumps(self._create_batch_spec()).encode('utf-8'))
        b.append(ord('}'))
        resp = await self._client._post(
            '/api/v1alpha/batches/create-fast',
            data=aiohttp.BytesPayload(b, content_type='application/json', encoding='utf-8'),
        )
        batch_json = await resp.json()
        pbar.update(n_jobs)
        batch = Batch(self._client,
                      batch_json['id'],
                      self.attributes,
                      self.token,
                      current_update_id=1,
                      n_jobs_in_current_update=n_jobs,
                      submission_info=BatchSubmissionInfo(used_fast_create=True))
        return batch

    async def _submit_jobs(self, batch_id: int, update_id: int, byte_job_specs: List[bytes], n_jobs: int, pbar):
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
            f'/api/v1alpha/batches/{batch_id}/updates/{update_id}/jobs/create',
            data=aiohttp.BytesPayload(b, content_type='application/json', encoding='utf-8'),
        )
        pbar.update(n_jobs)

    async def _update_fast(self, byte_job_specs: List[bytes], pbar) -> int:
        assert self._batch
        b = bytearray()
        b.extend(b'{"bunch":')
        b.append(ord('['))
        for i, spec in enumerate(byte_job_specs):
            if i > 0:
                b.append(ord(','))
            b.extend(spec)
        b.append(ord(']'))
        b.extend(b',"update":')
        b.extend(json.dumps(self._update_batch_spec()).encode('utf-8'))
        b.append(ord('}'))
        resp = await self._client._post(
            f'/api/v1alpha/batches/{self._batch.id}/update-fast',
            data=aiohttp.BytesPayload(b, content_type='application/json', encoding='utf-8'),
        )
        update_json = await resp.json()
        pbar.update(len(byte_job_specs))
        return int(update_json['start_job_id'])

    def _create_batch_spec(self):
        n_jobs = len(self._unsubmitted_jobs)
        batch_spec = {'billing_project': self._client.billing_project, 'n_jobs': n_jobs, 'token': self.token}
        if self.attributes:
            batch_spec['attributes'] = self.attributes
        if self.callback:
            batch_spec['callback'] = self.callback
        if self._cancel_after_n_failures is not None:
            batch_spec['cancel_after_n_failures'] = self._cancel_after_n_failures
        return batch_spec

    def _update_batch_spec(self):
        n_jobs = len(self._unsubmitted_jobs)
        return {'n_jobs': n_jobs, 'token': self._update_token}

    async def _create_batch(self) -> Batch:
        batch_spec = self._create_batch_spec()
        batch_json = await (await self._client._post('/api/v1alpha/batches/create', json=batch_spec)).json()
        b = Batch(
            self._client,
            batch_json['id'],
            self.attributes,
            self.token,
            current_update_id=batch_json['update_id'],
            n_jobs_in_current_update=batch_spec['n_jobs'],
            submission_info=BatchSubmissionInfo(used_fast_create=False)
        )
        return b

    async def _create_update(self, batch_id: int) -> int:
        update_spec = self._update_batch_spec()
        resp = await self._client._post(f'/api/v1alpha/batches/{batch_id}/updates/create', json=update_spec)
        update_json = await resp.json()
        return update_json['update_id']

    async def _commit_update(self, batch_id: int, update_id: int) -> int:
        resp = await self._client._patch(f'/api/v1alpha/batches/{batch_id}/updates/{update_id}/commit')
        commit_json = await resp.json()
        return commit_json['start_job_id']

    MAX_BUNCH_BYTESIZE = 1024 * 1024
    MAX_BUNCH_SIZE = 1024

    async def submit(self,
                     max_bunch_bytesize: int = MAX_BUNCH_BYTESIZE,
                     max_bunch_size: int = MAX_BUNCH_SIZE,
                     disable_progress_bar: Union[bool, None, TqdmDisableOption] = TqdmDisableOption.default
                     ) -> Batch:
        assert max_bunch_bytesize > 0
        assert max_bunch_size > 0
        if self._submitted:
            raise ValueError("cannot submit an already submitted batch")
        byte_job_specs = [json.dumps(j._job._spec.to_json()).encode('utf-8')
                          for j in self._unsubmitted_jobs]
        byte_job_specs_bunches: List[List[bytes]] = []
        bunch_sizes = []
        bunch: List[bytes] = []
        bunch_n_bytes = 0
        bunch_n_jobs = 0
        for spec in byte_job_specs:
            n_bytes = len(spec)
            assert n_bytes < max_bunch_bytesize, (
                'every job spec must be less than max_bunch_bytesize,'
                f' { max_bunch_bytesize }B, but {spec.decode()} is larger')
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

        with tqdm(total=len(self._unsubmitted_jobs),
                  disable=disable_progress_bar,
                  desc='jobs submitted to queue') as pbar:
            n_bunches = len(byte_job_specs_bunches)

            if self._batch is None:
                if n_bunches == 0:
                    self._batch = await self._create_batch()
                    return self._batch
                if n_bunches == 1:
                    self._batch = await self._create_fast(byte_job_specs_bunches[0], pbar)
                    start_job_id = 1
                else:
                    self._batch = await self._create_batch()
                    update_id = self._batch._current_update_id
                    assert update_id is not None
                    await bounded_gather(
                        *[functools.partial(self._submit_jobs, self._batch.id, update_id, bunch, size, pbar)
                          for bunch, size in zip(byte_job_specs_bunches, bunch_sizes)
                          ],
                        parallelism=6,
                    )
                    start_job_id = await self._commit_update(self._batch.id, update_id)
                    self._batch.submission_info.used_fast_update[update_id] = False
                    assert start_job_id == 1
            else:
                assert self._batch is not None
                if n_bunches == 0:
                    log.warning('Tried to submit an update with 0 jobs. Doing nothing.')
                    return self._batch
                if self._update_token is None:
                    self._update_token = secret_alnum_string(32)
                if n_bunches == 1:
                    start_job_id = await self._update_fast(byte_job_specs_bunches[0], pbar)
                else:
                    update_id = await self._create_update(self._batch.id)
                    assert update_id is not None
                    await bounded_gather(
                        *[functools.partial(self._submit_jobs, self._batch.id, update_id, bunch, size, pbar)
                          for bunch, size in zip(byte_job_specs_bunches, bunch_sizes)
                          ],
                        parallelism=6,
                    )
                    start_job_id = await self._commit_update(self._batch.id, update_id)
                    self._batch.submission_info.used_fast_update[update_id] = False
                self._update_token = None

        for j in self._unsubmitted_jobs:
            j._job = j._job._submit(self._batch, j._spec.relative_job_id + start_job_id - 1)

        self._unsubmitted_jobs = []
        self._job_idx = 0

        self._submitted = True
        return self._batch


class BatchClient:
    @staticmethod
    async def create(billing_project: str,
                     deploy_config: Optional[DeployConfig] = None,
                     session: Optional[httpx.ClientSession] = None,
                     headers: Optional[Dict[str, str]] = None,
                     _token: Optional[str] = None,
                     token_file: Optional[str] = None):
        if not deploy_config:
            deploy_config = get_deploy_config()
        url = deploy_config.base_url('batch')
        if session is None:
            session = httpx.client_session()
        if headers is None:
            headers = {}
        if _token:
            headers['Authorization'] = f'Bearer {_token}'
        else:
            headers.update(service_auth_headers(deploy_config, 'batch', token_file=token_file))
        return BatchClient(
            billing_project=billing_project,
            url=url,
            session=session,
            headers=headers)

    def __init__(self,
                 billing_project: str,
                 url: str,
                 session: httpx.ClientSession,
                 headers: Dict[str, str]):
        self.billing_project = billing_project
        self.url = url
        self._session = session
        self._headers = headers

    async def _get(self, path, params=None):
        return await request_retry_transient_errors(
            self._session, 'GET', self.url + path, params=params, headers=self._headers
        )

    async def _post(self, path, data=None, json=None):
        return await request_retry_transient_errors(
            self._session, 'POST', self.url + path, data=data, json=json, headers=self._headers
        )

    async def _patch(self, path):
        return await request_retry_transient_errors(self._session, 'PATCH', self.url + path, headers=self._headers)

    async def _delete(self, path):
        return await request_retry_transient_errors(self._session, 'DELETE', self.url + path, headers=self._headers)

    async def list_batches(self, q=None, last_batch_id=None, limit=2 ** 64):
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
                yield Batch(
                    self,
                    batch['id'],
                    attributes=batch.get('attributes'),
                    token=batch['token'],
                    last_known_status=batch,
                )
            last_batch_id = body.get('last_batch_id')
            if last_batch_id is None:
                break

    async def get_job(self, batch_id, job_id):
        b = await self.get_batch(batch_id)
        j_resp = await self._get(f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}')
        j = await j_resp.json()
        return Job.submitted_job(b, j['job_id'], _status=j)

    async def get_job_log(self, batch_id, job_id) -> Optional[Dict[str, Any]]:
        resp = await self._get(f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
        return await resp.json()

    async def get_job_attempts(self, batch_id, job_id):
        resp = await self._get(f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/attempts')
        return await resp.json()

    async def get_batch(self, batch_id):
        b_resp = await self._get(f'/api/v1alpha/batches/{batch_id}')
        b = await b_resp.json()
        return Batch(self,
                     b['id'],
                     attributes=b.get('attributes'),
                     token=b['token'],
                     last_known_status=b)

    def create_batch(self, attributes=None, callback=None, token=None, cancel_after_n_failures=None) -> BatchBuilder:
        return BatchBuilder(self, attributes=attributes, callback=callback, token=token, cancel_after_n_failures=cancel_after_n_failures)

    async def update_batch(self, batch_id: int, update_token=None) -> BatchBuilder:
        batch = await self.get_batch(batch_id)
        return BatchBuilder(self, batch=batch, update_token=update_token)

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
        bp_resp = await self._post(f'/api/v1alpha/billing_limits/{project}/edit', json={'limit': limit})
        return await bp_resp.json()

    async def close(self):
        await self._session.close()
        self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
