from typing import Optional, Dict, Any, List, Tuple, Union, AsyncIterator, TypedDict, cast
import math
import random
import logging
import json
import functools
import asyncio
import aiohttp
import orjson
import secrets

from hailtop import is_notebook
from hailtop.config import get_deploy_config, DeployConfig
from hailtop.aiocloud.common import Session
from hailtop.aiocloud.common.credentials import CloudCredentials
from hailtop.auth import hail_credentials
from hailtop.utils import bounded_gather, sleep_before_try
from hailtop.utils.rich_progress_bar import BatchProgressBar, BatchProgressBarTask
from hailtop import httpx

from .types import GetJobsResponseV1Alpha, JobListEntryV1Alpha, GetJobResponseV1Alpha
from .globals import tasks, complete_states

log = logging.getLogger('batch_client.aioclient')


class JobAlreadySubmittedError(Exception):
    pass


class JobNotSubmittedError(Exception):
    pass


class AbsoluteJobId(int):
    pass


class InUpdateJobId(int):
    pass


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

        durations = [_get_duration(container_status) for container_status in container_statuses.values()]

        if any(d is None for d in durations):
            return None
        return sum(durations)  # type: ignore

    @staticmethod
    def submitted_job(batch: 'Batch', job_id: int, _status: Optional[GetJobResponseV1Alpha] = None):
        return Job(batch, AbsoluteJobId(job_id), _status=_status)

    @staticmethod
    def unsubmitted_job(batch: 'Batch', job_id: int):
        return Job(batch, InUpdateJobId(job_id))

    def __init__(self,
                 batch: 'Batch',
                 job_id: Union[AbsoluteJobId, InUpdateJobId],
                 *,
                 _status: Optional[GetJobResponseV1Alpha] = None):
        self._batch = batch
        self._job_id = job_id
        self._status = _status

    def _raise_if_not_submitted(self):
        if not self.is_submitted:
            raise JobNotSubmittedError

    def _raise_if_submitted(self):
        if self.is_submitted:
            raise JobAlreadySubmittedError

    def _submit(self, in_update_start_job_id: int):
        self._raise_if_submitted()
        self._job_id = AbsoluteJobId(in_update_start_job_id + self._job_id - 1)

    @property
    def is_submitted(self):
        return isinstance(self._job_id, AbsoluteJobId)

    @property
    def batch_id(self) -> int:
        return self._batch.id

    @property
    def job_id(self) -> int:
        self._raise_if_not_submitted()
        return self._job_id

    @property
    def id(self) -> Tuple[int, int]:
        self._raise_if_not_submitted()
        return (self.batch_id, self.job_id)

    @property
    def _client(self) -> 'BatchClient':
        return self._batch._client

    async def attributes(self):
        if not self._status:
            await self.status()
        assert self._status is not None
        if 'attributes' in self._status:
            return self._status['attributes']
        return {}

    async def _is_job_in_state(self, states):
        await self.status()
        assert self._status is not None
        state = self._status['state']
        return state in states

    async def is_complete(self):
        return await self._is_job_in_state(complete_states)

    async def is_running(self):
        return await self._is_job_in_state(['Running'])

    async def is_pending(self):
        return await self._is_job_in_state(['Pending'])

    async def is_ready(self):
        return await self._is_job_in_state(['Ready'])

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
    async def status(self) -> Dict[str, Any]:
        self._raise_if_not_submitted()
        resp = await self._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}')
        self._status = await resp.json()
        assert self._status is not None
        return self._status

    async def wait(self) -> Dict[str, Any]:
        return cast(
            Dict[str, Any], # https://stackoverflow.com/a/76515675/6823256
            await self._wait_for_states(*complete_states)
        )

    async def _wait_for_states(self, *states: str) -> GetJobResponseV1Alpha:
        tries = 0
        while True:
            if await self._is_job_in_state(states) or await self.is_complete():
                assert self._status
                return self._status
            tries += 1
            await sleep_before_try(tries)

    async def container_log(self, container_name: str) -> bytes:
        self._raise_if_not_submitted()
        async with await self._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/log/{container_name}') as resp:
            return await resp.read()

    async def log(self):
        self._raise_if_not_submitted()
        resp = await self._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/log')
        return await resp.json()

    async def attempts(self):
        self._raise_if_not_submitted()
        resp = await self._client._get(f'/api/v1alpha/batches/{self.batch_id}/jobs/{self.job_id}/attempts')
        return await resp.json()


class BatchSubmissionInfo:
    def __init__(self, used_fast_path: Optional[bool] = None):
        self.used_fast_path = used_fast_path


class BatchNotCreatedError(Exception):
    pass


class BatchAlreadyCreatedError(Exception):
    pass


class BatchDebugInfo(TypedDict):
    status: Dict[str, Any]
    jobs: List[JobListEntryV1Alpha]

class Batch:
    def __init__(self,
                 client: 'BatchClient',
                 id: Optional[int],
                 *,
                 attributes: Optional[Dict[str, str]] = None,
                 callback: Optional[str] = None,
                 token: Optional[str] = None,
                 cancel_after_n_failures: Optional[int] = None,
                 last_known_status: Optional[Dict[str, Any]] = None):
        self._client = client
        self._id = id
        self.attributes: Dict[str, str] = attributes or {}
        self._callback = callback

        if token is None:
            token = secrets.token_urlsafe(32)
        self.token = token

        self._cancel_after_n_failures = cancel_after_n_failures
        self._submission_info = BatchSubmissionInfo()
        self._last_known_status = last_known_status

        self._job_idx = 0
        self._job_specs: List[Dict[str, Any]] = []
        self._jobs: List[Job] = []

    def _raise_if_not_created(self):
        if not self.is_created:
            raise BatchNotCreatedError

    def _raise_if_created(self):
        if self.is_created:
            raise BatchAlreadyCreatedError

    @property
    def id(self) -> int:
        self._raise_if_not_created()
        assert self._id
        return self._id

    @property
    def is_created(self):
        return self._id is not None

    async def cancel(self):
        self._raise_if_not_created()
        await self._client._patch(f'/api/v1alpha/batches/{self.id}/cancel')

    async def jobs(self,
                   q: Optional[str] = None,
                   version: Optional[int] = None
                   ) -> AsyncIterator[JobListEntryV1Alpha]:
        self._raise_if_not_created()
        if version is None:
            version = 1
        last_job_id = None
        while True:
            params = {}
            if q is not None:
                params['q'] = q
            if last_job_id is not None:
                params['last_job_id'] = last_job_id
            resp = await self._client._get(f'/api/v{version}alpha/batches/{self.id}/jobs', params=params)
            body = cast(
                GetJobsResponseV1Alpha,
                await resp.json()
            )
            for job in body['jobs']:
                yield job
            last_job_id = body.get('last_job_id')
            if last_job_id is None:
                break

    async def get_job(self, job_id: int) -> Job:
        self._raise_if_not_created()
        return await self._client.get_job(self.id, job_id)

    async def get_job_log(self, job_id: int) -> Dict[str, Any]:
        self._raise_if_not_created()
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
    async def status(self) -> Dict[str, Any]:
        self._raise_if_not_created()
        resp = await self._client._get(f'/api/v1alpha/batches/{self.id}')
        json_status = await resp.json()
        assert isinstance(json_status, dict), json_status
        self._last_known_status = json_status
        return self._last_known_status

    async def last_known_status(self) -> Dict[str, Any]:
        self._raise_if_not_created()
        if self._last_known_status is None:
            return await self.status()  # updates _last_known_status
        return self._last_known_status

    async def _wait(self,
                    description: str,
                    progress: BatchProgressBar,
                    disable_progress_bar: bool,
                    starting_job: int
                    ) -> Dict[str, Any]:
        self._raise_if_not_created()
        deploy_config = get_deploy_config()
        url = deploy_config.external_url('batch', f'/batches/{self.id}')
        i = 0
        status = await self.status()
        if is_notebook():
            description += f'[link={url}]{self.id}[/link]'
        else:
            description += url
        with progress.with_task(description,
                                total=status['n_jobs'] - starting_job + 1,
                                disable=disable_progress_bar) as progress_task:
            while True:
                status = await self.status()
                progress_task.update(None, total=status['n_jobs'] - starting_job + 1, completed=status['n_completed'] - starting_job + 1)
                if status['complete']:
                    return status
                j = random.randrange(math.floor(1.1 ** i))
                await asyncio.sleep(0.100 * j)
                # max 44.5s
                if i < 64:
                    i = i + 1

    # FIXME Error if this is called while within a job of the same Batch
    async def wait(self,
                   *,
                   disable_progress_bar: bool = False,
                   description: str = '',
                   progress: Optional[BatchProgressBar] = None,
                   starting_job: int = 1,
                   ) -> Dict[str, Any]:
        self._raise_if_not_created()
        if description:
            description += ': '
        if progress is not None:
            return await self._wait(description, progress, disable_progress_bar, starting_job)
        with BatchProgressBar(disable=disable_progress_bar) as progress2:
            return await self._wait(description, progress2, disable_progress_bar, starting_job)

    async def debug_info(self,
                         _jobs_query_string: Optional[str] = None,
                         _max_jobs: Optional[int] = None,
                         ) -> BatchDebugInfo:
        self._raise_if_not_created()
        batch_status = await self.status()
        jobs = []
        async for j_status in self.jobs(q=_jobs_query_string):
            if _max_jobs and len(jobs) == _max_jobs:
                break

            id = j_status['job_id']
            log, job = await asyncio.gather(self.get_job_log(id), self.get_job(id))
            jobs.append({'log': log, 'status': job._status})
        return {'status': batch_status, 'jobs': jobs}

    async def delete(self):
        self._raise_if_not_created()
        try:
            await self._client._delete(f'/api/v1alpha/batches/{self.id}')
        except httpx.ClientResponseError as err:
            if err.code != 404:
                raise

    def create_job(self, image: str, command: List[str], **kwargs):
        return self._create_job(
            {'command': command, 'image': image, 'type': 'docker'}, **kwargs
        )

    def create_jvm_job(self, jar_spec: Dict[str, str], argv: List[str], *, profile: bool = False, **kwargs):
        if 'always_copy_output' in kwargs:
            raise ValueError("the 'always_copy_output' option is not allowed for JVM jobs")
        return self._create_job({'type': 'jvm', 'jar_spec': jar_spec, 'command': argv, 'profile': profile}, **kwargs)

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
                    always_copy_output: bool = False,
                    timeout: Optional[Union[int, float]] = None,
                    cloudfuse: Optional[List[Tuple[str, str, bool]]] = None,
                    requester_pays_project: Optional[str] = None,
                    mount_tokens: bool = False,
                    network: Optional[str] = None,
                    unconfined: bool = False,
                    user_code: Optional[str] = None,
                    regions: Optional[List[str]] = None):
        self._job_idx += 1

        if parents is None:
            parents = []

        absolute_parent_ids = []
        in_update_parent_ids = []
        foreign_batches: List[Job] = []
        invalid_job_ids = []
        for parent in parents:
            if not parent.is_submitted:
                assert isinstance(parent._job_id, InUpdateJobId)
                if parent._batch != self:
                    foreign_batches.append(parent)
                elif not 0 < parent._job_id < self._job_idx:
                    invalid_job_ids.append(parent._job_id)
                else:
                    in_update_parent_ids.append(parent._job_id)
            elif not self.is_created or parent._batch.id != self.id:
                foreign_batches.append(parent)
            else:
                absolute_parent_ids.append(parent._job_id)

        error_msg = []
        if len(foreign_batches) != 0:
            error_msg.append(
                'Found {} parents from another batch:\n{}'.format(
                    str(len(foreign_batches)), "\n".join([str(j) for j in foreign_batches])
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

        job_spec = {
            'always_run': always_run,
            'always_copy_output': always_copy_output,
            'job_id': self._job_idx,
            'absolute_parent_ids': absolute_parent_ids,
            'in_update_parent_ids': in_update_parent_ids,
            'process': process,
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
        if cloudfuse:
            job_spec['cloudfuse'] = [{"bucket": bucket, "mount_path": mount_path, "read_only": read_only}
                                     for (bucket, mount_path, read_only) in cloudfuse]
        if requester_pays_project:
            job_spec['requester_pays_project'] = requester_pays_project
        if mount_tokens:
            job_spec['mount_tokens'] = mount_tokens
        if network:
            job_spec['network'] = network
        if unconfined:
            job_spec['unconfined'] = unconfined
        if user_code:
            job_spec['user_code'] = user_code
        if regions:
            job_spec['regions'] = regions

        self._job_specs.append(job_spec)

        j = Job.unsubmitted_job(self, self._job_idx)
        self._jobs.append(j)
        return j

    async def _create_fast(self, byte_job_specs: List[bytes], n_jobs: int, job_progress_task: BatchProgressBarTask):
        self._raise_if_created()
        assert n_jobs == len(self._job_specs)
        b = bytearray()
        b.extend(b'{"bunch":')
        b.append(ord('['))
        for i, spec in enumerate(byte_job_specs):
            if i > 0:
                b.append(ord(','))
            b.extend(spec)
        b.append(ord(']'))
        b.extend(b',"batch":')
        b.extend(json.dumps(self._batch_spec()).encode('utf-8'))
        b.append(ord('}'))
        resp = await self._client._post(
            '/api/v1alpha/batches/create-fast',
            data=aiohttp.BytesPayload(b, content_type='application/json', encoding='utf-8'),
        )
        batch_json = await resp.json()
        job_progress_task.update(n_jobs)

        self._id = batch_json['id']
        self._submission_info = BatchSubmissionInfo(used_fast_path=True)

    async def _update_fast(self, byte_job_specs: List[bytes], n_jobs: int, job_progress_task: BatchProgressBarTask) -> int:
        self._raise_if_not_created()
        assert n_jobs == len(self._job_specs)
        b = bytearray()
        b.extend(b'{"bunch":')
        b.append(ord('['))
        for i, spec in enumerate(byte_job_specs):
            if i > 0:
                b.append(ord(','))
            b.extend(spec)
        b.append(ord(']'))
        b.extend(b',"update":')
        b.extend(json.dumps(self._update_spec()).encode('utf-8'))
        b.append(ord('}'))
        resp = await self._client._post(
            f'/api/v1alpha/batches/{self.id}/update-fast',
            data=aiohttp.BytesPayload(b, content_type='application/json', encoding='utf-8'),
        )
        update_json = await resp.json()
        job_progress_task.update(n_jobs)
        self._submission_info = BatchSubmissionInfo(used_fast_path=True)
        return int(update_json['start_job_id'])

    def _create_bunches(self,
                        specs: List[dict],
                        max_bunch_bytesize: int,
                        max_bunch_size: int,
                        ) -> Tuple[List[List[bytes]], List[int]]:
        assert max_bunch_bytesize > 0
        assert max_bunch_size > 0
        byte_specs = [orjson.dumps(spec) for spec in specs]
        byte_specs_bunches: List[List[bytes]] = []
        bunch_sizes = []
        bunch: List[bytes] = []
        bunch_n_bytes = 0
        bunch_n_jobs = 0
        for spec in byte_specs:
            n_bytes = len(spec)
            assert n_bytes < max_bunch_bytesize, (
                'every spec must be less than max_bunch_bytesize,'
                f' { max_bunch_bytesize }B, but {spec.decode()} is larger')
            if bunch_n_bytes + n_bytes < max_bunch_bytesize and len(bunch) < max_bunch_size:
                bunch.append(spec)
                bunch_n_bytes += n_bytes
                bunch_n_jobs += 1
            else:
                byte_specs_bunches.append(bunch)
                bunch_sizes.append(bunch_n_jobs)
                bunch = [spec]
                bunch_n_bytes = n_bytes
                bunch_n_jobs = 1
        if bunch:
            byte_specs_bunches.append(bunch)
            bunch_sizes.append(bunch_n_jobs)

        return (byte_specs_bunches, bunch_sizes)

    async def _submit_jobs(self, update_id: int, byte_job_specs: List[bytes], n_jobs: int, progress_task: BatchProgressBarTask):
        self._raise_if_not_created()
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
            f'/api/v1alpha/batches/{self.id}/updates/{update_id}/jobs/create',
            data=aiohttp.BytesPayload(b, content_type='application/json', encoding='utf-8'),
        )
        progress_task.update(n_jobs)

    def _batch_spec(self):
        n_jobs = len(self._job_specs)
        batch_spec = {'billing_project': self._client.billing_project, 'n_jobs': n_jobs, 'token': self.token}
        if self.attributes:
            batch_spec['attributes'] = self.attributes
        if self._callback:
            batch_spec['callback'] = self._callback
        if self._cancel_after_n_failures is not None:
            batch_spec['cancel_after_n_failures'] = self._cancel_after_n_failures
        return batch_spec

    async def _open_batch(self) -> Optional[int]:
        self._raise_if_created()
        batch_spec = self._batch_spec()
        batch_json = await (await self._client._post('/api/v1alpha/batches/create', json=batch_spec)).json()
        self._id = batch_json['id']
        update_id = batch_json['update_id']
        if update_id is None:
            assert batch_spec['n_jobs'] == 0
        return update_id

    def _update_spec(self) -> dict:
        update_token = secrets.token_urlsafe(32)
        return {'n_jobs': len(self._jobs), 'token': update_token}

    async def _create_update(self) -> int:
        self._raise_if_not_created()
        update_spec = self._update_spec()
        update_json = await (await self._client._post(f'/api/v1alpha/batches/{self.id}/updates/create', json=update_spec)).json()
        return int(update_json['update_id'])

    async def _commit_update(self, update_id: int) -> int:
        self._raise_if_not_created()
        commit_json = await (await self._client._patch(f'/api/v1alpha/batches/{self.id}/updates/{update_id}/commit')).json()
        return int(commit_json['start_job_id'])

    async def _submit_job_bunches(self,
                                  update_id: int,
                                  byte_job_specs_bunches: List[List[bytes]],
                                  bunch_sizes: List[int],
                                  progress_task: BatchProgressBarTask):
        self._raise_if_not_created()
        await bounded_gather(
            *[functools.partial(self._submit_jobs, update_id, bunch, size, progress_task)
              for bunch, size in zip(byte_job_specs_bunches, bunch_sizes)
              ],
            parallelism=6,
            cancel_on_error=True,
        )

    async def _submit(self,
                      max_bunch_bytesize: int,
                      max_bunch_size: int,
                      disable_progress_bar: bool,
                      progress: BatchProgressBar) -> Optional[int]:
        n_jobs = len(self._jobs)
        byte_job_specs_bunches, job_bunch_sizes = self._create_bunches(self._job_specs, max_bunch_bytesize, max_bunch_size)
        n_job_bunches = len(byte_job_specs_bunches)

        with progress.with_task('submit job bunches', total=n_jobs, disable=(disable_progress_bar or n_job_bunches < 100)) as job_progress_task:
            if not self.is_created:
                if n_job_bunches == 0:
                    await self._open_batch()
                    log.info(f'created batch {self.id}')
                    return None
                if n_job_bunches == 1:
                    await self._create_fast(byte_job_specs_bunches[0], job_bunch_sizes[0], job_progress_task)
                    start_job_id = 1
                else:
                    update_id = await self._open_batch()
                    assert update_id is not None
                    await self._submit_job_bunches(update_id, byte_job_specs_bunches, job_bunch_sizes, job_progress_task)
                    start_job_id = await self._commit_update(update_id)
                    self._submission_info = BatchSubmissionInfo(used_fast_path=False)
                    assert start_job_id == 1
                log.info(f'created batch {self.id}')
            else:
                if n_job_bunches == 0:
                    log.warning('Tried to submit an update with 0 jobs. Doing nothing.')
                    return None
                if n_job_bunches == 1:
                    start_job_id = await self._update_fast(byte_job_specs_bunches[0], job_bunch_sizes[0], job_progress_task)
                else:
                    update_id = await self._create_update()
                    await self._submit_job_bunches(update_id, byte_job_specs_bunches, job_bunch_sizes, job_progress_task)
                    start_job_id = await self._commit_update(update_id)
                    self._submission_info = BatchSubmissionInfo(used_fast_path=False)
                log.info(f'updated batch {self.id}')
            return start_job_id

    MAX_BUNCH_BYTESIZE = 1024 * 1024
    MAX_BUNCH_SIZE = 1024

    async def submit(self,
                     max_bunch_bytesize: int = MAX_BUNCH_BYTESIZE,
                     max_bunch_size: int = MAX_BUNCH_SIZE,
                     disable_progress_bar: bool = False,
                     *,
                     progress: Optional[BatchProgressBar] = None
                     ):
        assert max_bunch_bytesize > 0
        assert max_bunch_size > 0

        if progress:
            start_job_id = await self._submit(max_bunch_bytesize, max_bunch_size, disable_progress_bar, progress)
        else:
            with BatchProgressBar(disable=disable_progress_bar) as progress2:
                start_job_id = await self._submit(max_bunch_bytesize, max_bunch_size, disable_progress_bar, progress2)

        assert self.is_created

        for j in self._jobs:
            assert start_job_id is not None
            j._submit(start_job_id)

        self._job_specs = []
        self._jobs = []
        self._job_idx = 0


class HailExplicitTokenCredentials(CloudCredentials):
    def __init__(self, token: str):
        self._token = token

    async def auth_headers(self) -> Dict[str, str]:
        return {'Authorization': f'Bearer {self._token}'}

    async def access_token(self) -> str:
        return self._token

    async def close(self):
        pass


class BatchClient:
    @staticmethod
    async def create(billing_project: str,
                     deploy_config: Optional[DeployConfig] = None,
                     session: Optional[httpx.ClientSession] = None,
                     headers: Optional[Dict[str, str]] = None,
                     _token: Optional[str] = None,
                     token_file: Optional[str] = None,
                     *,
                     cloud_credentials_file: Optional[str] = None):
        if not deploy_config:
            deploy_config = get_deploy_config()
        url = deploy_config.base_url('batch')
        if headers is None:
            headers = {}
        credentials: CloudCredentials
        if _token is not None:
            credentials = HailExplicitTokenCredentials(_token)
        else:
            credentials = hail_credentials(tokens_file=token_file, cloud_credentials_file=cloud_credentials_file)
        return BatchClient(
            billing_project=billing_project,
            url=url,
            session=Session(credentials=credentials, http_session=session, timeout=aiohttp.ClientTimeout(total=30)),
            headers=headers)

    def __init__(self,
                 billing_project: str,
                 url: str,
                 session: Session,
                 headers: Dict[str, str]):
        self.billing_project = billing_project
        self.url = url
        self._session: Session = session
        self._headers = headers

    async def _get(self, path, params=None) -> aiohttp.ClientResponse:
        return await self._session.get(self.url + path, params=params, headers=self._headers)

    async def _post(self, path, data=None, json=None) -> aiohttp.ClientResponse:
        return await self._session.post(self.url + path, data=data, json=json, headers=self._headers)

    async def _patch(self, path) -> aiohttp.ClientResponse:
        return await self._session.patch(self.url + path, headers=self._headers)

    async def _delete(self, path) -> aiohttp.ClientResponse:
        return await self._session.delete(self.url + path, headers=self._headers)

    def reset_billing_project(self, billing_project):
        self.billing_project = billing_project

    async def list_batches(self, q=None, last_batch_id=None, limit=2 ** 64, version=None):
        if version is None:
            version = 1
        n = 0
        while True:
            params = {}
            if q is not None:
                params['q'] = q
            if last_batch_id is not None:
                params['last_batch_id'] = last_batch_id

            resp = await self._get(f'/api/v{version}alpha/batches', params=params)
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
        j = cast(
            GetJobResponseV1Alpha,
            await j_resp.json()
        )
        return Job.submitted_job(b, j['job_id'], _status=j)

    async def get_job_log(self, batch_id, job_id) -> Dict[str, Any]:
        resp = await self._get(f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/log')
        return await resp.json()

    async def get_job_attempts(self, batch_id, job_id):
        resp = await self._get(f'/api/v1alpha/batches/{batch_id}/jobs/{job_id}/attempts')
        return await resp.json()

    async def get_batch(self, id) -> Batch:
        b_resp = await self._get(f'/api/v1alpha/batches/{id}')
        b = await b_resp.json()
        assert isinstance(b, dict), b
        attributes = b.get('attributes')
        assert attributes is None or isinstance(attributes, dict), attributes
        return Batch(self,
                     id=b['id'],
                     attributes=attributes,
                     token=b['token'],
                     last_known_status=b)

    def create_batch(self, attributes=None, callback=None, token=None, cancel_after_n_failures=None) -> Batch:
        return Batch(self,
                     id=None,
                     attributes=attributes,
                     callback=callback,
                     token=token,
                     cancel_after_n_failures=cancel_after_n_failures)

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

    async def supported_regions(self) -> List[str]:
        resp = await self._get('/api/v1alpha/supported_regions')
        return await resp.json()

    async def cloud(self):
        resp = await self._get('/api/v1alpha/cloud')
        return await resp.text()

    async def close(self):
        await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
