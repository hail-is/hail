from typing import Any, Dict, List, Optional

from hailtop.utils import async_to_blocking, ait_to_blocking
from ..config import DeployConfig
from . import aioclient
from .. import httpx


class Job:
    @staticmethod
    def _get_error(job_status, task):
        return aioclient.Job._get_error(job_status, task)

    @staticmethod
    def _get_out_of_memory(job_status, task):
        return aioclient.Job._get_out_of_memory(job_status, task)

    @staticmethod
    def _get_exit_code(job_status, task):
        return aioclient.Job._get_exit_code(job_status, task)

    @staticmethod
    def _get_exit_codes(job_status):
        return aioclient.Job._get_exit_codes(job_status)

    @staticmethod
    def exit_code(job_status):
        return aioclient.Job.exit_code(job_status)

    @staticmethod
    def total_duration_msecs(job_status):
        return aioclient.Job.total_duration_msecs(job_status)

    def __init__(self, async_job: aioclient.Job):
        self._async_job: aioclient.Job = async_job

    @property
    def _status(self):
        return self._async_job._status

    @property
    def batch_id(self):
        return self._async_job.batch_id

    @property
    def job_id(self):
        return self._async_job.job_id

    @property
    def id(self):
        return self._async_job.id

    def attributes(self):
        return async_to_blocking(self._async_job.attributes())

    def is_complete(self):
        return async_to_blocking(self._async_job.is_complete())

    def is_running(self):
        return async_to_blocking(self._async_job.is_running())

    def is_pending(self):
        return async_to_blocking(self._async_job.is_pending())

    def is_ready(self):
        return async_to_blocking(self._async_job.is_ready())

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
    def status(self):
        return async_to_blocking(self._async_job.status())

    def wait(self):
        return async_to_blocking(self._async_job.wait())

    def _wait_for_states(self, *states: str):
        return async_to_blocking(self._async_job._wait_for_states(*states))

    def container_log(self, container_name):
        return async_to_blocking(self._async_job.container_log(container_name))

    def log(self):
        return async_to_blocking(self._async_job.log())

    def attempts(self):
        return async_to_blocking(self._async_job.attempts())


class Batch:
    @staticmethod
    def _open_batch(client: 'BatchClient', token: Optional[str] = None) -> 'Batch':
        async_batch = client.create_batch(token=token)._async_batch
        async_to_blocking(async_batch._open_batch())
        return Batch(async_batch)

    def __init__(self, async_batch: aioclient.Batch):
        self._async_batch = async_batch

    @property
    def is_created(self) -> bool:
        return self._async_batch.is_created

    @property
    def id(self) -> int:
        return self._async_batch.id

    @property
    def attributes(self):
        return self._async_batch.attributes

    @property
    def token(self):
        return self._async_batch.token

    @property
    def _submission_info(self):
        return self._async_batch._submission_info

    def cancel(self):
        async_to_blocking(self._async_batch.cancel())

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
    def status(self):
        return async_to_blocking(self._async_batch.status())

    def last_known_status(self):
        return async_to_blocking(self._async_batch.last_known_status())

    def jobs(self, q=None, version=None):
        return ait_to_blocking(self._async_batch.jobs(q=q, version=version))

    def get_job(self, job_id: int) -> Job:
        j = async_to_blocking(self._async_batch.get_job(job_id))
        return Job(j)

    def get_job_log(self, job_id: int) -> Dict[str, Any]:
        return async_to_blocking(self._async_batch.get_job_log(job_id))

    def wait(self, *args, **kwargs):
        return async_to_blocking(self._async_batch.wait(*args, **kwargs))

    def debug_info(self):
        return async_to_blocking(self._async_batch.debug_info())

    def delete(self):
        async_to_blocking(self._async_batch.delete())

    def create_job(
        self,
        image,
        command,
        *,
        env=None,
        port=None,
        resources=None,
        secrets=None,
        service_account=None,
        attributes=None,
        parents=None,
        input_files=None,
        output_files=None,
        always_run=False,
        timeout=None,
        cloudfuse=None,
        requester_pays_project=None,
        mount_tokens=False,
        network: Optional[str] = None,
        unconfined: bool = False,
        user_code: Optional[str] = None,
        regions: Optional[List[str]] = None,
        always_copy_output: bool = False
    ) -> Job:
        if parents:
            parents = [parent._async_job for parent in parents]

        async_job = self._async_batch.create_job(
            image,
            command,
            env=env,
            port=port,
            resources=resources,
            secrets=secrets,
            service_account=service_account,
            attributes=attributes,
            parents=parents,
            input_files=input_files,
            output_files=output_files,
            always_run=always_run,
            always_copy_output=always_copy_output,
            timeout=timeout,
            cloudfuse=cloudfuse,
            requester_pays_project=requester_pays_project,
            mount_tokens=mount_tokens,
            network=network,
            unconfined=unconfined,
            user_code=user_code,
            regions=regions,
        )

        return Job(async_job)

    def create_jvm_job(self, command, *, profile: bool = False, parents=None, **kwargs) -> Job:
        if parents:
            parents = [parent._async_job for parent in parents]

        async_job = self._async_batch.create_jvm_job(command, profile=profile, parents=parents, **kwargs)

        return Job(async_job)

    def submit(self, *args, **kwargs):
        async_to_blocking(self._async_batch.submit(*args, **kwargs))


class BatchClient:
    @staticmethod
    def from_async(async_client: aioclient.BatchClient):
        bc = BatchClient.__new__(BatchClient)
        bc._async_client = async_client
        return bc

    def __init__(
        self,
        billing_project: str,
        deploy_config: Optional[DeployConfig] = None,
        session: Optional[httpx.ClientSession] = None,
        headers: Optional[Dict[str, str]] = None,
        _token: Optional[str] = None,
        token_file: Optional[str] = None,
    ):
        self._async_client = async_to_blocking(
            aioclient.BatchClient.create(
                billing_project, deploy_config, session, headers=headers, _token=_token, token_file=token_file
            )
        )

    @property
    def billing_project(self):
        return self._async_client.billing_project

    def reset_billing_project(self, billing_project):
        self._async_client.reset_billing_project(billing_project)

    def list_batches(self, q=None, last_batch_id=None, limit=2**64, version=None):
        for b in ait_to_blocking(
            self._async_client.list_batches(q=q, last_batch_id=last_batch_id, limit=limit, version=version)
        ):
            yield Batch(b)

    def get_job(self, batch_id, job_id):
        j = async_to_blocking(self._async_client.get_job(batch_id, job_id))
        return Job(j)

    def get_job_log(self, batch_id, job_id) -> Optional[Dict[str, Any]]:
        log = async_to_blocking(self._async_client.get_job_log(batch_id, job_id))
        return log

    def get_job_attempts(self, batch_id, job_id):
        attempts = async_to_blocking(self._async_client.get_job_attempts(batch_id, job_id))
        return attempts

    def get_batch(self, id):
        b = async_to_blocking(self._async_client.get_batch(id))
        return Batch(b)

    def create_batch(self, attributes=None, callback=None, token=None, cancel_after_n_failures=None) -> 'Batch':
        batch = self._async_client.create_batch(
            attributes=attributes, callback=callback, token=token, cancel_after_n_failures=cancel_after_n_failures
        )
        return Batch(batch)

    def get_billing_project(self, billing_project):
        return async_to_blocking(self._async_client.get_billing_project(billing_project))

    def list_billing_projects(self):
        return async_to_blocking(self._async_client.list_billing_projects())

    def create_billing_project(self, project):
        return async_to_blocking(self._async_client.create_billing_project(project))

    def add_user(self, user, project):
        return async_to_blocking(self._async_client.add_user(user, project))

    def remove_user(self, user, project):
        return async_to_blocking(self._async_client.remove_user(user, project))

    def close_billing_project(self, project):
        return async_to_blocking(self._async_client.close_billing_project(project))

    def reopen_billing_project(self, project):
        return async_to_blocking(self._async_client.reopen_billing_project(project))

    def delete_billing_project(self, project):
        return async_to_blocking(self._async_client.delete_billing_project(project))

    def edit_billing_limit(self, project, limit):
        return async_to_blocking(self._async_client.edit_billing_limit(project, limit))

    def supported_regions(self) -> List[str]:
        return async_to_blocking(self._async_client.supported_regions())

    def cloud(self):
        return async_to_blocking(self._async_client.cloud())

    def close(self):
        async_to_blocking(self._async_client.close())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
