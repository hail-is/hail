import asyncio

from . import aioclient


def async_to_blocking(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def sync_anext(ait):
    return async_to_blocking(ait.__anext__())


def agen_to_blocking(agen):
    while True:
        try:
            yield sync_anext(agen)
        except StopAsyncIteration:
            break


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

    @classmethod
    def from_async_job(cls, job):
        j = object.__new__(cls)
        j._async_job = job
        return j

    def __init__(self, batch, job_id, _status=None):
        j = aioclient.SubmittedJob(batch, job_id, _status)
        self._async_job = aioclient.Job(j)

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

    def status(self):
        return async_to_blocking(self._async_job.status())

    def wait(self):
        return async_to_blocking(self._async_job.wait())

    def log(self):
        return async_to_blocking(self._async_job.log())

    def attempts(self):
        return async_to_blocking(self._async_job.attempts())


class Batch:
    @classmethod
    def from_async_batch(cls, batch):
        b = object.__new__(cls)
        b._async_batch = batch
        return b

    def __init__(self, client, id, attributes, n_jobs):
        self._async_batch = aioclient.Batch(client, id, attributes, n_jobs)

    @property
    def id(self):
        return self._async_batch.id

    @property
    def attributes(self):
        return self._async_batch.attributes

    def cancel(self):
        async_to_blocking(self._async_batch.cancel())

    def status(self):
        return async_to_blocking(self._async_batch.status())

    def jobs(self, q=None):
        return agen_to_blocking(self._async_batch.jobs(q=q))

    def wait(self):
        return async_to_blocking(self._async_batch.wait())

    def delete(self):
        async_to_blocking(self._async_batch.delete())


class BatchBuilder:
    @classmethod
    def from_async_builder(cls, builder):
        b = object.__new__(cls)
        b._async_builder = builder
        return b

    def __init__(self, client, attributes, callback):
        self._async_builder = aioclient.BatchBuilder(client, attributes, callback)

    @property
    def attributes(self):
        return self._async_builder.attributes

    @property
    def callback(self):
        return self._async_builder.callback

    def create_job(self, image, command, env=None, mount_docker_socket=False,
                   port=None, resources=None, secrets=None,
                   service_account=None, attributes=None, parents=None,
                   input_files=None, output_files=None, always_run=False, pvc_size=None,
                   timeout=None):
        if parents:
            parents = [parent._async_job for parent in parents]

        async_job = self._async_builder.create_job(
            image, command, env=env, mount_docker_socket=mount_docker_socket,
            port=port, resources=resources, secrets=secrets,
            service_account=service_account,
            attributes=attributes, parents=parents,
            input_files=input_files, output_files=output_files, always_run=always_run,
            pvc_size=pvc_size, timeout=timeout)

        return Job.from_async_job(async_job)

    def _create(self, *args, **kwargs):
        async_batch = async_to_blocking(self._async_builder._create(*args, **kwargs))
        return Batch.from_async_batch(async_batch)

    def submit(self, *args, **kwargs):
        async_batch = async_to_blocking(self._async_builder.submit(*args, **kwargs))
        return Batch.from_async_batch(async_batch)


class BatchClient:
    def __init__(self, billing_project, deploy_config=None, session=None,
                 headers=None, _token=None):
        self._async_client = async_to_blocking(
            aioclient.BatchClient(billing_project, deploy_config, session, headers=headers, _token=_token))

    @property
    def bucket(self):
        return self._async_client.bucket

    @property
    def billing_project(self):
        return self._async_client.billing_project

    def list_batches(self, q=None):
        for b in agen_to_blocking(self._async_client.list_batches(q=q)):
            yield Batch.from_async_batch(b)

    def get_job(self, batch_id, job_id):
        j = async_to_blocking(self._async_client.get_job(batch_id, job_id))
        return Job.from_async_job(j)

    def get_job_log(self, batch_id, job_id):
        log = async_to_blocking(self._async_client.get_job_log(batch_id, job_id))
        return log

    def get_batch(self, id):
        b = async_to_blocking(self._async_client.get_batch(id))
        return Batch.from_async_batch(b)

    def create_batch(self, attributes=None, callback=None):
        builder = self._async_client.create_batch(attributes=attributes, callback=callback)
        return BatchBuilder.from_async_builder(builder)

    def close(self):
        async_to_blocking(self._async_client.close())
