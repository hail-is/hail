import asyncio

from . import aioclient


def async_to_blocking(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class Job:
    @staticmethod
    def exit_code(job_status):
        return aioclient.Job.exit_code(job_status)

    @staticmethod
    def total_duration(job_status):
        return aioclient.Job.total_duration(job_status)

    @classmethod
    def from_async_job(cls, job):
        j = object.__new__(cls)
        j._async_job = job
        return j

    def __init__(self, batch, job_id, attributes=None, parent_ids=None, _status=None):
        j = aioclient.SubmittedJob(batch, job_id, attributes, parent_ids, _status)
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

    @property
    def attributes(self):
        return self._async_job.attributes

    @property
    def parent_ids(self):
        return self._async_job.parent_ids

    def is_complete(self):
        return async_to_blocking(self._async_job.is_complete())

    def status(self):
        return async_to_blocking(self._async_job.status())

    def wait(self):
        return async_to_blocking(self._async_job.wait())

    def log(self):
        return async_to_blocking(self._async_job.log())

    def pod_status(self):
        return async_to_blocking(self._async_job.pod_status())


class Batch:
    @classmethod
    def from_async_batch(cls, batch):
        b = object.__new__(cls)
        b._async_batch = batch
        return b

    def __init__(self, client, id, attributes):
        self._async_batch = aioclient.Batch(client, id, attributes)

    @property
    def id(self):
        return self._async_batch.id

    @property
    def attributes(self):
        return self._async_batch.attributes

    def cancel(self):
        async_to_blocking(self._async_batch.cancel())

    def status(self, include_jobs=True):
        return async_to_blocking(self._async_batch.status(include_jobs=include_jobs))

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
                   resources=None, secrets=None,
                   service_account=None, attributes=None, parents=None,
                   input_files=None, output_files=None, always_run=False, pvc_size=None):
        if parents:
            parents = [parent._async_job for parent in parents]

        async_job = self._async_builder.create_job(
            image, command, env=env, mount_docker_socket=mount_docker_socket,
            resources=resources, secrets=secrets,
            service_account=service_account,
            attributes=attributes, parents=parents,
            input_files=input_files, output_files=output_files, always_run=always_run,
            pvc_size=pvc_size)

        return Job.from_async_job(async_job)

    def submit(self):
        async_batch = async_to_blocking(self._async_builder.submit())
        return Batch.from_async_batch(async_batch)


class BatchClient:
    def __init__(self, deploy_config=None, session=None, headers=None,
                 _token=None, _service='batch'):
        self._async_client = async_to_blocking(
            aioclient.BatchClient(deploy_config, session, headers=headers, _token=_token, _service=_service))

    @property
    def bucket(self):
        return self._async_client.bucket

    def _refresh_k8s_state(self):
        async_to_blocking(self._async_client._refresh_k8s_state())

    def list_batches(self, complete=None, success=None, attributes=None):
        batches = async_to_blocking(
            self._async_client.list_batches(complete=complete, success=success, attributes=attributes))
        return [Batch.from_async_batch(b) for b in batches]

    def get_job(self, batch_id, job_id):
        j = async_to_blocking(self._async_client.get_job(batch_id, job_id))
        return Job.from_async_job(j)

    def get_batch(self, id):
        b = async_to_blocking(self._async_client.get_batch(id))
        return Batch.from_async_batch(b)

    def create_batch(self, attributes=None, callback=None):
        builder = self._async_client.create_batch(attributes=attributes, callback=callback)
        return BatchBuilder.from_async_builder(builder)

    def close(self):
        async_to_blocking(self._async_client.close())
