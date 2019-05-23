import asyncio

from . import aioclient


def async_to_blocking(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class Job:
    @staticmethod
    def exit_code(job_status):
        return aioclient.Job.exit_code(job_status)

    @classmethod
    def from_async_job(cls, job):
        j = object.__new__(cls)
        j._async_job = job
        return j

    def __init__(self, client, id, attributes=None, parent_ids=None, _status=None):
        self._async_job = aioclient.Job(client, id, attributes=attributes,
                                        parent_ids=parent_ids, _status=_status)

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

    def create_job(self, image, command=None, args=None, env=None, ports=None,
                         resources=None, tolerations=None, volumes=None, security_context=None,
                         service_account_name=None, attributes=None, callback=None, parents=None,
                         input_files=None, output_files=None, always_run=False):
        coroutine = self._async_batch.create_job(
            image, command=command, args=args, env=env, ports=ports,
            resources=resources, tolerations=tolerations, volumes=volumes,
            security_context=security_context, service_account_name=service_account_name,
            attributes=attributes, callback=callback, parents=parents,
            input_files=input_files, output_files=output_files, always_run=always_run)
        return Job.from_async_job(async_to_blocking(coroutine))

    def close(self):
        async_to_blocking(self._async_batch.close())

    def cancel(self):
        async_to_blocking(self._async_batch.cancel())

    def status(self):
        return async_to_blocking(self._async_batch.status())

    def wait(self):
        return async_to_blocking(self._async_batch.wait())

    def delete(self):
        async_to_blocking(self._async_batch.delete())


class BatchClient:
    def __init__(self, session, url=None, token_file=None, token=None, headers=None):
        self._async_client = aioclient.BatchClient(session, url=url, token_file=token_file,
                                                   token=token, headers=headers)

    @property
    def url(self):
        return self._async_client.url

    @property
    def bucket(self):
        return self._async_client.bucket

    def _refresh_k8s_state(self):
        async_to_blocking(self._async_client._refresh_k8s_state())

    def list_batches(self, complete=None, success=None, attributes=None):
        batches = async_to_blocking(
            self._async_client.list_batches(complete=complete, success=success, attributes=attributes))
        return [Batch.from_async_batch(b) for b in batches]

    def get_job(self, id):
        j = async_to_blocking(self._async_client.get_job(id))
        return Job.from_async_job(j)

    def get_batch(self, id):
        b = async_to_blocking(self._async_client.get_batch(id))
        return Batch.from_async_batch(b)

    def create_batch(self, attributes=None, callback=None, ttl=None):
        b = async_to_blocking(
            self._async_client.create_batch(attributes=attributes, callback=callback, ttl=ttl))
        return Batch.from_async_batch(b)

    def close(self):
        async_to_blocking(self._async_client.close())
