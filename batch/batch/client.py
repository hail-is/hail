import asyncio

from . import aioclient


def async_to_blocking(coroutine):
    return asyncio.get_event_loop().run_until_complete(coroutine)


class Job(aioclient.Job):
    @staticmethod
    def from_async_job(client, job):
        return Job(client, job.id, job.attributes, job.parent_ids, job._status)

    def is_complete(self):
        return async_to_blocking(super().is_complete())

    def status(self):
        return async_to_blocking(super().status())

    def wait(self):
        return async_to_blocking(super().wait())

    def log(self):
        return async_to_blocking(super().log())


class Batch(aioclient.Batch):
    @staticmethod
    def from_async_batch(client, batch):
        return Batch(client, batch.id, batch.attributes)

    def create_job(self, image, command=None, args=None, env=None, ports=None,
                         resources=None, tolerations=None, volumes=None, security_context=None,
                         service_account_name=None, attributes=None, callback=None, parent_ids=None,
                         input_files=None, output_files=None, always_run=False):
        coroutine = super().create_job(image, command=command, args=args, env=env, ports=ports,
                                       resources=resources, tolerations=tolerations, volumes=volumes,
                                       security_context=security_context, service_account_name=service_account_name,
                                       attributes=attributes, callback=callback, parent_ids=parent_ids,
                                       input_files=input_files, output_files=output_files, always_run=always_run)
        return Job.from_async_job(self.client, async_to_blocking(coroutine))

    def close(self):
        async_to_blocking(super().close())

    def cancel(self):
        async_to_blocking(super().cancel())

    def status(self):
        return async_to_blocking(super().status())

    def wait(self):
        return async_to_blocking(super().wait())

    def delete(self):
        async_to_blocking(super().delete())


class BatchClient(aioclient.BatchClient):
    def _refresh_k8s_state(self):
        async_to_blocking(super()._refresh_k8s_state())

    def list_batches(self, complete=None, success=None, attributes=None):
        batches = async_to_blocking(
            super().list_batches(complete=complete, success=success, attributes=attributes))
        return [Batch.from_async_batch(self, b) for b in batches]

    def get_job(self, id):
        j = async_to_blocking(super().get_job(id))
        return Job.from_async_job(self, j)

    def get_batch(self, id):
        b = async_to_blocking(super().get_batch(id))
        return Batch.from_async_batch(self, b)

    def create_batch(self, attributes=None, callback=None, ttl=None):
        b = async_to_blocking(
            super().create_batch(attributes=attributes, callback=callback, ttl=ttl))
        return Batch.from_async_batch(self, b)

    def close(self):
        async_to_blocking(super().close())
