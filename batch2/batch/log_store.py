import os
import logging
import asyncio
import google

from .google_storage import GCS
from .globals import tasks


log = logging.getLogger('logstore')


class LogStore:
    @staticmethod
    def container_log_path(directory, container_name):
        assert container_name in tasks
        return f'{directory}{container_name}/job.log'

    @staticmethod
    def container_status_path(directory, container_name):
        assert container_name in tasks
        return f'{directory}{container_name}/status'

    def __init__(self, blocking_pool, instance_id, bucket_name):
        self.instance_id = instance_id
        self.batch_bucket_name = bucket_name

        batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(
            batch_gsa_key)

        self.gcs = GCS(blocking_pool, credentials)

    def gs_job_output_directory(self, batch_id, job_id):
        return f'gs://{self.batch_bucket_name}/{self.instance_id}/{batch_id}/{job_id}/'

    async def write_gs_file(self, uri, data):
        return await self.gcs.write_gs_file(uri, data)

    async def read_gs_file(self, uri):
        return await self.gcs.read_gs_file(uri)

    async def delete_gs_file(self, uri):
        err = await self.gcs.delete_gs_file(uri)
        if isinstance(err, google.api_core.exceptions.NotFound):
            log.info(f'ignoring: cannot delete file that does not exist: {err}')
            err = None
        return err

    async def delete_gs_files(self, directory):
        files = [LogStore.container_log_path(directory, container) for container in tasks]
        files.extend([LogStore.container_status_path(directory, container) for container in tasks])
        errors = await asyncio.gather(*[self.delete_gs_file(file) for file in files])
        return list(zip(files, errors))
