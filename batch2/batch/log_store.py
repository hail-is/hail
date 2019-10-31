import os
import logging
import asyncio
import google

from .google_storage import GCS
from .globals import tasks

log = logging.getLogger('logstore')


class LogStore:
    def __init__(self, app):
        self.log_root = app['log_root']

        batch_gsa_key = os.environ.get('BATCH_GSA_KEY', '/batch-gsa-key/privateKeyData')
        credentials = google.oauth2.service_account.Credentials.from_service_account_file(
            batch_gsa_key)

        self.gcs = GCS(app['blocking_pool'], credentials)

    @staticmethod
    def log_path(log_root, batch_id, job_id, task):
        return f'{log_root}/batch/{batch_id}/{job_id}/{task}/log'

    async def read_log_file(self, batch_id, job_id, task):
        return await self.gcs.read_gs_file(self.log_path(self.log_root, batch_id, job_id, task))

    async def write_log_file(self, batch_id, job_id, task, data):
        return await self.gcs.write_gs_file(self.log_path(self.log_root, batch_id, job_id, task), data)

    async def _delete_gs_file(self, uri):
        try:
            await self.gcs.delete_gs_file(uri)
        except google.api_core.exceptions.NotFound:
            log.exception(f'file not found: {uri}, ignoring')

    async def delete_log_files(self, batch_id, job_id):
        await asyncio.gather(*[
            self._delete_gs_file(self.log_path(self.log_root, batch_id, job_id, task))
            for task in tasks
        ])
