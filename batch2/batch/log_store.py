import os
import logging
import google

from .google_storage import GCS

log = logging.getLogger('logstore')


class LogStore:
    def __init__(self, log_root, blocking_pool, credentials=None):
        self.log_root = log_root
        self.gcs = GCS(blocking_pool, credentials)

    def batch_log_dir(self, batch_id):
        return f'{self.log_root}/batch/{batch_id}'

    def log_path(self, batch_id, job_id, task):
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{task}/log'

    async def read_log_file(self, batch_id, job_id, task):
        return await self.gcs.read_gs_file(self.log_path(self.log_root, batch_id, job_id, task))

    async def write_log_file(self, batch_id, job_id, task, data):
        return await self.gcs.write_gs_file(self.log_path(self.log_root, batch_id, job_id, task), data)

    async def delete_batch_logs(self, batch_id):
        await self.gcs.delete_gs_files(
            self.batch_log_dir(self.log_root, batch_id))
