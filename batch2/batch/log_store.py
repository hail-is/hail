import logging

from .batch_configuration import INSTANCE_ID
from .google_storage import GCS

log = logging.getLogger('logstore')


class LogStore:
    def __init__(self, bucket_name, blocking_pool, credentials=None):
        self.log_root = f'gs://{bucket_name}/batch2/logs/{INSTANCE_ID}'
        self.gcs = GCS(blocking_pool, credentials)

    def worker_log_path(self, machine_name, log_file):
        # this has to match worker startup-script
        return f'{self.log_root}/worker/{machine_name}/{log_file}'

    def batch_log_dir(self, batch_id):
        return f'{self.log_root}/batch/{batch_id}'

    def log_path(self, batch_id, job_id, task):
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{task}/log'

    async def read_log_file(self, batch_id, job_id, task):
        path = self.log_path(batch_id, job_id, task)
        log.info(f'read log file {path}')
        return await self.gcs.read_gs_file(path)

    async def write_log_file(self, batch_id, job_id, task, data):
        return await self.gcs.write_gs_file(self.log_path(batch_id, job_id, task), data)

    async def delete_batch_logs(self, batch_id):
        await self.gcs.delete_gs_files(
            self.batch_log_dir(self.log_root, batch_id))
