import logging

from .google_storage import GCS

log = logging.getLogger('logstore')


class LogStore:
    def __init__(self, bucket_name, instance_id, blocking_pool, *, project=None, credentials=None):
        self.bucket_name = bucket_name
        self.instance_id = instance_id
        self.log_root = f'gs://{bucket_name}/batch2/logs/{instance_id}'
        self.gcs = GCS(blocking_pool, project=project, credentials=credentials)

    def worker_log_path(self, machine_name, log_file):
        # this has to match worker startup-script
        return f'{self.log_root}/worker/{machine_name}/{log_file}'

    def batch_log_dir(self, batch_id):
        return f'{self.log_root}/batch/{batch_id}'

    def log_path(self, batch_id, job_id, task):
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{task}/log'

    async def read_log_file(self, batch_id, job_id, task):
        path = self.log_path(batch_id, job_id, task)
        return await self.gcs.read_gs_file(path)

    async def write_log_file(self, batch_id, job_id, task, data):
        return await self.gcs.write_gs_file(self.log_path(batch_id, job_id, task), data)

    async def delete_batch_logs(self, batch_id):
        await self.gcs.delete_gs_files(
            self.batch_log_dir(batch_id))
