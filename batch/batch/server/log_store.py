import os

import google
import hailjwt as hj

from .google_storage import GCS


class LogStore:
    def __init__(self, blocking_pool, instance_id, log, batch_gsa_key=None, batch_bucket_name=None):
        self.instance_id = instance_id
        self.log = log
        self.gcs = GCS(blocking_pool, batch_gsa_key)
        if batch_bucket_name is None:
            batch_jwt = os.environ.get('BATCH_JWT', '/batch-jwt/jwt')
            with open(batch_jwt, 'r') as f:
                batch_bucket_name = hj.JWTClient.unsafe_decode(f.read())['bucket_name']
        self.batch_bucket_name = batch_bucket_name

    def _gs_log_path(self, job_id, task_name):
        return f'{self.instance_id}/{job_id}/{task_name}/job.log'

    async def write_gs_log_file(self, job_id, task_name, log):
        path = self._gs_log_path(job_id, task_name)
        err = await self.gcs.upload_private_gs_file_from_string(self.batch_bucket_name, path, log)
        if err is None:
            return (f'gs://{self.batch_bucket_name}/{path}', err)
        return (None, err)

    async def read_gs_log_file(self, uri):
        assert uri.startswith('gs://')
        uri = uri.lstrip('gs://').split('/')
        bucket_name = uri[0]
        path = '/'.join(uri[1:])
        return await self.gcs.download_gs_file_as_string(bucket_name, path)

    async def delete_gs_log_file(self, job_id, task_name):
        path = self._gs_log_path(job_id, task_name)
        err = await self.gcs.delete_gs_file(self.batch_bucket_name, path)
        if isinstance(err, google.api_core.exceptions.NotFound):
            self.log.info(f'ignoring: cannot delete log file that does not exist: {err}')
            err = None
        return err
