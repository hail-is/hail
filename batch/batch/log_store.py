import logging

import google

from .google_storage import GCS


log = logging.getLogger('batch.logstore')


class LogStore:
    log_file_name = 'container_logs'
    pod_status_file_name = 'pod_status'

    files = (log_file_name, pod_status_file_name)

    @staticmethod
    def _parse_uri(uri):
        assert uri.startswith('gs://')
        uri = uri.lstrip('gs://').split('/')
        bucket = uri[0]
        path = '/'.join(uri[1:])
        return bucket, path

    def __init__(self, blocking_pool, instance_id, batch_bucket_name):
        self.instance_id = instance_id
        self.gcs = GCS(blocking_pool)
        self.batch_bucket_name = batch_bucket_name

    def gs_job_output_directory(self, batch_id, job_id, token):
        return f'gs://{self.batch_bucket_name}/{self.instance_id}/{batch_id}/{job_id}/{token}/'

    async def write_gs_file(self, directory, file_name, data):
        assert file_name in LogStore.files
        bucket, path = LogStore._parse_uri(f'{directory}{file_name}')
        return await self.gcs.upload_private_gs_file_from_string(bucket, path, data)

    async def read_gs_file(self, directory, file_name):
        assert file_name in LogStore.files
        bucket, path = LogStore._parse_uri(f'{directory}{file_name}')
        return await self.gcs.download_gs_file_as_string(bucket, path)

    async def delete_gs_file(self, directory, file_name):
        assert file_name in LogStore.files
        bucket, path = LogStore._parse_uri(f'{directory}{file_name}')
        err = await self.gcs.delete_gs_file(bucket, path)
        if isinstance(err, google.api_core.exceptions.NotFound):
            log.info(f'ignoring: cannot delete file that does not exist: {err}')
            err = None
        return err

    async def delete_gs_files(self, directory):
        errors = []
        for file in LogStore.files:
            err = await self.delete_gs_file(directory, file)
            errors.append((file, err))
        return errors
