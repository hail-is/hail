import logging
import asyncio

from hailtop.aiogoogle.client.storage_client import GoogleStorageAsyncFS

from .spec_writer import SpecWriter
from .globals import BATCH_FORMAT_VERSION
from .batch_format_version import BatchFormatVersion

log = logging.getLogger('logstore')


class LogStore:
    def __init__(self, batch_logs_bucket_name, instance_id, *, credentials=None):
        self.batch_logs_bucket_name = batch_logs_bucket_name
        self.instance_id = instance_id
        self.batch_logs_root = f'gs://{batch_logs_bucket_name}/batch/logs/{instance_id}/batch'

        self.gcs_fs = GoogleStorageAsyncFS(credentials=credentials)
        self.sem = asyncio.Semaphore(50)

        log.info(f'BATCH_LOGS_ROOT {self.batch_logs_root}')
        format_version = BatchFormatVersion(BATCH_FORMAT_VERSION)
        log.info(f'EXAMPLE BATCH_JOB_LOGS_PATH {self.log_path(format_version, 1, 1, "abc123", "main")}')

    async def _write(self, url, data, binary=True):
        async with await self.gcs_fs.create(url) as f:
            if not binary:
                data = data.encode('utf-8')
            await f.write(data)

    async def _read(self, url, binary=True):
        async with await self.gcs_fs.open(url) as f:
            data = await f.read()
            if not binary:
                data = data.decode('utf-8')
            return data

    async def _rmtree(self, url):
        async with self.sem:
            await self.gcs_fs.rmtree(self.sem, url)

    async def _open_from(self, url, start, end, binary=True):
        async with await self.gcs_fs.open_from(url, start, end) as f:
            data = await f.read()
            if not binary:
                data = data.decode('utf-8')
            return data

    def batch_log_dir(self, batch_id):
        return f'{self.batch_logs_root}/{batch_id}'

    def log_path(self, format_version, batch_id, job_id, attempt_id, task):
        if not format_version.has_attempt_in_log_path():
            return f'{self.batch_log_dir(batch_id)}/{job_id}/{task}/log'
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{attempt_id}/{task}/log'

    async def read_log_file(self, format_version, batch_id, job_id, attempt_id, task):
        url = self.log_path(format_version, batch_id, job_id, attempt_id, task)
        return await self._read(url, binary=False)

    async def write_log_file(self, format_version, batch_id, job_id, attempt_id, task, data):
        url = self.log_path(format_version, batch_id, job_id, attempt_id, task)
        await self._write(url, data, binary=False)

    async def delete_batch_logs(self, batch_id):
        url = self.batch_log_dir(batch_id)
        await self._rmtree(url)

    def status_path(self, batch_id, job_id, attempt_id):
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{attempt_id}/status.json'

    async def read_status_file(self, batch_id, job_id, attempt_id):
        url = self.status_path(batch_id, job_id, attempt_id)
        return await self._read(url, binary=False)

    async def write_status_file(self, batch_id, job_id, attempt_id, status):
        url = self.status_path(batch_id, job_id, attempt_id)
        await self._write(url, status, binary=False)

    async def delete_status_file(self, batch_id, job_id, attempt_id):
        url = self.status_path(batch_id, job_id, attempt_id)
        return await self.gcs_fs.remove(url)

    def specs_dir(self, batch_id, token):
        return f'{self.batch_logs_root}/{batch_id}/bunch/{token}'

    def specs_path(self, batch_id, token):
        return f'{self.specs_dir(batch_id, token)}/specs'

    def specs_index_path(self, batch_id, token):
        return f'{self.specs_dir(batch_id, token)}/specs.idx'

    async def read_spec_file(self, batch_id, token, start_job_id, job_id):
        idx_url = self.specs_index_path(batch_id, token)
        idx_start, idx_end = SpecWriter.get_index_file_offsets(job_id, start_job_id)
        offsets = await self._open_from(idx_url, idx_start, idx_end)

        spec_url = self.specs_path(batch_id, token)
        spec_start, spec_end = SpecWriter.get_spec_file_offsets(offsets)
        return await self._open_from(spec_url, spec_start, spec_end)

    async def write_spec_file(self, batch_id, token, data_bytes, offsets_bytes):
        idx_url = self.specs_index_path(batch_id, token)
        write1 = self._write(idx_url, offsets_bytes)

        specs_url = self.specs_path(batch_id, token)
        write2 = self._write(specs_url, data_bytes)

        await asyncio.gather(write1, write2)

    async def delete_spec_file(self, batch_id, token):
        url = self.specs_dir(batch_id, token)
        await self._rmtree(url)

    async def close(self):
        await self.gcs_fs.close()
