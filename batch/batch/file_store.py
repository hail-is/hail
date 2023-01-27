import asyncio
import logging
from typing import Optional

import pandas as pd

from hailtop.aiotools.fs import AsyncFS

from .batch_format_version import BatchFormatVersion
from .globals import BATCH_FORMAT_VERSION
from .resource_usage import ResourceUsageMonitor
from .spec_writer import SpecWriter

log = logging.getLogger('logstore')


class FileStore:
    def __init__(self, fs: AsyncFS, batch_logs_storage_uri, instance_id):
        self.fs = fs
        self.batch_logs_storage_uri = batch_logs_storage_uri
        self.instance_id = instance_id

        self.batch_logs_root = f'{batch_logs_storage_uri}/batch/logs/{instance_id}/batch'

        log.info(f'BATCH_LOGS_ROOT {self.batch_logs_root}')
        format_version = BatchFormatVersion(BATCH_FORMAT_VERSION)
        log.info(f'EXAMPLE BATCH_JOB_LOGS_PATH {self.log_path(format_version, 1, 1, "abc123", "main")}')

    def batch_log_dir(self, batch_id):
        return f'{self.batch_logs_root}/{batch_id}'

    def log_path(self, format_version, batch_id, job_id, attempt_id, task):
        if not format_version.has_attempt_in_log_path():
            return f'{self.batch_log_dir(batch_id)}/{job_id}/{task}/log'
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{attempt_id}/{task}/log'

    def resource_usage_path(self, format_version, batch_id, job_id, attempt_id, task):
        if not format_version.has_attempt_in_log_path():
            return f'{self.batch_log_dir(batch_id)}/{job_id}/{task}/resource_usage'
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{attempt_id}/{task}/resource_usage'

    def jvm_profile_path(self, format_version, batch_id, job_id, attempt_id, task):
        if not format_version.has_attempt_in_log_path():
            return f'{self.batch_log_dir(batch_id)}/{job_id}/{task}/jvm_profile'
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{attempt_id}/{task}/jvm_profile'

    async def read_log_file(self, format_version, batch_id, job_id, attempt_id, task) -> bytes:
        url = self.log_path(format_version, batch_id, job_id, attempt_id, task)
        return await self.fs.read(url)

    async def write_log_file(self, format_version, batch_id, job_id, attempt_id, task, data: bytes):
        url = self.log_path(format_version, batch_id, job_id, attempt_id, task)
        await self.fs.write(url, data)

    async def write_jvm_profile(self, format_version, batch_id, job_id, attempt_id, task, data):
        url = self.jvm_profile_path(format_version, batch_id, job_id, attempt_id, task)
        await self.fs.write(url, data)

    async def read_jvm_profile(self, format_version, batch_id, job_id, attempt_id, task):
        url = self.jvm_profile_path(format_version, batch_id, job_id, attempt_id, task)
        return await self.fs.read(url)

    async def read_resource_usage_file(
        self, format_version, batch_id, job_id, attempt_id, task
    ) -> Optional[pd.DataFrame]:
        url = self.resource_usage_path(format_version, batch_id, job_id, attempt_id, task)
        data = await self.fs.read(url)
        return ResourceUsageMonitor.decode_to_df(data)

    async def write_resource_usage_file(self, format_version, batch_id, job_id, attempt_id, task, data):
        url = self.resource_usage_path(format_version, batch_id, job_id, attempt_id, task)
        await self.fs.write(url, data)

    async def delete_batch_logs(self, batch_id):
        url = self.batch_log_dir(batch_id)
        await self.fs.rmtree(None, url)

    def status_path(self, batch_id, job_id, attempt_id):
        return f'{self.batch_log_dir(batch_id)}/{job_id}/{attempt_id}/status.json'

    async def read_status_file(self, batch_id, job_id, attempt_id):
        url = self.status_path(batch_id, job_id, attempt_id)
        data = await self.fs.read(url)
        return data.decode('utf-8')

    async def write_status_file(self, batch_id, job_id, attempt_id, status):
        url = self.status_path(batch_id, job_id, attempt_id)
        await self.fs.write(url, status.encode('utf-8'))

    async def delete_status_file(self, batch_id, job_id, attempt_id):
        url = self.status_path(batch_id, job_id, attempt_id)
        return await self.fs.remove(url)

    def specs_dir(self, batch_id, token):
        return f'{self.batch_logs_root}/{batch_id}/bunch/{token}'

    def specs_path(self, batch_id, token):
        return f'{self.specs_dir(batch_id, token)}/specs'

    def specs_index_path(self, batch_id, token):
        return f'{self.specs_dir(batch_id, token)}/specs.idx'

    async def read_spec_file(self, batch_id, token, start_job_id, job_id):
        idx_url = self.specs_index_path(batch_id, token)
        idx_start, idx_end = SpecWriter.get_index_file_offsets(job_id, start_job_id)
        offsets = await self.fs.read_range(idx_url, idx_start, idx_end)

        spec_url = self.specs_path(batch_id, token)
        spec_start, spec_end = SpecWriter.get_spec_file_offsets(offsets)
        data = await self.fs.read_range(spec_url, spec_start, spec_end)
        return data.decode('utf-8')

    async def write_spec_file(self, batch_id, token, data_bytes, offsets_bytes):
        idx_url = self.specs_index_path(batch_id, token)
        write1 = self.fs.write(idx_url, offsets_bytes)

        specs_url = self.specs_path(batch_id, token)
        write2 = self.fs.write(specs_url, data_bytes)

        await asyncio.gather(write1, write2)

    async def delete_spec_file(self, batch_id, token):
        url = self.specs_dir(batch_id, token)
        await self.fs.rmtree(None, url)

    async def close(self):
        await self.fs.close()
