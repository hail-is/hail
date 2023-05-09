import os
import tempfile
from typing import Dict

import aiohttp

from hailtop import httpx
from hailtop.aiocloud import aiogoogle
from hailtop.utils import check_exec_output, request_retry_transient_errors

from ....worker.worker_api import CloudWorkerAPI
from ..instance_config import GCPSlimInstanceConfig
from .credentials import GCPUserCredentials
from .disk import GCPDisk


class GCPWorkerAPI(CloudWorkerAPI[GCPUserCredentials]):
    nameserver_ip = '169.254.169.254'

    # async because GoogleSession must be created inside a running event loop
    @staticmethod
    async def from_env() -> 'GCPWorkerAPI':
        project = os.environ['PROJECT']
        zone = os.environ['ZONE'].rsplit('/', 1)[1]
        session = aiogoogle.GoogleSession()
        return GCPWorkerAPI(project, zone, session)

    def __init__(self, project: str, zone: str, session: aiogoogle.GoogleSession):
        self.project = project
        self.zone = zone
        self._google_session = session
        self._compute_client = aiogoogle.GoogleComputeClient(project, session=session)
        self._gcsfuse_credential_files: Dict[str, str] = {}

    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> GCPDisk:
        return GCPDisk(
            zone=self.zone,
            project=self.project,
            instance_name=instance_name,
            name=disk_name,
            size_in_gb=size_in_gb,
            mount_path=mount_path,
            compute_client=self._compute_client,
        )

    def get_cloud_async_fs(self) -> aiogoogle.GoogleStorageAsyncFS:
        return aiogoogle.GoogleStorageAsyncFS(session=self._google_session)

    def get_compute_client(self) -> aiogoogle.GoogleComputeClient:
        return self._compute_client

    def user_credentials(self, credentials: Dict[str, str]) -> GCPUserCredentials:
        return GCPUserCredentials(credentials)

    async def worker_access_token(self, session: httpx.ClientSession) -> Dict[str, str]:
        async with await request_retry_transient_errors(
            session,
            'POST',
            'http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token',
            headers={'Metadata-Flavor': 'Google'},
            timeout=aiohttp.ClientTimeout(total=60),  # type: ignore
        ) as resp:
            access_token = (await resp.json())['access_token']
            return {'username': 'oauth2accesstoken', 'password': access_token}

    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> GCPSlimInstanceConfig:
        return GCPSlimInstanceConfig.from_dict(config_dict)

    def _write_gcsfuse_credentials(self, credentials: GCPUserCredentials, mount_base_path_data: str) -> str:
        if mount_base_path_data not in self._gcsfuse_credential_files:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as credsfile:
                credsfile.write(credentials.key)
                self._gcsfuse_credential_files[mount_base_path_data] = credsfile.name
        return self._gcsfuse_credential_files[mount_base_path_data]

    async def _mount_cloudfuse(
        self,
        credentials: GCPUserCredentials,
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ):  # pylint: disable=unused-argument

        fuse_credentials_path = self._write_gcsfuse_credentials(credentials, mount_base_path_data)

        bucket = config['bucket']
        assert bucket

        options = ['allow_other']
        if config['read_only']:
            options.append('ro')

        try:
            billing_project_flag = ['--billing-project', config["requester_pays_project"]]
        except KeyError:
            billing_project_flag = []

        await check_exec_output(
            '/usr/bin/gcsfuse',
            '-o',
            ','.join(options),
            '--file-mode',
            '770',
            '--dir-mode',
            '770',
            '--implicit-dirs',
            '--key-file',
            fuse_credentials_path,
            *billing_project_flag,
            bucket,
            mount_base_path_data,
        )

    async def unmount_cloudfuse(self, mount_base_path_data: str):
        try:
            await check_exec_output('fusermount', '-u', mount_base_path_data)
        finally:
            os.remove(self._gcsfuse_credential_files[mount_base_path_data])
            del self._gcsfuse_credential_files[mount_base_path_data]

    def __str__(self):
        return f'project={self.project} zone={self.zone}'
