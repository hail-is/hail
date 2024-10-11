import os
from typing import Dict, List, Optional

import orjson
from aiohttp import web

from hailtop import httpx
from hailtop.aiocloud.aioazure import AzureCredentials
from hailtop.aiocloud.aioterra.azure import TerraAzureAsyncFS
from hailtop.aiotools.fs import AsyncFS
from hailtop.auth.auth import IdentityProvider
from hailtop.auth.flow import AzureFlow

from .....worker.disk import CloudDisk
from .....worker.worker_api import CloudWorkerAPI, ContainerRegistryCredentials
from ....terra.azure.instance_config import TerraAzureSlimInstanceConfig


class TerraAzureWorkerAPI(CloudWorkerAPI):
    nameserver_ip = '168.63.129.16'

    # async because ClientSession must be created inside a running event loop
    @staticmethod
    def from_env() -> 'TerraAzureWorkerAPI':
        return TerraAzureWorkerAPI(
            httpx.client_session(),
            os.environ['WORKSPACE_STORAGE_CONTAINER_ID'],
            os.environ['WORKSPACE_STORAGE_CONTAINER_URL'],
            os.environ['WORKSPACE_ID'],
            os.environ['WORKSPACE_MANAGER_URL'],
        )

    def __init__(
        self,
        http_session: httpx.ClientSession,
        workspace_storage_container_id: str,
        workspace_storage_container_url: str,
        workspace_id: str,
        workspace_manager_url: str,
    ):
        self._http_session = http_session
        self.workspace_storage_container_id = workspace_storage_container_id
        self.workspace_storage_container_url = workspace_storage_container_url
        self.workspace_id = workspace_id
        self.workspace_manager_url = workspace_manager_url
        self._managed_identity_credentials = AzureCredentials.default_credentials()

    @property
    def cloud_specific_env_vars_for_user_jobs(self) -> List[str]:
        idp_json = orjson.dumps({'idp': IdentityProvider.MICROSOFT.value}).decode('utf-8')
        return [
            'HAIL_TERRA=1',
            'HAIL_LOCATION=external',  # There is no internal gateway, jobs must communicate over the internet
            f'HAIL_IDENTITY_PROVIDER_JSON={idp_json}',
            f'WORKSPACE_STORAGE_CONTAINER_ID={self.workspace_storage_container_id}',
            f'WORKSPACE_STORAGE_CONTAINER_URL={self.workspace_storage_container_url}',
            f'WORKSPACE_ID={self.workspace_id}',
            f'WORKSPACE_MANAGER_URL={self.workspace_manager_url}',
        ]

    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> CloudDisk:
        del instance_name, disk_name, size_in_gb, mount_path
        raise NotImplementedError

    def get_cloud_async_fs(self) -> AsyncFS:
        return TerraAzureAsyncFS()

    async def worker_container_registry_credentials(self, session: httpx.ClientSession) -> ContainerRegistryCredentials:
        return {}

    async def user_container_registry_credentials(self, credentials: Dict[str, str]) -> ContainerRegistryCredentials:
        return {}

    def create_metadata_server_app(self, credentials: Dict[str, str]) -> web.Application:
        raise NotImplementedError

    async def identity_uid(self, token: str) -> Optional[str]:
        # Terra Azure does not have custom OAuth clients for apps
        hail_oauth_config = {'appIdentifierUri': AzureCredentials.DEFAULT_SCOPE}
        return await AzureFlow.get_identity_uid_from_access_token(
            self._http_session, token, oauth2_client=hail_oauth_config
        )

    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> TerraAzureSlimInstanceConfig:
        return TerraAzureSlimInstanceConfig.from_dict(config_dict)

    async def extra_hail_headers(self) -> Dict[str, str]:
        token = await self._managed_identity_credentials.access_token()
        return {'Authorization': f'Bearer {token}'}

    async def _mount_cloudfuse(
        self,
        credentials: Dict[str, str],
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ):
        del credentials, mount_base_path_data, mount_base_path_tmp, config
        raise NotImplementedError

    async def unmount_cloudfuse(self, mount_base_path_data: str):
        raise NotImplementedError

    async def close(self):
        await self._http_session.close()
