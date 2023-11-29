import os
from typing import Dict, List

from hailtop import httpx
from hailtop.aiocloud.aioazure import AzureCredentials
from hailtop.aiocloud.aioterra.azure import TerraAzureAsyncFS
from hailtop.aiotools.fs import AsyncFS
from hailtop.auth.auth import IdentityProvider

from .....worker.credentials import CloudUserCredentials
from .....worker.disk import CloudDisk
from .....worker.worker_api import CloudWorkerAPI, ContainerRegistryCredentials
from ....terra.azure.instance_config import TerraAzureSlimInstanceConfig


class AzureManagedIdentityCredentials(CloudUserCredentials):
    @property
    def cloud_env_name(self) -> str:
        return 'HAIL_IGNORE'

    @property
    def mount_path(self) -> str:
        return ''

    @property
    def identity_provider_json(self):
        return {'idp': IdentityProvider.MICROSOFT.value}


class TerraAzureWorkerAPI(CloudWorkerAPI[AzureManagedIdentityCredentials]):
    nameserver_ip = '168.63.129.16'

    @staticmethod
    def from_env() -> 'TerraAzureWorkerAPI':
        return TerraAzureWorkerAPI(
            os.environ['WORKSPACE_STORAGE_CONTAINER_ID'],
            os.environ['WORKSPACE_STORAGE_CONTAINER_URL'],
            os.environ['WORKSPACE_ID'],
            os.environ['WORKSPACE_MANAGER_URL'],
        )

    def __init__(
        self,
        workspace_storage_container_id: str,
        workspace_storage_container_url: str,
        workspace_id: str,
        workspace_manager_url: str,
    ):
        self.workspace_storage_container_id = workspace_storage_container_id
        self.workspace_storage_container_url = workspace_storage_container_url
        self.workspace_id = workspace_id
        self.workspace_manager_url = workspace_manager_url
        self._managed_identity_credentials = AzureCredentials.default_credentials()

    @property
    def cloud_specific_env_vars_for_user_jobs(self) -> List[str]:
        return [
            'HAIL_TERRA=1',
            f'WORKSPACE_STORAGE_CONTAINER_ID={self.workspace_storage_container_id}',
            f'WORKSPACE_STORAGE_CONTAINER_URL={self.workspace_storage_container_url}',
            f'WORKSPACE_ID={self.workspace_id}',
            f'WORKSPACE_MANAGER_URL={self.workspace_manager_url}',
        ]

    def create_disk(self, *_) -> CloudDisk:
        raise NotImplementedError

    def get_cloud_async_fs(self) -> AsyncFS:
        return TerraAzureAsyncFS()

    def user_credentials(self, _: Dict[str, str]) -> AzureManagedIdentityCredentials:
        return AzureManagedIdentityCredentials()

    async def worker_container_registry_credentials(self, _: httpx.ClientSession) -> ContainerRegistryCredentials:
        return {}

    async def user_container_registry_credentials(
        self, _: AzureManagedIdentityCredentials
    ) -> ContainerRegistryCredentials:
        return {}

    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> TerraAzureSlimInstanceConfig:
        return TerraAzureSlimInstanceConfig.from_dict(config_dict)

    async def extra_hail_headers(self) -> Dict[str, str]:
        token = await self._managed_identity_credentials.access_token()
        return {'Authorization': f'Bearer {token}'}

    async def _mount_cloudfuse(self, *_):
        raise NotImplementedError

    async def unmount_cloudfuse(self, _: str):
        raise NotImplementedError

    async def close(self):
        pass
