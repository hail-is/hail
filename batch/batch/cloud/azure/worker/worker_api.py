import abc
import os
import tempfile
from typing import Dict, Optional, Tuple

import aiohttp

from gear.cloud_config import get_azure_config
from hailtop import httpx
from hailtop.aiocloud import aioazure
from hailtop.utils import check_exec_output, request_retry_transient_errors, time_msecs

from ....worker.worker_api import CloudWorkerAPI
from ..instance_config import AzureSlimInstanceConfig
from .credentials import AzureUserCredentials
from .disk import AzureDisk


class AzureWorkerAPI(CloudWorkerAPI[AzureUserCredentials]):
    nameserver_ip = '168.63.129.16'

    @staticmethod
    def from_env():
        subscription_id = os.environ['SUBSCRIPTION_ID']
        resource_group = os.environ['RESOURCE_GROUP']
        acr_url = os.environ['DOCKER_PREFIX']
        assert acr_url.endswith('azurecr.io'), acr_url
        return AzureWorkerAPI(subscription_id, resource_group, acr_url)

    def __init__(self, subscription_id: str, resource_group: str, acr_url: str):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.acr_refresh_token = AcrRefreshToken(acr_url, AadAccessToken())
        self.azure_credentials = aioazure.AzureCredentials.default_credentials()
        self._blobfuse_credential_files: Dict[str, str] = {}

    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> AzureDisk:
        return AzureDisk(disk_name, instance_name, size_in_gb, mount_path)

    def get_cloud_async_fs(self) -> aioazure.AzureAsyncFS:
        return aioazure.AzureAsyncFS(credentials=self.azure_credentials)

    def get_compute_client(self) -> aioazure.AzureComputeClient:
        azure_config = get_azure_config()
        return aioazure.AzureComputeClient(azure_config.subscription_id, azure_config.resource_group)

    def user_credentials(self, credentials: Dict[str, str]) -> AzureUserCredentials:
        return AzureUserCredentials(credentials)

    async def worker_access_token(self, session: httpx.ClientSession) -> Dict[str, str]:
        # https://docs.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli#az-acr-login-with---expose-token
        return {
            'username': '00000000-0000-0000-0000-000000000000',
            'password': await self.acr_refresh_token.token(session),
        }

    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> AzureSlimInstanceConfig:
        return AzureSlimInstanceConfig.from_dict(config_dict)

    def _write_blobfuse_credentials(
        self,
        credentials: AzureUserCredentials,
        account: str,
        container: str,
        mount_base_path_data: str,
    ) -> str:
        if mount_base_path_data not in self._blobfuse_credential_files:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as credsfile:
                credsfile.write(credentials.blobfuse_credentials(account, container))
                self._blobfuse_credential_files[mount_base_path_data] = credsfile.name
        return self._blobfuse_credential_files[mount_base_path_data]

    async def _mount_cloudfuse(
        self,
        credentials: AzureUserCredentials,
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ):
        # https://docs.microsoft.com/en-us/azure/storage/blobs/storage-how-to-mount-container-linux#mount
        bucket = config['bucket']
        account, container = bucket.split('/', maxsplit=1)
        assert account and container

        fuse_credentials_path = self._write_blobfuse_credentials(credentials, account, container, mount_base_path_data)

        options = ['allow_other']
        if config['read_only']:
            options.append('ro')

        await check_exec_output(
            'blobfuse',
            mount_base_path_data,
            f'--tmp-path={mount_base_path_tmp}',
            f'--config-file={fuse_credentials_path}',
            '--pre-mount-validate=true',
            '-o',
            ','.join(options),
            '-o',
            'attr_timeout=240',
            '-o',
            'entry_timeout=240',
            '-o',
            'negative_timeout=120',
        )

    async def unmount_cloudfuse(self, mount_base_path_data: str):
        try:
            # blobfuse cleans up the temporary directory when unmounting
            await check_exec_output('fusermount', '-u', mount_base_path_data)
        finally:
            os.remove(self._blobfuse_credential_files[mount_base_path_data])
            del self._blobfuse_credential_files[mount_base_path_data]

    def __str__(self):
        return f'subscription_id={self.subscription_id} resource_group={self.resource_group}'


class LazyShortLivedToken(abc.ABC):
    def __init__(self):
        self._token: Optional[str] = None
        self._expiration_time_ms: Optional[int] = None

    async def token(self, session: httpx.ClientSession) -> str:
        now = time_msecs()
        if not self._expiration_time_ms or now >= self._expiration_time_ms:
            self._token, self._expiration_time_ms = await self._fetch(session)
        assert self._token
        return self._token

    @abc.abstractmethod
    async def _fetch(self, session: httpx.ClientSession) -> Tuple[str, int]:
        raise NotImplementedError()


class AadAccessToken(LazyShortLivedToken):
    async def _fetch(self, session: httpx.ClientSession) -> Tuple[str, int]:
        # https://docs.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/how-to-use-vm-token#get-a-token-using-http
        params = {'api-version': '2018-02-01', 'resource': 'https://management.azure.com/'}
        async with await request_retry_transient_errors(
            session,
            'GET',
            'http://169.254.169.254/metadata/identity/oauth2/token',
            headers={'Metadata': 'true'},
            params=params,
            timeout=aiohttp.ClientTimeout(total=60),  # type: ignore
        ) as resp:
            resp_json = await resp.json()
            access_token: str = resp_json['access_token']
            expiration_time_ms = int(resp_json['expires_on']) * 1000
            return access_token, expiration_time_ms


class AcrRefreshToken(LazyShortLivedToken):
    def __init__(self, acr_url: str, aad_access_token: AadAccessToken):
        super().__init__()
        self.acr_url: str = acr_url
        self.aad_access_token: AadAccessToken = aad_access_token

    async def _fetch(self, session: httpx.ClientSession) -> Tuple[str, int]:
        # https://github.com/Azure/acr/blob/main/docs/AAD-OAuth.md#calling-post-oauth2exchange-to-get-an-acr-refresh-token
        data = {
            'grant_type': 'access_token',
            'service': self.acr_url,
            'access_token': await self.aad_access_token.token(session),
        }
        async with await request_retry_transient_errors(
            session,
            'POST',
            f'https://{self.acr_url}/oauth2/exchange',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data=data,
            timeout=aiohttp.ClientTimeout(total=60),  # type: ignore
        ) as resp:
            refresh_token: str = (await resp.json())['refresh_token']
            expiration_time_ms = time_msecs() + 60 * 60 * 1000  # token expires in 3 hours so we refresh after 1 hour
            return refresh_token, expiration_time_ms
