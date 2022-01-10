import aiohttp
import abc
import base64
from typing import Dict, Optional, Tuple
import os
import json

from hailtop import httpx
from hailtop.utils import request_retry_transient_errors, time_msecs

from ....worker.instance_env import CloudWorkerAPI
from .disk import AzureDisk, AzureDiskManager
from .credentials import AzureUserCredentials
from ..instance_config import AzureSlimInstanceConfig


class AzureWorkerAPI(CloudWorkerAPI):
    @staticmethod
    def from_env():
        machine_name = os.environ['NAME']
        subscription_id = os.environ['SUBSCRIPTION_ID']
        resource_group = os.environ['RESOURCE_GROUP']
        acr_url = os.environ['DOCKER_PREFIX']
        assert acr_url.endswith('azurecr.io'), acr_url
        instance_config = AzureSlimInstanceConfig.from_dict(
            json.loads(base64.b64decode(os.environ['INSTANCE_CONFIG']).decode())
        )

        disk_manager = AzureDiskManager(machine_name, instance_config, subscription_id, resource_group)

        return AzureWorkerAPI(subscription_id, resource_group, acr_url, disk_manager, instance_config)

    def __init__(self, subscription_id: str, resource_group: str, acr_url: str, disk_manager: AzureDiskManager, instance_config: AzureSlimInstanceConfig):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.acr_refresh_token = AcrRefreshToken(acr_url, AadAccessToken())
        self.disk_manager = disk_manager
        self.instance_config = instance_config

    @property
    def nameserver_ip(self):
        return '168.63.129.16'

    async def new_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> AzureDisk:
        return await self.disk_manager.new_disk(disk_name, instance_name, size_in_gb, mount_path)

    async def delete_disk(self, disk: AzureDisk):  # type: ignore[override]
        await self.disk_manager.delete_disk(disk)

    def user_credentials(self, credentials: Dict[str, bytes]) -> AzureUserCredentials:
        return AzureUserCredentials(credentials)

    async def worker_access_token(self, session: httpx.ClientSession) -> Dict[str, str]:
        # https://docs.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli#az-acr-login-with---expose-token
        return {
            'username': '00000000-0000-0000-0000-000000000000',
            'password': await self.acr_refresh_token.token(session),
        }

    async def close(self):
        await self.disk_manager.close()

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
