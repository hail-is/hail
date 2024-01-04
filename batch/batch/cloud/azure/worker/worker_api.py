import abc
import base64
import os
import tempfile
from typing import Dict, List, Optional, Tuple, cast

import aiohttp
import orjson
from azure.identity.aio import ClientSecretCredential

from hailtop import httpx
from hailtop.aiocloud import aioazure
from hailtop.auth.auth import IdentityProvider
from hailtop.utils import check_exec_output, retry_transient_errors, time_msecs

from ....worker.worker_api import CloudWorkerAPI, ContainerRegistryCredentials
from ..instance_config import AzureSlimInstanceConfig
from .disk import AzureDisk


class AzureWorkerAPI(CloudWorkerAPI[aioazure.AzureCredentials]):
    nameserver_ip = '168.63.129.16'

    @staticmethod
    def from_env():
        subscription_id = os.environ['SUBSCRIPTION_ID']
        resource_group = os.environ['RESOURCE_GROUP']
        acr_url = os.environ['DOCKER_PREFIX']
        hail_oauth_scope = os.environ['HAIL_AZURE_OAUTH_SCOPE']
        assert acr_url.endswith('azurecr.io'), acr_url
        return AzureWorkerAPI(subscription_id, resource_group, acr_url, hail_oauth_scope)

    def __init__(self, subscription_id: str, resource_group: str, acr_url: str, hail_oauth_scope: str):
        super().__init__()
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.hail_oauth_scope = hail_oauth_scope
        self.azure_credentials = aioazure.AzureCredentials.default_credentials()
        self.acr_refresh_token = AcrRefreshToken(acr_url, self.azure_credentials)
        self._blobfuse_credential_files: Dict[str, str] = {}

    @property
    def cloud_specific_env_vars_for_user_jobs(self) -> List[str]:
        idp_json = orjson.dumps({"idp": IdentityProvider.MICROSOFT.value}).decode('utf-8')
        return [
            f'HAIL_AZURE_OAUTH_SCOPE={self.hail_oauth_scope}',
            'AZURE_APPLICATION_CREDENTIALS=/azure-credentials/key.json',
            f'HAIL_IDENTITY_PROVIDER_JSON={idp_json}',
        ]

    def create_disk(self, instance_name: str, disk_name: str, size_in_gb: int, mount_path: str) -> AzureDisk:
        return AzureDisk(disk_name, instance_name, size_in_gb, mount_path)

    def get_cloud_async_fs(self) -> aioazure.AzureAsyncFS:
        return aioazure.AzureAsyncFS(credentials=self.azure_credentials)

    def _load_user_credentials(self, credentials: Dict[str, str]) -> aioazure.AzureCredentials:
        sp_credentials = orjson.loads(base64.b64decode(credentials['key.json']).decode())
        return aioazure.AzureCredentials.from_credentials_data(sp_credentials)

    def _get_user_hail_identity(self, credentials: Dict[str, str]) -> str:
        sp_credentials = orjson.loads(base64.b64decode(credentials['key.json']).decode())
        return sp_credentials['appId']

    async def worker_container_registry_credentials(self, session: httpx.ClientSession) -> ContainerRegistryCredentials:
        # https://docs.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli#az-acr-login-with---expose-token
        return {
            'username': '00000000-0000-0000-0000-000000000000',
            'password': await self.acr_refresh_token.token(session),
        }

    def _user_service_principal_client_id_secret_tenant(self, hail_identity: str) -> Tuple[str, str, str]:
        credential = self._user_credentials[hail_identity].credential

        assert isinstance(credential, ClientSecretCredential)
        return credential._client_id, cast(str, credential._secret), credential._client._tenant_id

    async def user_container_registry_credentials(self, hail_identity: str) -> ContainerRegistryCredentials:
        username, password, _ = self._user_service_principal_client_id_secret_tenant(hail_identity)
        return {'username': username, 'password': password}

    def instance_config_from_config_dict(self, config_dict: Dict[str, str]) -> AzureSlimInstanceConfig:
        return AzureSlimInstanceConfig.from_dict(config_dict)

    def _write_blobfuse_credentials(
        self,
        hail_identity: str,
        account: str,
        container: str,
        mount_base_path_data: str,
    ) -> str:
        if mount_base_path_data not in self._blobfuse_credential_files:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as credsfile:
                credsfile.write(self.blobfuse_credentials(hail_identity, account, container))
                self._blobfuse_credential_files[mount_base_path_data] = credsfile.name
        return self._blobfuse_credential_files[mount_base_path_data]

    async def _mount_cloudfuse(
        self,
        hail_identity: str,
        mount_base_path_data: str,
        mount_base_path_tmp: str,
        config: dict,
    ):
        # https://docs.microsoft.com/en-us/azure/storage/blobs/storage-how-to-mount-container-linux#mount
        bucket = config['bucket']
        account, container = bucket.split('/', maxsplit=1)
        assert account and container

        fuse_credentials_path = self._write_blobfuse_credentials(
            hail_identity, account, container, mount_base_path_data
        )

        options = ['allow_other']
        if config['read_only']:
            options.append('ro')

        await check_exec_output(
            'blobfuse2',
            'mountv1',
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

    def blobfuse_credentials(self, hail_identity: str, account: str, container: str) -> str:
        client_id, client_secret, tenant_id = self._user_service_principal_client_id_secret_tenant(hail_identity)
        # https://github.com/Azure/azure-storage-fuse
        return f'''
accountName {account}
authType SPN
servicePrincipalClientId {client_id}
servicePrincipalClientSecret {client_secret}
servicePrincipalTenantId {tenant_id}
containerName {container}
'''

    async def close(self):
        pass

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


class AcrRefreshToken(LazyShortLivedToken):
    def __init__(self, acr_url: str, credentials: aioazure.AzureCredentials):
        super().__init__()
        self.acr_url: str = acr_url
        self.credentials = credentials

    async def _fetch(self, session: httpx.ClientSession) -> Tuple[str, int]:
        # https://github.com/Azure/acr/blob/main/docs/AAD-OAuth.md#calling-post-oauth2exchange-to-get-an-acr-refresh-token
        data = {
            'grant_type': 'access_token',
            'service': self.acr_url,
            'access_token': await self.credentials.access_token(),
        }
        resp_json = await retry_transient_errors(
            session.post_read_json,
            f'https://{self.acr_url}/oauth2/exchange',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data=data,
            timeout=aiohttp.ClientTimeout(total=60),  # type: ignore
        )
        refresh_token: str = resp_json['refresh_token']
        expiration_time_ms = time_msecs() + 60 * 60 * 1000  # token expires in 3 hours so we refresh after 1 hour
        return refresh_token, expiration_time_ms
