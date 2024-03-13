import os
from typing import Dict, Tuple

from azure.storage.blob.aio import BlobServiceClient

from hailtop.aiocloud.aioazure import AzureAsyncFS, AzureAsyncFSURL
from hailtop.utils import time_msecs

from .client import TerraClient

WORKSPACE_STORAGE_CONTAINER_ID = os.environ.get('WORKSPACE_STORAGE_CONTAINER_ID')
WORKSPACE_STORAGE_CONTAINER_URL = os.environ.get('WORKSPACE_STORAGE_CONTAINER_URL')


class TerraAzureAsyncFS(AzureAsyncFS):
    def __init__(self, **azure_kwargs):
        assert WORKSPACE_STORAGE_CONTAINER_URL is not None
        super().__init__(**azure_kwargs)
        self._terra_client = TerraClient()
        self._sas_token_cache: Dict[str, Tuple[AzureAsyncFSURL, int]] = {}
        self._workspace_container = AzureAsyncFS.parse_url(WORKSPACE_STORAGE_CONTAINER_URL)

    @staticmethod
    def enabled() -> bool:
        return WORKSPACE_STORAGE_CONTAINER_ID is not None and WORKSPACE_STORAGE_CONTAINER_URL is not None

    async def get_blob_service_client(self, url: AzureAsyncFSURL) -> BlobServiceClient:
        if self._in_workspace_container(url):
            return await super().get_blob_service_client(await self._get_terra_sas_token_url(url))
        return await super().get_blob_service_client(url)

    async def _get_terra_sas_token_url(self, url: AzureAsyncFSURL) -> AzureAsyncFSURL:
        if url.base in self._sas_token_cache:
            sas_token_url, expiration = self._sas_token_cache[url.base]
            ten_minutes_from_now = time_msecs() + 10 * 60
            if expiration > ten_minutes_from_now:
                return sas_token_url

        sas_token_url, expiration = await self._create_terra_sas_token(url)
        self._sas_token_cache[url.base] = (sas_token_url, expiration)
        return sas_token_url

    async def _create_terra_sas_token(self, url: AzureAsyncFSURL) -> Tuple[AzureAsyncFSURL, int]:
        an_hour_in_seconds = 3600
        expiration = time_msecs() + an_hour_in_seconds * 1000

        assert WORKSPACE_STORAGE_CONTAINER_ID is not None
        sas_token = await self._terra_client.get_storage_container_sas_token(
            WORKSPACE_STORAGE_CONTAINER_ID, url.path, expires_after=an_hour_in_seconds
        )

        return AzureAsyncFS.parse_url(sas_token), expiration

    def _in_workspace_container(self, url: AzureAsyncFSURL) -> bool:
        return url.account == self._workspace_container.account and url.container == self._workspace_container.container
