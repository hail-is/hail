import asyncio
import os
from typing import AsyncContextManager, AsyncIterator, Dict, Set, Tuple, Optional

from hailtop.aiocloud.aioazure import AzureAsyncFS, AzureAsyncFSURL
from hailtop.aiotools.fs import AsyncFS, ReadableStream, WritableStream, MultiPartCreate, FileStatus, FileListEntry
from hailtop.utils import time_msecs


from .client import TerraClient


WORKSPACE_STORAGE_CONTAINER_ID = os.environ.get('WORKSPACE_STORAGE_CONTAINER_ID')
WORKSPACE_STORAGE_CONTAINER_URL = os.environ.get('WORKSPACE_STORAGE_CONTAINER_URL')


class TerraAzureAsyncFS(AsyncFS):
    def __init__(self, **azure_kwargs):
        assert WORKSPACE_STORAGE_CONTAINER_URL is not None
        self._terra_client = TerraClient()
        self._azure_fs = AzureAsyncFS(**azure_kwargs)
        self._sas_token_cache: Dict[str, Tuple[str, int]] = {}

    @staticmethod
    def schemes() -> Set[str]:
        return {'https'}

    @staticmethod
    def enabled() -> bool:
        return WORKSPACE_STORAGE_CONTAINER_ID is not None and WORKSPACE_STORAGE_CONTAINER_URL is not None

    @staticmethod
    def valid_url(url: str) -> bool:
        return AzureAsyncFS.valid_url(url)

    @staticmethod
    def parse_url(url: str, *, error_if_bucket: bool = False) -> AzureAsyncFSURL:
        return AzureAsyncFS.parse_url(url, error_if_bucket=error_if_bucket)

    async def _to_azure_url(self, url: str) -> str:
        if url in self._sas_token_cache:
            sas_token, expiration = self._sas_token_cache[url]
            ten_minutes_from_now = time_msecs() + 10 * 60
            if expiration > ten_minutes_from_now:
                return sas_token

        assert WORKSPACE_STORAGE_CONTAINER_URL is not None
        if url.startswith(WORKSPACE_STORAGE_CONTAINER_URL):
            sas_token, expiration = await self._create_terra_sas_token(url)
            self._sas_token_cache[url] = (sas_token, expiration)
            return sas_token

        return url

    async def _create_terra_sas_token(self, url: str) -> Tuple[str, int]:
        blob_name = self.parse_url(url).path
        an_hour_in_seconds = 3600
        expiration = time_msecs() + an_hour_in_seconds * 1000

        assert WORKSPACE_STORAGE_CONTAINER_ID is not None
        sas_token = await self._terra_client.get_storage_container_sas_token(
            WORKSPACE_STORAGE_CONTAINER_ID, blob_name, expires_after=an_hour_in_seconds
        )

        return sas_token, expiration

    async def open(self, url: str) -> ReadableStream:
        return await self._azure_fs.open(await self._to_azure_url(url))

    async def _open_from(self, url: str, start: int, *, length: Optional[int] = None) -> ReadableStream:
        return await self._azure_fs._open_from(await self._to_azure_url(url), start, length=length)

    async def create(self, url: str, *, retry_writes: bool = True) -> AsyncContextManager[WritableStream]:
        return await self._azure_fs.create(
            await self._to_azure_url(url),
            retry_writes=retry_writes,
        )

    async def multi_part_create(self, sema: asyncio.Semaphore, url: str, num_parts: int) -> MultiPartCreate:
        return await self._azure_fs.multi_part_create(
            sema,
            await self._to_azure_url(url),
            num_parts,
        )

    async def mkdir(self, url: str) -> None:
        return await self._azure_fs.mkdir(await self._to_azure_url(url))

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        return await self._azure_fs.makedirs(
            await self._to_azure_url(url),
            exist_ok=exist_ok,
        )

    async def statfile(self, url: str) -> FileStatus:
        return await self._azure_fs.statfile(await self._to_azure_url(url))

    async def listfiles(
        self, url: str, recursive: bool = False, exclude_trailing_slash_files: bool = True
    ) -> AsyncIterator[FileListEntry]:
        return await self._azure_fs.listfiles(
            await self._to_azure_url(url),
            recursive,
            exclude_trailing_slash_files,
        )

    async def staturl(self, url: str) -> str:
        return await self._azure_fs.staturl(await self._to_azure_url(url))

    async def isfile(self, url: str) -> bool:
        return await self._azure_fs.isfile(await self._to_azure_url(url))

    async def isdir(self, url: str) -> bool:
        return await self._azure_fs.isdir(await self._to_azure_url(url))

    async def remove(self, url: str) -> None:
        return await self._azure_fs.remove(await self._to_azure_url(url))
