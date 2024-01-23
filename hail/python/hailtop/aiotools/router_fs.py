from typing import Any, Optional, List, Set, AsyncIterator, Dict, AsyncContextManager, Callable, ClassVar, Type
import asyncio
from contextlib import AsyncExitStack

from ..aiocloud import aioaws, aioazure, aiogoogle
from ..aiocloud.aioterra import azure as aioterra_azure
from .fs import AsyncFS, MultiPartCreate, FileStatus, FileListEntry, ReadableStream, WritableStream, AsyncFSURL
from .local_fs import LocalAsyncFS

from hailtop.config import ConfigVariable, configuration_of


class RouterAsyncFS(AsyncFS):
    FS_CLASSES: ClassVar[List[type[AsyncFS]]] = [
        LocalAsyncFS,
        aiogoogle.GoogleStorageAsyncFS,
        aioterra_azure.TerraAzureAsyncFS,  # Must precede Azure since Terra URLs are also valid Azure URLs
        aioazure.AzureAsyncFS,
        aioaws.S3AsyncFS,
    ]

    def __init__(
        self,
        *,
        local_kwargs: Optional[Dict[str, Any]] = None,
        gcs_kwargs: Optional[Dict[str, Any]] = None,
        azure_kwargs: Optional[Dict[str, Any]] = None,
        s3_kwargs: Optional[Dict[str, Any]] = None,
        gcs_bucket_allow_list: Optional[List[str]] = None,
    ):
        self._local_fs: Optional[LocalAsyncFS] = None
        self._google_fs: Optional[aiogoogle.GoogleStorageAsyncFS] = None
        self._terra_azure_fs: Optional[aioterra_azure.TerraAzureAsyncFS] = None
        self._azure_fs: Optional[aioazure.AzureAsyncFS] = None
        self._s3_fs: Optional[aioaws.S3AsyncFS] = None
        self._exit_stack = AsyncExitStack()

        self._local_kwargs = local_kwargs or {}
        self._gcs_kwargs = gcs_kwargs or {}
        self._azure_kwargs = azure_kwargs or {}
        self._s3_kwargs = s3_kwargs or {}
        self._gcs_bucket_allow_list = (
            gcs_bucket_allow_list
            if gcs_bucket_allow_list is not None
            else configuration_of(ConfigVariable.GCS_BUCKET_ALLOW_LIST, None, fallback="").split(",")
        )

    @staticmethod
    def schemes() -> Set[str]:
        return {scheme for fs_class in RouterAsyncFS.FS_CLASSES for scheme in fs_class.schemes()}

    @staticmethod
    def copy_part_size(url: str) -> int:
        klass = RouterAsyncFS._fs_class(url)
        return klass.copy_part_size(url)

    @staticmethod
    def parse_url(url: str) -> AsyncFSURL:
        klass = RouterAsyncFS._fs_class(url)
        return klass.parse_url(url)

    @staticmethod
    def _fs_class(url: str) -> Type[AsyncFS]:
        for klass in RouterAsyncFS.FS_CLASSES:
            if klass.valid_url(url):
                return klass
        raise ValueError(f'no file system found for url {url}')

    @staticmethod
    def valid_url(url) -> bool:
        return (
            LocalAsyncFS.valid_url(url)
            or aiogoogle.GoogleStorageAsyncFS.valid_url(url)
            or aioterra_azure.TerraAzureAsyncFS.valid_url(url)
            or aioazure.AzureAsyncFS.valid_url(url)
            or aioaws.S3AsyncFS.valid_url(url)
        )

    async def _get_fs(self, url: str):
        if LocalAsyncFS.valid_url(url):
            if self._local_fs is None:
                self._local_fs = LocalAsyncFS(**self._local_kwargs)
                self._exit_stack.push_async_callback(self._local_fs.close)
            return self._local_fs
        if aiogoogle.GoogleStorageAsyncFS.valid_url(url):
            if self._google_fs is None:
                self._google_fs = aiogoogle.GoogleStorageAsyncFS(
                    **self._gcs_kwargs, bucket_allow_list=self._gcs_bucket_allow_list.copy()
                )
                self._exit_stack.push_async_callback(self._google_fs.close)
            return self._google_fs
        if aioterra_azure.TerraAzureAsyncFS.enabled() and aioterra_azure.TerraAzureAsyncFS.valid_url(url):
            if self._terra_azure_fs is None:
                self._terra_azure_fs = aioterra_azure.TerraAzureAsyncFS(**self._azure_kwargs)
                self._exit_stack.push_async_callback(self._terra_azure_fs.close)
            return self._terra_azure_fs
        if aioazure.AzureAsyncFS.valid_url(url):
            if self._azure_fs is None:
                self._azure_fs = aioazure.AzureAsyncFS(**self._azure_kwargs)
                self._exit_stack.push_async_callback(self._azure_fs.close)
            return self._azure_fs
        if aioaws.S3AsyncFS.valid_url(url):
            if self._s3_fs is None:
                self._s3_fs = aioaws.S3AsyncFS(**self._s3_kwargs)
                self._exit_stack.push_async_callback(self._s3_fs.close)
            return self._s3_fs
        raise ValueError(f'no file system found for url {url}')

    async def open(self, url: str) -> ReadableStream:
        fs = await self._get_fs(url)
        return await fs.open(url)

    async def _open_from(self, url: str, start: int, *, length: Optional[int] = None) -> ReadableStream:
        fs = await self._get_fs(url)
        return await fs.open_from(url, start, length=length)

    async def create(self, url: str, *, retry_writes: bool = True) -> AsyncContextManager[WritableStream]:
        fs = await self._get_fs(url)
        return await fs.create(url, retry_writes=retry_writes)

    async def multi_part_create(self, sema: asyncio.Semaphore, url: str, num_parts: int) -> MultiPartCreate:
        fs = await self._get_fs(url)
        return await fs.multi_part_create(sema, url, num_parts)

    async def statfile(self, url: str) -> FileStatus:
        fs = await self._get_fs(url)
        return await fs.statfile(url)

    async def listfiles(
        self, url: str, recursive: bool = False, exclude_trailing_slash_files: bool = True
    ) -> AsyncIterator[FileListEntry]:
        fs = await self._get_fs(url)
        return await fs.listfiles(url, recursive, exclude_trailing_slash_files)

    async def staturl(self, url: str) -> str:
        fs = await self._get_fs(url)
        return await fs.staturl(url)

    async def mkdir(self, url: str) -> None:
        fs = await self._get_fs(url)
        return await fs.mkdir(url)

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        fs = await self._get_fs(url)
        return await fs.makedirs(url, exist_ok=exist_ok)

    async def isfile(self, url: str) -> bool:
        fs = await self._get_fs(url)
        return await fs.isfile(url)

    async def isdir(self, url: str) -> bool:
        fs = await self._get_fs(url)
        return await fs.isdir(url)

    async def remove(self, url: str) -> None:
        fs = await self._get_fs(url)
        return await fs.remove(url)

    async def rmtree(
        self, sema: Optional[asyncio.Semaphore], url: str, listener: Optional[Callable[[int], None]] = None
    ) -> None:
        fs = await self._get_fs(url)
        return await fs.rmtree(sema, url, listener)

    async def close(self) -> None:
        await self._exit_stack.aclose()
