from typing import Any, Optional, List, Set, AsyncIterator, Dict, AsyncContextManager, Callable
import asyncio

from ..aiocloud import aioaws, aioazure, aiogoogle
from .fs import AsyncFS, MultiPartCreate, FileStatus, FileListEntry, ReadableStream, WritableStream, AsyncFSURL
from .local_fs import LocalAsyncFS

from hailtop.config import ConfigVariable, configuration_of


class RouterAsyncFS(AsyncFS):
    def __init__(
        self,
        *,
        filesystems: Optional[List[AsyncFS]] = None,
        local_kwargs: Optional[Dict[str, Any]] = None,
        gcs_kwargs: Optional[Dict[str, Any]] = None,
        azure_kwargs: Optional[Dict[str, Any]] = None,
        s3_kwargs: Optional[Dict[str, Any]] = None,
        gcs_bucket_allow_list: Optional[List[str]] = None,
    ):
        self._filesystems = [] if filesystems is None else filesystems
        self._local_kwargs = local_kwargs or {}
        self._gcs_kwargs = gcs_kwargs or {}
        self._azure_kwargs = azure_kwargs or {}
        self._s3_kwargs = s3_kwargs or {}
        self._gcs_bucket_allow_list = (
            gcs_bucket_allow_list
            if gcs_bucket_allow_list is not None
            else configuration_of(ConfigVariable.GCS_BUCKET_ALLOW_LIST, None, fallback="").split(",")
        )

    def parse_url(self, url: str) -> AsyncFSURL:
        return self._get_fs(url).parse_url(url)

    @property
    def schemes(self) -> Set[str]:
        return set().union(*(fs.schemes for fs in self._filesystems))

    @staticmethod
    def valid_url(url) -> bool:
        return (
            LocalAsyncFS.valid_url(url)
            or aiogoogle.GoogleStorageAsyncFS.valid_url(url)
            or aioazure.AzureAsyncFS.valid_url(url)
            or aioaws.S3AsyncFS.valid_url(url)
        )

    def _load_fs(self, uri: str):
        fs: AsyncFS

        if LocalAsyncFS.valid_url(uri):
            fs = LocalAsyncFS(**self._local_kwargs)
        elif aiogoogle.GoogleStorageAsyncFS.valid_url(uri):
            fs = aiogoogle.GoogleStorageAsyncFS(
                **self._gcs_kwargs, bucket_allow_list=self._gcs_bucket_allow_list.copy()
            )
        elif aioazure.AzureAsyncFS.valid_url(uri):
            fs = aioazure.AzureAsyncFS(**self._azure_kwargs)
        elif aioaws.S3AsyncFS.valid_url(uri):
            fs = aioaws.S3AsyncFS(**self._s3_kwargs)
        else:
            raise ValueError(f'no file system found for url {uri}')

        self._filesystems.append(fs)
        return fs

    def _get_fs(self, uri: str) -> AsyncFS:
        for fs in self._filesystems:
            if fs.valid_url(uri):
                return fs
        return self._load_fs(uri)

    async def open(self, url: str) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open(url)

    async def _open_from(self, url: str, start: int, *, length: Optional[int] = None) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open_from(url, start, length=length)

    async def create(self, url: str, retry_writes: bool = True) -> AsyncContextManager[WritableStream]:
        fs = self._get_fs(url)
        return await fs.create(url, retry_writes=retry_writes)

    async def multi_part_create(self, sema: asyncio.Semaphore, url: str, num_parts: int) -> MultiPartCreate:
        fs = self._get_fs(url)
        return await fs.multi_part_create(sema, url, num_parts)

    async def statfile(self, url: str) -> FileStatus:
        fs = self._get_fs(url)
        return await fs.statfile(url)

    async def listfiles(
        self, url: str, recursive: bool = False, exclude_trailing_slash_files: bool = True
    ) -> AsyncIterator[FileListEntry]:
        fs = self._get_fs(url)
        return await fs.listfiles(url, recursive, exclude_trailing_slash_files)

    async def staturl(self, url: str) -> str:
        fs = self._get_fs(url)
        return await fs.staturl(url)

    async def mkdir(self, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.mkdir(url)

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        fs = self._get_fs(url)
        return await fs.makedirs(url, exist_ok=exist_ok)

    async def isfile(self, url: str) -> bool:
        fs = self._get_fs(url)
        return await fs.isfile(url)

    async def isdir(self, url: str) -> bool:
        fs = self._get_fs(url)
        return await fs.isdir(url)

    async def remove(self, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.remove(url)

    async def rmtree(
        self, sema: Optional[asyncio.Semaphore], url: str, listener: Optional[Callable[[int], None]] = None
    ) -> None:
        fs = self._get_fs(url)
        return await fs.rmtree(sema, url, listener)

    async def close(self) -> None:
        for fs in self._filesystems:
            await fs.close()

    def copy_part_size(self, url: str) -> int:
        fs = self._get_fs(url)
        return fs.copy_part_size(url)
