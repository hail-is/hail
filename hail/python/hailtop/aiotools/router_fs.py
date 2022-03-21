from typing import Any, Optional, List, Set, AsyncIterator, Dict, AsyncContextManager, Callable
import asyncio
import urllib.parse

from ..aiocloud import aioaws, aioazure, aiogoogle
from .fs import AsyncFS, MultiPartCreate, FileStatus, FileListEntry, ReadableStream, WritableStream
from .local_fs import LocalAsyncFS


class RouterAsyncFS(AsyncFS):
    def __init__(self,
                 default_scheme: Optional[str],
                 *,
                 filesystems: Optional[List[AsyncFS]] = None,
                 local_kwargs: Optional[Dict[str, Any]] = None,
                 gcs_kwargs: Optional[Dict[str, Any]] = None,
                 azure_kwargs: Optional[Dict[str, Any]] = None,
                 s3_kwargs: Optional[Dict[str, Any]] = None):
        scheme_fs: Dict[str, AsyncFS] = {}

        filesystems = [] if filesystems is None else filesystems

        for fs in filesystems:
            for scheme in fs.schemes:
                if scheme not in scheme_fs:
                    scheme_fs[scheme] = fs

        self._default_scheme = default_scheme
        self._filesystems = filesystems
        self._scheme_fs = scheme_fs

        self._local_kwargs = local_kwargs or {}
        self._gcs_kwargs = gcs_kwargs or {}
        self._azure_kwargs = azure_kwargs or {}
        self._s3_kwargs = s3_kwargs or {}

    @property
    def schemes(self) -> Set[str]:
        return set(self._scheme_fs.keys())

    def _load_fs(self, scheme: str):
        fs: AsyncFS

        if scheme in LocalAsyncFS.schemes:
            fs = LocalAsyncFS(**self._local_kwargs)
        elif scheme in aiogoogle.GoogleStorageAsyncFS.schemes:
            fs = aiogoogle.GoogleStorageAsyncFS(**self._gcs_kwargs)
        elif scheme in aioazure.AzureAsyncFS.schemes:
            fs = aioazure.AzureAsyncFS(**self._azure_kwargs)
        elif scheme in aioaws.S3AsyncFS.schemes:
            fs = aioaws.S3AsyncFS(**self._s3_kwargs)
        else:
            raise ValueError(f'no file system found for scheme {scheme}')

        self._scheme_fs[scheme] = fs
        self._filesystems.append(fs)

    def _get_fs(self, url: str) -> AsyncFS:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme:
            if self._default_scheme:
                parsed = parsed._replace(scheme=self._default_scheme)
            else:
                raise ValueError(f"no default scheme and URL has no scheme: {url}")

        scheme = parsed.scheme
        if scheme not in self._scheme_fs:
            self._load_fs(scheme)

        fs = self._scheme_fs.get(parsed.scheme)
        assert fs is not None
        return fs

    async def open(self, url: str) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open(url)

    async def open_from(self, url: str, start: int) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open_from(url, start)

    async def create(self, url: str, retry_writes: bool = True) -> AsyncContextManager[WritableStream]:
        fs = self._get_fs(url)
        return await fs.create(url, retry_writes=retry_writes)

    async def multi_part_create(
            self,
            sema: asyncio.Semaphore,
            url: str,
            num_parts: int) -> MultiPartCreate:
        fs = self._get_fs(url)
        return await fs.multi_part_create(sema, url, num_parts)

    async def statfile(self, url: str) -> FileStatus:
        fs = self._get_fs(url)
        return await fs.statfile(url)

    async def listfiles(self,
                        url: str,
                        recursive: bool = False,
                        exclude_trailing_slash_files: bool = True
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

    async def rmtree(self,
                     sema: Optional[asyncio.Semaphore],
                     url: str,
                     listener: Optional[Callable[[int], None]] = None) -> None:
        fs = self._get_fs(url)
        return await fs.rmtree(sema, url, listener)

    async def close(self) -> None:
        for fs in self._filesystems:
            await fs.close()

    def copy_part_size(self, url: str) -> int:
        fs = self._get_fs(url)
        return fs.copy_part_size(url)
