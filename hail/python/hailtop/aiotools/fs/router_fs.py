from typing import Optional, List, Set, AsyncIterator
import asyncio
import urllib.parse

from ..stream import ReadableStream, WritableStream
from .fs import AsyncFS, MultiPartCreate, FileStatus, FileListEntry


class RouterAsyncFS(AsyncFS):
    def __init__(self, default_scheme: Optional[str], filesystems: List[AsyncFS]):
        scheme_fs = {}
        schemes = set()
        for fs in filesystems:
            for scheme in fs.schemes():
                if scheme not in schemes:
                    scheme_fs[scheme] = fs
                    schemes.add(scheme)

        if default_scheme is not None and default_scheme not in schemes:
            raise ValueError(f'default scheme {default_scheme} not in set of schemes: {", ".join(schemes)}')

        self._default_scheme = default_scheme
        self._filesystems = filesystems
        self._schemes = schemes
        self._scheme_fs = scheme_fs

    def schemes(self) -> Set[str]:
        return self._schemes

    def _get_fs(self, url):
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme:
            if self._default_scheme:
                parsed = parsed._replace(scheme=self._default_scheme)
            else:
                raise ValueError(f"no default scheme and URL has no scheme: {url}")

        fs = self._scheme_fs.get(parsed.scheme)
        if fs is None:
            raise ValueError(f"unknown scheme: {parsed.scheme}")

        return fs

    async def open(self, url: str) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open(url)

    async def open_from(self, url: str, start: int) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open_from(url, start)

    async def create(self, url: str, retry_writes: bool = True) -> WritableStream:
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

    async def listfiles(self, url: str, recursive: bool = False) -> AsyncIterator[FileListEntry]:
        fs = self._get_fs(url)
        return await fs.listfiles(url, recursive)

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

    async def rmtree(self, sema: Optional[asyncio.Semaphore], url: str) -> None:
        fs = self._get_fs(url)
        return await fs.rmtree(sema, url)

    async def close(self) -> None:
        for fs in self._filesystems:
            await fs.close()
