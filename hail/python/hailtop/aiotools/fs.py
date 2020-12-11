from typing import TypeVar, Any, Optional, List, Type, BinaryIO, cast, Set, AsyncIterator
from types import TracebackType
import abc
import os
import os.path
import stat
import shutil
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
from hailtop.utils import blocking_to_async
from .stream import ReadableStream, WritableStream, blocking_readable_stream_to_async, blocking_writable_stream_to_async

AsyncFSType = TypeVar('AsyncFSType', bound='AsyncFS')


class FileStatus(abc.ABC):
    @abc.abstractmethod
    async def size(self) -> int:
        pass

    @abc.abstractmethod
    async def __getitem__(self, key: str) -> Any:
        pass


class FileListEntry(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    async def url(self) -> str:
        pass

    @abc.abstractmethod
    async def is_file(self) -> bool:
        pass

    @abc.abstractmethod
    async def is_dir(self) -> bool:
        pass

    @abc.abstractmethod
    async def status(self) -> FileStatus:
        pass


class AsyncFS(abc.ABC):
    @abc.abstractmethod
    def schemes(self) -> Set[str]:
        pass

    @abc.abstractmethod
    async def open(self, url: str) -> ReadableStream:
        pass

    @abc.abstractmethod
    async def create(self, url: str) -> WritableStream:
        pass

    @abc.abstractmethod
    async def mkdir(self, url: str) -> None:
        pass

    @abc.abstractmethod
    async def statfile(self, url: str) -> FileStatus:
        pass

    @abc.abstractmethod
    async def listfiles(self, url: str, recursive: bool = False) -> AsyncIterator[FileListEntry]:
        pass

    @abc.abstractmethod
    async def isfile(self, url: str) -> bool:
        pass

    @abc.abstractmethod
    async def isdir(self, url: str) -> bool:
        pass

    @abc.abstractmethod
    async def remove(self, url: str) -> None:
        pass

    @abc.abstractmethod
    async def rmtree(self, url: str) -> None:
        pass

    async def touch(self, url: str) -> None:
        async with await self.create(url):
            pass

    async def close(self) -> None:
        pass

    async def __aenter__(self: AsyncFSType) -> AsyncFSType:
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self.close()


class LocalStatFileStatus(FileStatus):
    def __init__(self, stat_result):
        self._stat_result = stat_result
        self._items = None

    async def size(self) -> int:
        return self._stat_result.st_size

    async def __getitem__(self, key: str) -> Any:
        raise KeyError(key)


class LocalFileListEntry(FileListEntry):
    def __init__(self, thread_pool, base_url, entry):
        assert '/' not in entry.name
        self._thread_pool = thread_pool
        if not base_url.endswith('/'):
            base_url = f'{base_url}/'
        self._base_url = base_url
        self._entry = entry
        self._status = None

    def name(self) -> str:
        return self._entry.name

    async def url(self) -> str:
        trailing_slash = "/" if await self.is_dir() else ""
        return f'{self._base_url}{self._entry.name}{trailing_slash}'

    async def is_file(self) -> bool:
        return not await self.is_dir()

    async def is_dir(self) -> bool:
        return await blocking_to_async(self._thread_pool, self._entry.is_dir)

    async def status(self) -> LocalStatFileStatus:
        if self._status is None:
            if await self.is_dir():
                raise ValueError("directory has no file status")
            self._status = LocalStatFileStatus(await blocking_to_async(self._thread_pool, self._entry.stat))
        return self._status


class LocalAsyncFS(AsyncFS):
    def __init__(self, thread_pool: ThreadPoolExecutor, max_workers=None):
        if not thread_pool:
            thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._thread_pool = thread_pool

    def schemes(self) -> Set[str]:
        return {'file'}

    @staticmethod
    def _get_path(url):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme and parsed.scheme != 'file':
            raise ValueError(f"invalid scheme, expected file: {parsed.scheme}")
        return parsed.path

    async def open(self, url: str) -> ReadableStream:
        return blocking_readable_stream_to_async(self._thread_pool, cast(BinaryIO, open(self._get_path(url), 'rb')))

    async def create(self, url: str) -> WritableStream:
        return blocking_writable_stream_to_async(self._thread_pool, cast(BinaryIO, open(self._get_path(url), 'wb')))

    async def statfile(self, url: str) -> LocalStatFileStatus:
        path = self._get_path(url)
        stat_result = await blocking_to_async(self._thread_pool, os.stat, path)
        if stat.S_ISDIR(stat_result.st_mode):
            raise FileNotFoundError(f'is directory: {url}')
        return LocalStatFileStatus(stat_result)

    # entries has no type hint because the return type of os.scandir
    # appears to be a private type, posix.ScandirIterator.
    # >>> import os
    # >>> entries = os.scandir('.')
    # >>> type(entries)
    # <class 'posix.ScandirIterator'>
    # >>> import posix
    # >>> posix.ScandirIterator
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # AttributeError: module 'posix' has no attribute 'ScandirIterator'
    async def _listfiles_recursive(self, url: str, entries) -> AsyncIterator[FileListEntry]:
        async for file in self._listfiles_flat(url, entries):
            if await file.is_file():
                yield file
            else:
                new_url = await file.url()
                new_path = self._get_path(new_url)
                new_entries = await blocking_to_async(self._thread_pool, os.scandir, new_path)
                async for subfile in self._listfiles_recursive(new_url, new_entries):
                    yield subfile

    async def _listfiles_flat(self, url: str, entries) -> AsyncIterator[FileListEntry]:
        with entries:
            for entry in entries:
                yield LocalFileListEntry(self._thread_pool, url, entry)

    async def listfiles(self, url: str, recursive: bool = False) -> AsyncIterator[FileListEntry]:
        path = self._get_path(url)
        entries = await blocking_to_async(self._thread_pool, os.scandir, path)
        if recursive:
            return self._listfiles_recursive(url, entries)
        return self._listfiles_flat(url, entries)

    async def mkdir(self, url: str) -> None:
        path = self._get_path(url)
        os.mkdir(path)

    async def isfile(self, url: str) -> bool:
        path = self._get_path(url)
        return await blocking_to_async(self._thread_pool, os.path.isfile, path)

    async def isdir(self, url: str) -> bool:
        path = self._get_path(url)
        return await blocking_to_async(self._thread_pool, os.path.isdir, path)

    async def remove(self, url: str) -> None:
        path = self._get_path(url)
        return os.remove(path)

    async def rmtree(self, url: str) -> None:
        path = self._get_path(url)
        await blocking_to_async(self._thread_pool, shutil.rmtree, path)


class RouterAsyncFS(AsyncFS):
    def __init__(self, default_scheme: Optional[str], filesystems: List[AsyncFS]):
        scheme_fs = {}
        schemes = set()
        for fs in filesystems:
            for scheme in fs.schemes():
                if scheme not in schemes:
                    scheme_fs[scheme] = fs
                    schemes.add(scheme)

        if default_scheme not in schemes:
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
                url = urllib.parse.urlunparse(parsed)
            else:
                raise ValueError(f"no default scheme and URL has no scheme: {url}")

        fs = self._scheme_fs.get(parsed.scheme)
        if fs is None:
            raise ValueError(f"unknown scheme: {parsed.scheme}")

        return fs

    async def open(self, url: str) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open(url)

    async def create(self, url: str) -> WritableStream:
        fs = self._get_fs(url)
        return await fs.create(url)

    async def statfile(self, url: str) -> FileStatus:
        fs = self._get_fs(url)
        return await fs.statfile(url)

    async def listfiles(self, url: str, recursive: bool = False) -> AsyncIterator[FileListEntry]:
        fs = self._get_fs(url)
        return await fs.listfiles(url, recursive)

    async def mkdir(self, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.mkdir(url)

    async def isfile(self, url: str) -> bool:
        fs = self._get_fs(url)
        return await fs.isfile(url)

    async def isdir(self, url: str) -> bool:
        fs = self._get_fs(url)
        return await fs.isdir(url)

    async def remove(self, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.remove(url)

    async def rmtree(self, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.rmtree(url)

    async def close(self) -> None:
        for fs in self._filesystems:
            await fs.close()
