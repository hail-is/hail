import asyncio
import datetime
import io
import os
import os.path
import stat
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, AsyncIterator, BinaryIO, Callable, ClassVar, Dict, Iterator, List, Optional, Set, Type, cast

from ..utils import OnlineBoundedGather2, blocking_to_async
from .fs import (
    AsyncFS,
    AsyncFSURL,
    FileListEntry,
    FileStatus,
    MultiPartCreate,
    ReadableStream,
    WritableStream,
    blocking_readable_stream_to_async,
    blocking_writable_stream_to_async,
)


class LocalStatFileStatus(FileStatus):
    def __init__(self, stat_result: os.stat_result):
        self._stat_result = stat_result
        self._items = None

    async def size(self) -> int:
        return self._stat_result.st_size

    def time_created(self) -> datetime.datetime:
        raise ValueError('LocalFS does not support time created.')

    def time_modified(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self._stat_result.st_mtime, tz=datetime.timezone.utc)

    async def __getitem__(self, key: str) -> Any:
        raise KeyError(key)


class LocalFileListEntry(FileListEntry):
    def __init__(self, thread_pool: ThreadPoolExecutor, base_url: str, entry: os.DirEntry):
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

    async def url_maybe_trailing_slash(self) -> str:
        return f'{self._base_url}{self._entry.name}'

    async def is_file(self) -> bool:
        return not await self.is_dir()

    async def is_dir(self) -> bool:
        return await blocking_to_async(self._thread_pool, self._entry.is_dir)

    async def status(self) -> LocalStatFileStatus:
        if self._status is None:
            if await self.is_dir():
                raise IsADirectoryError()
            self._status = LocalStatFileStatus(await blocking_to_async(self._thread_pool, self._entry.stat))
        return self._status


class LocalMultiPartCreate(MultiPartCreate):
    def __init__(self, fs: 'LocalAsyncFS', path: str, num_parts: int):
        self._fs = fs
        self._path = path
        self._num_parts = num_parts

    async def create_part(
        self, number: int, start: int, size_hint: Optional[int] = None
    ):  # pylint: disable=unused-argument
        assert 0 <= number < self._num_parts
        f = await blocking_to_async(self._fs._thread_pool, open, self._path, 'r+b')
        f.seek(start)
        return blocking_writable_stream_to_async(self._fs._thread_pool, cast(BinaryIO, f))

    async def __aenter__(self) -> 'LocalMultiPartCreate':
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        if exc_val:
            try:
                await self._fs.remove(self._path)
            except FileNotFoundError:
                pass


class LocalAsyncFSURL(AsyncFSURL):
    def __init__(self, path: str):
        self._path = path

    @property
    def bucket_parts(self) -> List[str]:
        return []

    @property
    def path(self) -> str:
        return self._path

    @property
    def query(self) -> Optional[str]:
        return None

    @property
    def scheme(self) -> str:
        return 'file'

    def with_path(self, path) -> 'LocalAsyncFSURL':
        return LocalAsyncFSURL(path)

    def __str__(self) -> str:
        return self._path


class TruncatedReadableBinaryIO(BinaryIO):
    def __init__(self, bio: BinaryIO, limit: int):
        self.bio = bio
        self.offset = 0
        self.limit = limit

    def write(self, s) -> int:  # pylint: disable=unused-argument
        raise NotImplementedError

    def __enter__(self) -> 'BinaryIO':
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def mode(self) -> str:
        return self.bio.mode

    @property
    def name(self) -> str:
        return self.bio.name

    def close(self) -> None:
        return self.bio.close()

    @property
    def closed(self) -> bool:
        return self.bio.closed

    def fileno(self) -> int:
        return self.bio.fileno()

    def flush(self) -> None:
        raise NotImplementedError

    def isatty(self) -> bool:
        return self.bio.isatty()

    def read(self, n: int = -1):
        assert self.offset <= self.limit

        if n == -1:
            n = self.limit - self.offset
        else:
            n = min(self.limit - self.offset, n)
        b = self.bio.read(n)
        self.offset += len(b)
        return b

    def readable(self) -> bool:
        return True

    def readline(self, limit: int = -1):  # pylint: disable=unused-argument
        raise NotImplementedError

    def readlines(self, hint: int = -1):  # pylint: disable=unused-argument
        raise NotImplementedError

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        res = self.bio.seek(offset, whence)
        if whence == os.SEEK_SET:
            self.offset = offset
        elif whence == os.SEEK_CUR:
            self.offset += offset
        elif whence == os.SEEK_END:
            assert offset < 0
            self.offset = self.limit + offset
        else:
            raise ValueError(f'Unsupported seek whence: {whence}')
        return res

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.bio.tell()

    def truncate(self, size: Optional[int] = None):
        raise NotImplementedError

    def writable(self) -> bool:
        return False

    def writelines(self, lines) -> None:  # pylint: disable=unused-argument
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class LocalAsyncFS(AsyncFS):
    schemes: ClassVar[Set[str]] = {'file'}

    def __init__(self, thread_pool: Optional[ThreadPoolExecutor] = None, max_workers: Optional[int] = None):
        if not thread_pool:
            thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._thread_pool = thread_pool

    @staticmethod
    def valid_url(url: str) -> bool:
        return url.startswith('file://') or '://' not in url

    @staticmethod
    def parse_url(url: str) -> LocalAsyncFSURL:
        return LocalAsyncFSURL(LocalAsyncFS._get_path(url))

    @staticmethod
    def _get_path(url):
        parsed = urllib.parse.urlparse(url)
        prefix = ''
        if parsed.scheme:
            if parsed.scheme != 'file':
                raise ValueError(f"invalid file URL: {url}, invalid scheme: expected file, got {parsed.scheme}")
            prefix += f'{parsed.scheme}://'
        if parsed.netloc:
            if parsed.netloc != 'localhost':
                raise ValueError(
                    f"invalid file URL: {url}, invalid netloc: expected localhost or empty, got {parsed.netloc}"
                )
            prefix += parsed.netloc
        return url[len(prefix) :]

    async def open(self, url: str) -> ReadableStream:
        f = await blocking_to_async(self._thread_pool, open, self._get_path(url), 'rb')
        return blocking_readable_stream_to_async(self._thread_pool, cast(BinaryIO, f))

    async def _open_from(self, url: str, start: int, *, length: Optional[int] = None) -> ReadableStream:
        f = await blocking_to_async(self._thread_pool, open, self._get_path(url), 'rb')
        f.seek(start, io.SEEK_SET)
        bio = cast(BinaryIO, f)
        if length is not None:
            assert length >= 1
            bio = TruncatedReadableBinaryIO(bio, length)
        return blocking_readable_stream_to_async(self._thread_pool, bio)

    async def create(self, url: str, *, retry_writes: bool = True) -> WritableStream:  # pylint: disable=unused-argument
        f = await blocking_to_async(self._thread_pool, open, self._get_path(url), 'wb')
        return blocking_writable_stream_to_async(self._thread_pool, cast(BinaryIO, f))

    async def multi_part_create(
        self, sema: asyncio.Semaphore, url: str, num_parts: int  # pylint: disable=unused-argument
    ) -> MultiPartCreate:
        # create an empty file
        # will be opened r+b to write the parts
        async with await self.create(url):
            pass
        return LocalMultiPartCreate(self, self._get_path(url), num_parts)

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
    async def _listfiles_recursive(
        self, url: str, entries: AbstractContextManager[Iterator[os.DirEntry]]
    ) -> AsyncIterator[FileListEntry]:
        async for file in self._listfiles_flat(url, entries):
            if await file.is_file():
                yield file
            else:
                new_url = await file.url()
                new_path = self._get_path(new_url)
                new_entries = await blocking_to_async(self._thread_pool, os.scandir, new_path)
                async for subfile in self._listfiles_recursive(new_url, new_entries):
                    yield subfile

    async def _listfiles_flat(
        self, url: str, entries: AbstractContextManager[Iterator[os.DirEntry]]
    ) -> AsyncIterator[FileListEntry]:
        with entries as it:
            for entry in it:
                yield LocalFileListEntry(self._thread_pool, url, entry)

    async def listfiles(
        self, url: str, recursive: bool = False, exclude_trailing_slash_files: bool = True
    ) -> AsyncIterator[FileListEntry]:
        del exclude_trailing_slash_files  # such files do not exist on local file systems
        path = self._get_path(url)
        entries = await blocking_to_async(self._thread_pool, os.scandir, path)
        if recursive:
            return self._listfiles_recursive(url, entries)
        return self._listfiles_flat(url, entries)

    async def staturl(self, url: str) -> str:
        path = self._get_path(url)
        stat_result = await blocking_to_async(self._thread_pool, os.stat, path)
        if stat.S_ISDIR(stat_result.st_mode):
            return AsyncFS.DIR
        return AsyncFS.FILE

    async def mkdir(self, url: str) -> None:
        path = self._get_path(url)
        await blocking_to_async(self._thread_pool, os.mkdir, path)

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        path = self._get_path(url)
        await blocking_to_async(self._thread_pool, os.makedirs, path, exist_ok=exist_ok)

    async def isfile(self, url: str) -> bool:
        path = self._get_path(url)
        return await blocking_to_async(self._thread_pool, os.path.isfile, path)

    async def isdir(self, url: str) -> bool:
        path = self._get_path(url)
        return await blocking_to_async(self._thread_pool, os.path.isdir, path)

    async def remove(self, url: str) -> None:
        path = self._get_path(url)
        return await blocking_to_async(self._thread_pool, os.remove, path)

    async def rmdir(self, url: str) -> None:
        path = self._get_path(url)
        return await blocking_to_async(self._thread_pool, os.rmdir, path)

    async def rmtree(
        self, sema: Optional[asyncio.Semaphore], url: str, listener: Optional[Callable[[int], None]] = None
    ) -> None:
        path = self._get_path(url)
        if listener is None:
            listener = lambda _: None
        if sema is None:
            sema = asyncio.Semaphore(50)

        async def rm_file(path: str):
            assert listener is not None
            listener(1)
            await self.remove(path)
            listener(-1)

        async def rm_dir(pool: OnlineBoundedGather2, contents_tasks: List[asyncio.Task], path: str):
            assert listener is not None
            listener(1)
            if contents_tasks:
                await pool.wait(contents_tasks)

                def raise_them_all(exceptions: List[BaseException]):
                    if exceptions:
                        try:
                            raise exceptions[0]
                        finally:
                            raise_them_all(exceptions[1:])

                excs = [exc for t in contents_tasks for exc in [t.exception()] if exc is not None]
                raise_them_all(excs)
            await self.rmdir(path)
            listener(-1)

        async with OnlineBoundedGather2(sema) as pool:
            contents_tasks_by_dir: Dict[str, List[asyncio.Task]] = {}
            for dirpath, dirnames, filenames in os.walk(path, topdown=False):

                def rm_dir_or_symlink(path: str):
                    if os.path.islink(path):
                        return pool.call(rm_file, path)
                    return pool.call(rm_dir, pool, contents_tasks_by_dir.get(path, []), path)

                contents_tasks = [pool.call(rm_file, os.path.join(dirpath, filename)) for filename in filenames] + [
                    rm_dir_or_symlink(os.path.join(dirpath, dirname)) for dirname in dirnames
                ]
                contents_tasks_by_dir[dirpath] = contents_tasks
            await rm_dir(pool, contents_tasks_by_dir.get(path, []), path)
