from typing import Any, AsyncContextManager, Optional, Type, Set, AsyncIterator, Callable
from types import TracebackType
import abc
import asyncio
from hailtop.utils import retry_transient_errors, OnlineBoundedGather2
from .stream import ReadableStream, WritableStream
from .exceptions import FileAndDirectoryError


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
    def url_maybe_trailing_slash(self) -> str:
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


class MultiPartCreate(abc.ABC):
    @abc.abstractmethod
    async def create_part(self, number: int, start: int, size_hint: Optional[int] = None) -> AsyncContextManager[WritableStream]:
        pass

    @abc.abstractmethod
    async def __aenter__(self) -> 'MultiPartCreate':
        pass

    @abc.abstractmethod
    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        pass


class AsyncFS(abc.ABC):
    FILE = 'file'
    DIR = 'dir'

    @property
    @abc.abstractmethod
    def schemes(self) -> Set[str]:
        pass

    @abc.abstractmethod
    async def open(self, url: str) -> ReadableStream:
        pass

    @abc.abstractmethod
    async def open_from(self, url: str, start: int) -> ReadableStream:
        pass

    @abc.abstractmethod
    async def create(self, url: str, *, retry_writes: bool = True) -> AsyncContextManager[WritableStream]:
        pass

    @abc.abstractmethod
    async def multi_part_create(
            self,
            sema: asyncio.Semaphore,
            url: str,
            num_parts: int) -> MultiPartCreate:
        pass

    @abc.abstractmethod
    async def mkdir(self, url: str) -> None:
        pass

    @abc.abstractmethod
    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        pass

    @abc.abstractmethod
    async def statfile(self, url: str) -> FileStatus:
        pass

    @abc.abstractmethod
    async def listfiles(self,
                        url: str,
                        recursive: bool = False,
                        exclude_trailing_slash_files: bool = True) -> AsyncIterator[FileListEntry]:
        pass

    @abc.abstractmethod
    async def staturl(self, url: str) -> str:
        pass

    async def _staturl_parallel_isfile_isdir(self, url: str) -> str:
        assert not url.endswith('/')

        async def with_exception(f, *args, **kwargs):
            try:
                return (await f(*args, **kwargs)), None
            except Exception as e:
                return None, e

        [(is_file, isfile_exc), (is_dir, isdir_exc)] = await asyncio.gather(
            with_exception(self.isfile, url), with_exception(self.isdir, url + '/'))
        # raise exception deterministically
        if isfile_exc:
            raise isfile_exc
        if isdir_exc:
            raise isdir_exc

        if is_file:
            if is_dir:
                raise FileAndDirectoryError(url)
            return AsyncFS.FILE

        if is_dir:
            return AsyncFS.DIR

        raise FileNotFoundError(url)

    @abc.abstractmethod
    async def isfile(self, url: str) -> bool:
        pass

    @abc.abstractmethod
    async def isdir(self, url: str) -> bool:
        pass

    @abc.abstractmethod
    async def remove(self, url: str) -> None:
        pass

    async def _remove_doesnt_exist_ok(self, url):
        try:
            await self.remove(url)
        except FileNotFoundError:
            pass

    async def rmtree(self,
                     sema: Optional[asyncio.Semaphore],
                     url: str,
                     listener: Optional[Callable[[int], None]] = None) -> None:
        if listener is None:
            listener = lambda _: None  # noqa: E731
        if sema is None:
            sema = asyncio.Semaphore(50)

        async def rm(entry: FileListEntry):
            assert listener is not None
            listener(1)
            await self._remove_doesnt_exist_ok(await entry.url())
            listener(-1)

        try:
            it = await self.listfiles(url, recursive=True, exclude_trailing_slash_files=False)
        except FileNotFoundError:
            return

        async with OnlineBoundedGather2(sema) as pool:
            tasks = [pool.call(rm, entry) async for entry in it]
            if tasks:
                await pool.wait(tasks)

    async def touch(self, url: str) -> None:
        async with await self.create(url):
            pass

    async def read(self, url: str) -> bytes:
        async with await self.open(url) as f:
            return await f.read()

    async def read_from(self, url: str, start: int) -> bytes:
        async with await self.open_from(url, start) as f:
            return await f.read()

    async def read_range(self, url: str, start: int, end: int) -> bytes:
        n = (end - start) + 1
        async with await self.open_from(url, start) as f:
            return await f.readexactly(n)

    async def write(self, url: str, data: bytes) -> None:
        async def _write() -> None:
            async with await self.create(url, retry_writes=False) as f:
                await f.write(data)

        await retry_transient_errors(_write)

    async def exists(self, url: str) -> bool:
        try:
            await self.statfile(url)
        except FileNotFoundError:
            return False
        else:
            return True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> 'AsyncFS':
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self.close()

    def copy_part_size(self, url: str) -> int:  # pylint: disable=unused-argument,no-self-use
        '''Part size when copying using multi-part uploads.  The part size of
        the destination filesystem is used.'''
        return 128 * 1024 * 1024
