import abc
import asyncio
import datetime
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec, Self

from hailtop.utils import OnlineBoundedGather2, retry_transient_errors

from .exceptions import FileAndDirectoryError
from .stream import EmptyReadableStream, ReadableStream, WritableStream

T = TypeVar("T")
P = ParamSpec("P")


async def with_exception(
    f: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs
) -> Union[Tuple[T, None], Tuple[None, Exception]]:
    try:
        return (await f(*args, **kwargs)), None
    except Exception as e:
        return None, e


class FileStatus(abc.ABC):
    @abc.abstractmethod
    def basename(self) -> str:
        """The basename of the object.

        Examples
        --------

        The basename of all of these objects is "file":

        - s3://bucket/folder/file
        - gs://bucket/folder/file
        - https://account.blob.core.windows.net/container/folder/file
        - https://account.blob.core.windows.net/container/folder/file?sv=2023-01-01&sr=bv&sig=abc123&sp=rcw
        - /folder/file
        """

    @abc.abstractmethod
    def url(self) -> str:
        """The URL of the object without any query parameters.

        Examples
        --------

        - s3://bucket/folder/file
        - gs://bucket/folder/file
        - https://account.blob.core.windows.net/container/folder/file
        - /folder/file

        Note that the following URL

            https://account.blob.core.windows.net/container/folder/file?sv=2023-01-01&sr=bv&sig=abc123&sp=rcw

        becomes

            https://account.blob.core.windows.net/container/folder/file

        """

    @abc.abstractmethod
    async def size(self) -> int:
        pass

    @abc.abstractmethod
    def time_created(self) -> datetime.datetime:
        """The time the object was created in seconds since the epcoh, UTC.

        Some filesystems do not support creation time. In that case, an error is raised.

        """

    @abc.abstractmethod
    def time_modified(self) -> datetime.datetime:
        """The time the object was last modified in seconds since the epoch, UTC.

        The meaning of modification time is cloud-defined. In some clouds, it is the creation
        time. In some clouds, it is the more recent of the creation time or the time of the most
        recent metadata modification.

        """

    @abc.abstractmethod
    async def __getitem__(self, key: str) -> Any:
        pass


class FileListEntry(abc.ABC):
    @abc.abstractmethod
    def basename(self) -> str:
        """The basename of the object.

        Examples
        --------

        The basename of all of these objects is "file":

        - s3://bucket/folder/file
        - gs://bucket/folder/file
        - https://account.blob.core.windows.net/container/folder/file
        - https://account.blob.core.windows.net/container/folder/file?sv=2023-01-01&sr=bv&sig=abc123&sp=rcw
        - /folder/file
        """

    @abc.abstractmethod
    async def url(self) -> str:
        """The URL of the object without any query parameters.

        Examples
        --------

        - s3://bucket/folder/file
        - gs://bucket/folder/file
        - https://account.blob.core.windows.net/container/folder/file
        - /folder/file

        Note that the following URL

            https://account.blob.core.windows.net/container/folder/file?sv=2023-01-01&sr=bv&sig=abc123&sp=rcw

        becomes

            https://account.blob.core.windows.net/container/folder/file

        """

    async def url_maybe_trailing_slash(self) -> str:
        return await self.url()

    async def url_full(self) -> str:
        """The URL of the object with any query parameters.

        Examples
        --------

        The only interesting case is for signed URLs in Azure. These are called shared signature tokens or SAS tokens.
        For example, the following URL

            https://account.blob.core.windows.net/container/folder/file?sv=2023-01-01&sr=bv&sig=abc123&sp=rcw

        is a signed version of this URL

            https://account.blob.core.windows.net/container/folder/file

        """
        return await self.url()

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
    async def create_part(
        self, number: int, start: int, size_hint: Optional[int] = None
    ) -> AsyncContextManager[WritableStream]:
        pass

    @abc.abstractmethod
    async def __aenter__(self) -> "MultiPartCreate":
        pass

    @abc.abstractmethod
    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        pass


class AsyncFSURL(abc.ABC):
    @property
    @abc.abstractmethod
    def bucket_parts(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def path(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def query(self) -> Optional[str]:
        pass

    @property
    @abc.abstractmethod
    def scheme(self) -> str:
        pass

    @abc.abstractmethod
    def with_path(self, path) -> "AsyncFSURL":
        pass

    @abc.abstractmethod
    def with_root_path(self) -> "AsyncFSURL":
        pass

    def with_new_path_component(self, new_path_component: str) -> "AsyncFSURL":
        if new_path_component == '':
            raise ValueError('new path component must be non-empty')
        return self.with_new_path_components(new_path_component)

    def with_new_path_components(self, *parts: str) -> "AsyncFSURL":
        if len(parts) == 0:
            return self

        prefix = self.path
        if not prefix.endswith("/") and not prefix == '':
            prefix += "/"

        suffix = '/'.join(parts)
        if suffix[0] == '/':
            suffix = suffix[1:]

        return self.with_path(prefix + suffix)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass


class AsyncFS(abc.ABC):
    FILE = "file"
    DIR = "dir"

    @staticmethod
    @abc.abstractmethod
    def schemes() -> Set[str]:
        pass

    @staticmethod
    def copy_part_size(url: str) -> int:  # pylint: disable=unused-argument
        """Part size when copying using multi-part uploads.  The part size of
        the destination filesystem is used."""
        return 128 * 1024 * 1024

    @staticmethod
    @abc.abstractmethod
    def valid_url(url: str) -> bool:
        pass

    @staticmethod
    @abc.abstractmethod
    def parse_url(url: str, *, error_if_bucket: bool = False) -> AsyncFSURL:
        pass

    @abc.abstractmethod
    async def open(self, url: str) -> ReadableStream:
        pass

    async def open_from(self, url: str, start: int, *, length: Optional[int] = None) -> ReadableStream:
        if length == 0:
            fs_url = self.parse_url(url)
            if fs_url.path.endswith("/"):
                file_url = str(fs_url.with_path(fs_url.path.rstrip("/")))
                dir_url = str(fs_url)
            else:
                file_url = str(fs_url)
                dir_url = str(fs_url.with_path(fs_url.path + "/"))
            isfile, isdir = await asyncio.gather(self.isfile(file_url), self.isdir(dir_url))
            if isfile:
                if isdir:
                    raise FileAndDirectoryError
                return EmptyReadableStream()
            if isdir:
                raise IsADirectoryError
            raise FileNotFoundError
        return await self._open_from(url, start, length=length)

    @abc.abstractmethod
    async def _open_from(self, url: str, start: int, *, length: Optional[int] = None) -> ReadableStream:
        pass

    @abc.abstractmethod
    async def create(self, url: str, *, retry_writes: bool = True) -> AsyncContextManager[WritableStream]:
        pass

    @abc.abstractmethod
    async def multi_part_create(self, sema: asyncio.Semaphore, url: str, num_parts: int) -> MultiPartCreate:
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
    async def listfiles(
        self, url: str, recursive: bool = False, exclude_trailing_slash_files: bool = True
    ) -> AsyncIterator[FileListEntry]:
        pass

    @abc.abstractmethod
    async def staturl(self, url: str) -> str:
        pass

    async def _staturl_parallel_isfile_isdir(self, url: str) -> str:
        assert not url.endswith("/")

        [(is_file, isfile_exc), (is_dir, isdir_exc)] = await asyncio.gather(
            with_exception(self.isfile, url), with_exception(self.isdir, url + "/")
        )
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

    async def rmtree(
        self, sema: Optional[asyncio.Semaphore], url: str, listener: Optional[Callable[[int], None]] = None
    ) -> None:
        if listener is None:
            listener = lambda _: None
        if sema is None:
            sema = asyncio.Semaphore(50)

        async def rm(entry: FileListEntry):
            assert listener is not None
            listener(1)
            await self._remove_doesnt_exist_ok(await entry.url_full())
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

    async def read_range(self, url: str, start: int, end: int, *, end_inclusive=True) -> bytes:
        n = (end - start) + bool(end_inclusive)
        async with await self.open_from(url, start, length=n) as f:
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
        return True

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        await self.close()


T = TypeVar("T", bound=AsyncFS)


class AsyncFSFactory(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def from_credentials_data(self, credentials_data: dict) -> T:
        pass

    @abc.abstractmethod
    def from_credentials_file(self, credentials_file: str) -> T:
        pass

    @abc.abstractmethod
    def from_default_credentials(self) -> T:
        pass
