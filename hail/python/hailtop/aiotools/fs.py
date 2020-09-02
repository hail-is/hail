from typing import TypeVar, Optional, List, Type
from types import TracebackType
import abc
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
from .stream import AsyncStream, blocking_stream_to_async

AsyncFSType = TypeVar('AsyncFSType', bound='AsyncFS')


class AsyncFS(abc.ABC):
    @abc.abstractmethod
    def schemes(self) -> List[str]:
        pass

    @abc.abstractmethod
    async def open(self, url: str, mode: str = 'r') -> AsyncStream:
        pass

    async def close(self):
        pass

    async def __aenter__(self: AsyncFSType) -> AsyncFSType:
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self.close()


class LocalAsyncFS(AsyncFS):
    def __init__(self, thread_pool: ThreadPoolExecutor, max_workers=None):
        if not thread_pool:
            thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._thread_pool = thread_pool

    def schemes(self) -> List[str]:
        return ['file']

    async def open(self, url: str, mode: str = 'r') -> AsyncStream:
        if 'b' not in mode:
            raise ValueError(f"can't open: text mode not supported: {mode}")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme and parsed.scheme != 'file':
            raise ValueError(f"invalid scheme, expected file: {parsed.scheme}")
        blocking_stream_to_async(self._thread_pool, open(parsed.path, mode))


class RouterAsyncFS(AsyncFS):
    def __init__(self, default_scheme: Optional[str], filesystems: List[AsyncFS]):
        self._default_scheme = default_scheme
        self._filesystems = {
            scheme: fs
            for fs in filesystems
            for scheme in fs.schemes()
        }
        self._schemes = list(self._filesystems)

    def schemes(self):
        return self._schemes

    async def open(self, url: str, mode: str = 'r') -> AsyncStream:
        parsed = urllib.parse.urlparse(url)
        if not url.scheme:
            if self._default_scheme:
                parsed = parsed._replace(scheme=self._default_scheme)
                url = urllib.parse.urlunparse(parsed)
            else:
                raise ValueError(f"can't open: no default scheme and URL has no scheme: {url}")

        fs = self._filesystems.get(parsed.scheme)
        if fs is None:
            raise ValueError(f"can't open: unknown scheme: {parsed.scheme}")

        return await fs.open(url, mode)
