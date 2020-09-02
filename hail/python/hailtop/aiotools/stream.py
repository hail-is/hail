from typing import Optional, Type
from types import TracebackType
import abc
import io
from concurrent.futures import ThreadPoolExecutor
from hailtop.utils import blocking_to_async


class AsyncStream(abc.ABC):
    def __init__(self):
        self._closed = False
        self._waited_closed = False

    def readable(self) -> bool:  # pylint: disable=no-self-use
        return False

    async def read(self, n: int = -1) -> bytes:
        raise NotImplementedError

    def writable(self) -> bool:  # pylint: disable=no-self-use
        return False

    async def write(self, b: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True

    @abc.abstractmethod
    async def _wait_closed(self) -> None:
        pass

    async def wait_closed(self) -> None:
        self._closed = True
        if not self._waited_closed:
            try:
                await self._wait_closed()
            finally:
                self._waited_closed = True

    @property
    def closed(self) -> None:
        return self._closed

    async def __aenter__(self) -> 'AsyncStream[bytes]':
        return self

    async def __aexit__(
            self, exc_type: Optional[Type[BaseException]] = None,
            exc_value: Optional[BaseException] = None,
            exc_traceback: Optional[TracebackType] = None) -> None:
        await self.wait_closed()


class _AsyncStreamFromBlocking(AsyncStream):
    def __init__(self, thread_pool: ThreadPoolExecutor, f: io.RawIOBase):
        super().__init__()
        self._thread_pool = thread_pool
        self._f = f

    def readable(self) -> bool:
        return self._f.readable()

    async def read(self, n: int = -1) -> bytes:
        if not self.readable():
            raise NotImplementedError
        assert not self.closed
        return await blocking_to_async(self._thread_pool, self._f.read, n)

    def writable(self) -> bool:
        return self._f.writable()

    async def write(self, b: bytes) -> int:
        if not self.writable():
            raise NotImplementedError
        assert not self.closed
        return await blocking_to_async(self._thread_pool, self._f.write, b)

    async def _wait_closed(self) -> None:
        await blocking_to_async(self._thread_pool, self._f.close)
        self._f = None


def blocking_stream_to_async(thread_pool: ThreadPoolExecutor, f: io.RawIOBase) -> _AsyncStreamFromBlocking:
    return _AsyncStreamFromBlocking(thread_pool, f)
