from typing import Optional, Type, BinaryIO
from types import TracebackType
import abc
from concurrent.futures import ThreadPoolExecutor
from hailtop.utils import blocking_to_async


class ReadableStream(abc.ABC):
    def __init__(self):
        self._closed = False
        self._waited_closed = False

    async def read(self, n: int = -1) -> bytes:
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

    async def __aenter__(self) -> 'ReadableStream':
        return self

    async def __aexit__(
            self, exc_type: Optional[Type[BaseException]] = None,
            exc_value: Optional[BaseException] = None,
            exc_traceback: Optional[TracebackType] = None) -> None:
        await self.wait_closed()


class WritableStream(abc.ABC):
    def __init__(self):
        self._closed = False
        self._waited_closed = False

    def writable(self) -> bool:  # pylint: disable=no-self-use
        return False

    async def write(self, b: bytes) -> int:
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

    async def __aenter__(self) -> 'WritableStream':
        return self

    async def __aexit__(
            self, exc_type: Optional[Type[BaseException]] = None,
            exc_value: Optional[BaseException] = None,
            exc_traceback: Optional[TracebackType] = None) -> None:
        await self.wait_closed()


class _ReadableStreamFromBlocking(ReadableStream):
    _thread_pool: ThreadPoolExecutor
    _f: BinaryIO

    def __init__(self, thread_pool: ThreadPoolExecutor, f: BinaryIO):
        super().__init__()
        self._thread_pool = thread_pool
        self._f = f

    async def read(self, n: int = -1) -> bytes:
        return await blocking_to_async(self._thread_pool, self._f.read, n)

    async def _wait_closed(self) -> None:
        await blocking_to_async(self._thread_pool, self._f.close)
        del self._f


class _WritableStreamFromBlocking(WritableStream):
    _thread_pool: ThreadPoolExecutor
    _f: BinaryIO

    def __init__(self, thread_pool: ThreadPoolExecutor, f: BinaryIO):
        super().__init__()
        self._thread_pool = thread_pool
        self._f = f

    def writable(self) -> bool:
        return self._f.writable()

    async def write(self, b: bytes) -> int:
        return await blocking_to_async(self._thread_pool, self._f.write, b)

    async def _wait_closed(self) -> None:
        await blocking_to_async(self._thread_pool, self._f.close)
        del self._f


def blocking_readable_stream_to_async(thread_pool: ThreadPoolExecutor, f: BinaryIO) -> _ReadableStreamFromBlocking:
    return _ReadableStreamFromBlocking(thread_pool, f)


def blocking_writable_stream_to_async(thread_pool: ThreadPoolExecutor, f: BinaryIO) -> _WritableStreamFromBlocking:
    return _WritableStreamFromBlocking(thread_pool, f)
