from typing import BinaryIO, Optional, Tuple, Type
from types import TracebackType
import abc
import io
import os
from concurrent.futures import ThreadPoolExecutor
import janus
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
    def closed(self) -> bool:
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
    def closed(self) -> bool:
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
        if n == -1:
            return await blocking_to_async(self._thread_pool, self._f.read)
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
        await blocking_to_async(self._thread_pool, self._f.flush)
        await blocking_to_async(self._thread_pool, os.fsync, self._f.fileno())
        await blocking_to_async(self._thread_pool, self._f.close)
        del self._f


def blocking_readable_stream_to_async(thread_pool: ThreadPoolExecutor, f: BinaryIO) -> _ReadableStreamFromBlocking:
    return _ReadableStreamFromBlocking(thread_pool, f)


def blocking_writable_stream_to_async(thread_pool: ThreadPoolExecutor, f: BinaryIO) -> _WritableStreamFromBlocking:
    return _WritableStreamFromBlocking(thread_pool, f)


class BlockingQueueReadableStream(io.RawIOBase):
    # self.closed and self.close() must be multithread safe, because
    # they can be accessed by both the stream reader and writer which
    # are in different threads.
    def __init__(self, q: janus.Queue):
        super().__init__()
        self._q = q
        self._saw_eos = False
        self._closed = False
        self._unread = b''

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        if self._closed:
            raise ValueError('read on closed stream')
        if self._saw_eos:
            return 0

        if not self._unread:
            self._unread = self._q.sync_q.get()
            if self._unread is None:
                self._saw_eos = True
                return 0
        assert self._unread

        n = min(len(self._unread), len(b))
        b[:n] = self._unread[:n]
        self._unread = self._unread[n:]
        return n

    def close(self):
        self._closed = True
        # drain the q so the writer doesn't deadlock
        while not self._saw_eos:
            c = self._q.sync_q.get()
            if c is None:
                self._saw_eos = True


class AsyncQueueWritableStream(WritableStream):
    def __init__(self, q: janus.Queue, blocking_readable: BlockingQueueReadableStream):
        super().__init__()
        self._sent_eos = False
        self._q = q
        self._blocking_readable = blocking_readable

    async def write(self, b: bytes) -> int:
        if self._blocking_readable._closed:
            if not self._sent_eos:
                await self._q.async_q.put(None)
                self._sent_eos = True
            raise ValueError('reader closed')
        if b:
            await self._q.async_q.put(b)
        return len(b)

    async def _wait_closed(self) -> None:
        if not self._sent_eos:
            await self._q.async_q.put(None)
            self._sent_eos = True


def async_writable_blocking_readable_stream_pair() -> Tuple[AsyncQueueWritableStream, BlockingQueueReadableStream]:
    q: janus.Queue = janus.Queue(maxsize=1)
    blocking_readable = BlockingQueueReadableStream(q)
    async_writable = AsyncQueueWritableStream(q, blocking_readable)
    return async_writable, blocking_readable
