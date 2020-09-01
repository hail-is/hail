from typing import TypeVar, AsyncIterator
import io
import asyncio


_T = TypeVar('_T')


class _StopFeedableAsyncIterator:
    pass


class FeedableAsyncIterable(AsyncIterator[_T]):
    def __init__(self, maxsize: int = 1):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._stopped = False

    def __aiter__(self) -> 'FeedableAsyncIterable[_T]':
        return self

    async def __anext__(self) -> _T:
        next = await self._queue.get()
        if isinstance(next, _StopFeedableAsyncIterator):
            raise StopAsyncIteration
        return next

    async def feed(self, next: _T) -> None:
        assert not self._stopped
        await self._queue.put(next)

    async def stop(self) -> None:
        await self._queue.put(_StopFeedableAsyncIterator())
        self._stopped = True


class _AsyncStreamFromBlocking(AsyncStream):
    def __init__(self, f: io.RawIOBase):
        self._f = f

    @property
    def readable(self) -> bool:
        return self._f.readable()

    async def read(self, n: int = -1) -> bytes:
        if not self.readable():
            raise NotImplementedError
        assert not self.closed
        return self._f.read(n)

    def writable(self) -> bool:
        return self._f.writable()

    async def write(self, b: bytes) -> None:
        if not self.writable():
            raise NotImplementedError
        assert not self.closed
        return self._f.write(b)

    async def _wait_closed(self) -> None:
        self._f.close()
        self._f = None


def blocking_stream_to_async(f: io.RawIOBase) -> _AsyncStreamFromBlocking:
    return _AsyncStreamFromBlocking(f)
