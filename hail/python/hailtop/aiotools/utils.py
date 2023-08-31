from typing import Deque, TypeVar, AsyncIterator, Iterator
import collections
from contextlib import contextmanager
import asyncio
import random
from hailtop.utils import TransientError


_T = TypeVar('_T')


class _StopFeedableAsyncIterator:
    pass


class FeedableAsyncIterable(AsyncIterator[_T]):
    def __init__(self, maxsize: int = 1):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
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


class WriteBuffer:
    def __init__(self):
        """Long writes that might fail need to be broken into smaller chunks
        that can be retried. WriteBuffer stores data at the end of
        the write stream that has not been committed and may be needed
        to retry the failed write of a chunk."""
        self._buffers: Deque[bytes] = collections.deque()
        self._offset = 0
        self._size = 0
        self._iterating = False

    def append(self, b: bytes):
        """`b` can be any length"""
        self._buffers.append(b)
        self._size += len(b)

    def size(self) -> int:
        """Return the total number of bytes stored in the write buffer.  This
        is the sum of the length of the bytes in `_buffers`."""
        return self._size

    def offset(self) -> int:
        """Return the offset in the write stream of the first byte in the
        write buffer."""
        return self._offset

    def advance_offset(self, new_offset: int):
        """Inform the write buffer that bytes before `new_offset` have been
        committed and can be discarded.  After calling advance_offset,
        `self.offset() == new_offset`."""
        assert not self._iterating
        assert new_offset <= self._offset + self._size
        while self._buffers and new_offset >= self._offset + len(self._buffers[0]):
            b = self._buffers.popleft()
            n = len(b)
            self._offset += n
            self._size -= n
        if new_offset > self._offset:
            n = new_offset - self._offset
            b = self._buffers[0]
            assert n < len(b)
            b = b[n:]
            self._buffers[0] = b
            self._offset += n
            self._size -= n
        assert self._offset == new_offset

    @contextmanager
    def chunks(self, chunk_size: int) -> Iterator[Iterator[bytes]]:
        """Return an iterator that yields bytes whose total size is
        `chunk_size` from the beginning of the write buffer."""
        assert not self._iterating

        def _chunks_iterator() -> Iterator[bytes]:
            if random.randrange(100) == 0:
                raise TransientError()

            remaining = chunk_size
            i = 0
            while remaining > 0:
                b = self._buffers[i]
                n = len(b)
                if n <= remaining:
                    yield b
                    remaining -= n
                    i += 1
                else:
                    yield b[:remaining]
                    break

        self._iterating = True
        yield _chunks_iterator()
        self._iterating = False
