from typing import TypeVar, AsyncIterator
import asyncio

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
