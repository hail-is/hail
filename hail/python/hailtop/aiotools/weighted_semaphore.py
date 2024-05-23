from typing import Optional, Type
from sortedcontainers import SortedKeyList
from types import TracebackType
import asyncio


class _AcquireManager:
    def __init__(self, ws: 'WeightedSemaphore', n: int):
        self._ws = ws
        self._n = n

    async def __aenter__(self) -> '_AcquireManager':
        await self._ws.acquire(self._n)
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self._ws.release(self._n)


class WeightedSemaphore:
    def __init__(self, value: int):
        self.max = value
        self.value = value
        self.events = SortedKeyList(key=lambda x: x[0])

    def release(self, n: int) -> None:
        self.value += n
        while self.events:
            n, event = self.events[0]
            if self.value >= n:
                self.events.pop(0)
                self.value -= n
                event.set()
            else:
                break

    def acquire_manager(self, n: int) -> _AcquireManager:
        return _AcquireManager(self, n)

    async def acquire(self, n) -> None:
        assert n <= self.max
        if self.value >= n:
            self.value -= n
            return

        event = asyncio.Event()
        self.events.add((n, event))
        await event.wait()
