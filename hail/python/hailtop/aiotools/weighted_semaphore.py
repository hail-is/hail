from typing import List, Optional, Tuple, Type
from types import TracebackType
import asyncio


class _AcquireManager:
    def __init__(self, ws: 'WeightedSemaphore', n: int):
        self._ws = ws
        self._n = n

    async def __aenter__(self) -> '_AcquireManager':
        await self._ws.acquire(self._n)
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        self._ws.release(self._n)


class WeightedSemaphore:
    def __init__(self, value: int):
        self.value = value
        self.events: List[Tuple[int, asyncio.Event]] = []

    def release(self, n: int) -> None:
        self.value += n
        while self.events:
            n, event = self.events[-1]
            if n >= self.value:
                self.events.pop()
                self.value -= n
                event.set()

    def acquire_manager(self, n: int) -> _AcquireManager:
        return _AcquireManager(self, n)

    async def acquire(self, n) -> None:
        if self.value >= n:
            self.value -= n
            return

        event = asyncio.Event()
        self.events.append((n, event))
        await event.wait()
