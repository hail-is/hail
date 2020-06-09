from types import TracebackType
from typing import Optional, Type
import collections
import time
import asyncio


class RateLimit:
    def __init__(self, count: int, window_seconds: float):
        self.count = count
        self.window_seconds = window_seconds


class RateLimiter:
    def __init__(self, rate_limit: RateLimit):
        self._count = rate_limit.count
        self._window_seconds = rate_limit.window_seconds
        # oldest leftmost
        self._items = collections.deque()

    async def __aenter__(self) -> 'RateLimiter':
        while True:
            now = time.time()

            while len(self._items) > 0 and self._items[0] <= (now - self._window_seconds):
                self._items.popleft()

            if len(self._items) < self._count:
                self._items.append(now)
                return self

            await asyncio.sleep(self._items[0] - (now - self._window_seconds))

        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        pass
