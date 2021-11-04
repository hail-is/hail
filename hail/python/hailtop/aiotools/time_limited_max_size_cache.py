from typing import TypeVar, Callable, Awaitable, Dict, Tuple, Generic

import time
import asyncio
import sortedcontainers


T = TypeVar('T')
U = TypeVar('U')


class TimeLimitedMaxSizeCache(Generic[T, U]):
    def __init__(self, load: Callable[[T], Awaitable[U]], lifetime_ns: int, max_size: int):
        assert lifetime_ns > 0
        assert max_size > 0
        self.load = load
        self.lifetime_ns = lifetime_ns
        self.max_size = max_size
        self._futures: Dict[T, asyncio.Future] = {}
        self._cache: Dict[T, Tuple[U, int]] = {}
        self._keys_with_expiry = sortedcontainers.SortedSet(key=lambda t: t[1])
        self._shutting_down = False

    async def shutdown(self, cleanup_value: Callable[[U], Awaitable[None]]):
        """Wait for all outstanding futures to complete and cleanup values.

        It is not necessary to shutdown a TimeLimitedMaxSizeCache unless the values need to be
        cleaned up.

        """
        self._shutting_down = True
        await asyncio.wait(self._futures.values())
        assert len(self._futures) == 0
        for u, _ in self._cache.values():
            await cleanup_value(u)

    async def lookup(self, k: T) -> U:
        if self._shutting_down:
            raise ValueError('Cache is shutting down.')

        # if self._cache[k] is present, then self._futures[k] is not present
        # self._cache[k] == _, expiry_time  iff  (k, expiry_time) in self._keys_with_expiry
        try:
            try:
                v, expiry_time = self._cache[k]
            except KeyError:
                future_value = self._futures[k]
                return await future_value

            if expiry_time <= time.monotonic_ns():
                del self._cache[k]
                self._keys_with_expiry.remove((k, expiry_time))
                raise KeyError
            return v
        except KeyError:
            # self._cache[k] is not present
            # self._futures[k] is not present
            self._futures[k] = asyncio.Future()

            try:
                v = await self.load(k)
            except Exception as exc:
                self._futures[k].set_exception(exc)
                self._futures[k].cancel()  # prevent "unretrieved exception" logs
                del self._futures[k]
                raise exc

            expiry_time = time.monotonic_ns() + self.lifetime_ns
            self._futures[k].set_result(v)
            self._cache[k] = (v, expiry_time)
            self._keys_with_expiry.add((k, expiry_time))
            del self._futures[k]

            if len(self._keys_with_expiry) > self.max_size:
                oldest_key, _ = self._keys_with_expiry.pop(0)
                del self._cache[oldest_key]
            return v
