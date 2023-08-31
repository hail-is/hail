import asyncio
import time
from typing import Callable, Coroutine, Dict, Generic, TypeVar

import prometheus_client as pc  # type: ignore
import sortedcontainers
from prometheus_async.aio import time as prom_async_time  # type: ignore

CACHE_HITS = pc.Counter('cache_hits_count', 'Number of Cache Hits', ['cache_name'])
CACHE_MISSES = pc.Counter('cache_misses_count', 'Number of Cache Hits', ['cache_name'])
CACHE_EVICTIONS = pc.Counter('cache_evictions_count', 'Number of Cache Hits', ['cache_name'])
CACHE_LOAD_LATENCY = pc.Summary(
    'cache_load_latency_seconds', 'Latency of loading cache values in seconds', ['cache_name']
)

T = TypeVar('T')
U = TypeVar('U')


class TimeLimitedMaxSizeCache(Generic[T, U]):
    def __init__(
        self, load: Callable[[T], Coroutine[None, None, U]], lifetime_ns: int, num_slots: int, cache_name: str
    ):
        assert lifetime_ns > 0
        assert num_slots > 0
        self.load = load
        self.lifetime_ns = lifetime_ns
        self.num_slots = num_slots
        self.cache_name = cache_name
        self._futures: Dict[T, asyncio.Future] = {}
        self._cache: Dict[T, U] = {}
        self._expiry_time: Dict[T, int] = {}
        self._keys_by_expiry = sortedcontainers.SortedSet(key=lambda k: self._expiry_time[k])
        self._shutting_down = False

    async def shutdown(self):
        """Wait for all outstanding futures to complete and prevent new lookups.

        This class does not manage any resources itself and this function is *not required* to be
        called.

        """
        self._shutting_down = True
        await asyncio.wait(self._futures.values())
        assert len(self._futures) == 0

    async def lookup(self, k: T) -> U:
        if self._shutting_down:
            raise ValueError('Cache is shutting down.')

        if k in self._expiry_time:
            assert k in self._cache
            if self._expiry_time[k] <= time.monotonic_ns():
                self._remove(k)

        if k in self._cache:
            CACHE_HITS.labels(cache_name=self.cache_name).inc()
            return self._cache[k]

        CACHE_MISSES.labels(cache_name=self.cache_name).inc()
        if k in self._futures:
            return await self._futures[k]

        self._futures[k] = asyncio.create_task(self.load(k))
        try:
            v = await prom_async_time(CACHE_LOAD_LATENCY.labels(cache_name=self.cache_name), self._futures[k])  # type: ignore
        finally:
            del self._futures[k]

        self._put(k, v)

        if self._over_capacity():
            CACHE_EVICTIONS.labels(cache_name=self.cache_name).inc()
            self._evict_oldest()

        return v

    def _put(self, k: T, v: U) -> None:
        expiry_time = time.monotonic_ns() + self.lifetime_ns
        self._cache[k] = v
        self._expiry_time[k] = expiry_time
        self._keys_by_expiry.add(k)

    def _remove(self, k: T) -> None:
        del self._cache[k]
        self._keys_by_expiry.remove(k)
        del self._expiry_time[k]

    def _over_capacity(self) -> bool:
        return len(self._keys_by_expiry) > self.num_slots

    def _evict_oldest(self) -> None:
        oldest_key: T = self._keys_by_expiry[0]  # type: ignore
        self._remove(oldest_key)
