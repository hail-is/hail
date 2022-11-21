import asyncio
import collections
import logging
from typing import Deque, Tuple

log = logging.getLogger('semaphore')


class ANullContextManager:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


class NullWeightedSemaphore:
    def __call__(self, weight, id):
        return ANullContextManager()


class FIFOWeightedSemaphoreContextManager:
    def __init__(self, sem, weight):
        self.sem = sem
        self.weight = weight

    async def __aenter__(self):
        await self.sem.acquire(self.weight)

    async def __aexit__(self, exc_type, exc, tb):
        self.sem.release(self.weight)


class FIFOWeightedSemaphore:
    def __init__(self, value=1):
        self.value = value
        self.queue: Deque[Tuple[asyncio.Event, int]] = collections.deque()

    async def acquire(self, weight):
        if not self.queue and self.value >= weight:
            self.value -= weight
            return

        event = asyncio.Event()
        self.queue.append((event, weight))
        event.clear()
        await event.wait()

    def release(self, weight):
        self.value += weight
        n_notified = 0
        while self.queue:
            head_event, head_weight = self.queue[0]
            if self.value >= head_weight:
                head_event.set()
                self.queue.popleft()
                self.value -= head_weight
                n_notified += 1
            else:
                break

    def __call__(self, weight):
        return FIFOWeightedSemaphoreContextManager(self, weight)
