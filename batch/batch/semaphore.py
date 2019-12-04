import asyncio
import sortedcontainers
import time


class ANullContextManager:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


class NullWeightedSemaphore:
    def __call__(self, weight):
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
        self.event_age = {}
        self.queue = sortedcontainers.SortedSet(key=lambda event: self.event_age[event])

    async def acquire(self, weight):
        event = asyncio.Event()
        age = time.time()
        self.event_age[event] = age
        self.queue.add(event)

        while self.value < weight:
            event.clear()
            await event.wait()

        self.queue.remove(event)
        del self.event_age[event]
        self.value -= weight

    def release(self, weight):
        self.value += weight
        if self.queue:
            first = self.queue[0]
            first.set()

    def __call__(self, weight):
        return FIFOWeightedSemaphoreContextManager(self, weight)
