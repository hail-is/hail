import asyncio


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
        self.queue = []

    async def acquire(self, weight):
        event = asyncio.Event()
        self.queue.append(event)

        while self.value < weight:
            event.clear()
            await event.wait()

        self.queue.remove(event)
        self.value -= weight

    def release(self, weight):
        self.value += weight
        if self.queue:
            first = self.queue[0]
            first.set()

    def __call__(self, weight):
        return FIFOWeightedSemaphoreContextManager(self, weight)
