import asyncio


class ANullContextManager:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


class NullWeightedSemaphore:
    def __call__(self, weight):
        return ANullContextManager()


class WeightedSemaphoreContextManager:
    def __init__(self, sem, weight):
        self.sem = sem
        self.weight = weight

    async def __aenter__(self):
        await self.sem.acquire(self.weight)

    async def __aexit__(self, exc_type, exc, tb):
        await self.sem.release(self.weight)


class WeightedSemaphore:
    def __init__(self, value=1):
        self.value = value
        self.cond = asyncio.Condition()

    async def acquire(self, weight):
        while self.value < weight:
            async with self.cond:
                await self.cond.wait()
        self.value -= weight

    async def release(self, weight):
        self.value += weight
        async with self.cond:
            self.cond.notify_all()

    def __call__(self, weight):
        return WeightedSemaphoreContextManager(self, weight)
