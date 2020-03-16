import asyncio
import collections
import logging

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
    def __init__(self, sem, weight, id):
        self.sem = sem
        self.weight = weight
        self.id = id

    async def __aenter__(self):
        await self.sem.acquire(self.weight, self.id)

    async def __aexit__(self, exc_type, exc, tb):
        self.sem.release(self.weight, self.id)


class FIFOWeightedSemaphore:
    def __init__(self, value=1):
        self.value = value
        self.queue = collections.deque()

    async def acquire(self, weight, id):
        if not self.queue and self.value >= weight:
            self.value -= weight
            log.info(f'acquired semaphore for {id} with weight {weight} without waiting, new value = {self.value}')
            return

        event = asyncio.Event()
        self.queue.append((event, weight, id))
        event.clear()
        await event.wait()

    def release(self, weight, id):
        self.value += weight
        log.info(f'released semaphore for {id} with weight {weight}, new value = {self.value}')

        n_notified = 0
        log.info(f'found {len(self.queue)} events waiting in the queue')
        while self.queue:
            head_event, head_weight, head_id = self.queue[0]
            if self.value >= head_weight:
                head_event.set()
                self.queue.popleft()
                self.value -= head_weight
                n_notified += 1
                log.info(f'notified event for {head_id} with weight {head_weight}, new value = {self.value}')
            else:
                log.info(f'notified {n_notified} events before being blocked by {head_id} with weight {head_weight}, value = {self.value}')
                break

    def __call__(self, weight, id):
        return FIFOWeightedSemaphoreContextManager(self, weight, id)
