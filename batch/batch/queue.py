import asyncio


async def scale_queue_consumers(queue, f, n=1):
    for _ in range(n):
        asyncio.ensure_future(f(queue))