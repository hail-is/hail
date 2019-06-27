import asyncio


async def scale_queue_consumers(queue, f, n=1, loop=None):
    for _ in range(n):
        asyncio.ensure_future(f(queue), loop=loop)