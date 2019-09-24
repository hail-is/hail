import asyncio


async def blocking_to_async(thread_pool, fun, *args, **kwargs):
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: fun(*args, **kwargs))
