import asyncio


def unzip(l):
    a = []
    b = []
    for x, y in l:
        a.append(x)
        b.append(y)
    return a, b


def async_to_blocking(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def blocking_to_async(thread_pool, fun, *args, **kwargs):
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: fun(*args, **kwargs))
