import asyncio
import concurrent
import os


async def blocking_to_async(thread_pool, fun, *args, **kwargs):
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: fun(*args, **kwargs))


class AIOFiles:
    def __init__(self, pool=None):
        self.pool = pool or concurrent.futures.ThreadPoolExecutor()

    async def open(self, *args, **kwargs):
        return await blocking_to_async(self.pool, open, *args, **kwargs)

    async def write(self, f, *args, **kwargs):
        return await blocking_to_async(self.pool, f.write, *args, **kwargs)

    async def seek(self, f, *args, **kwargs):
        return await blocking_to_async(self.pool, f.seek, *args, **kwargs)

    async def read(self, f, *args, **kwargs):
        return await blocking_to_async(self.pool, f.read, *args, **kwargs)

    async def readinto(self, f, *args, **kwargs):
        return await blocking_to_async(self.pool, f.readinto, *args, **kwargs)

    async def mkdir(self, *args, **kwargs):
        return await blocking_to_async(self.pool, os.mkdir, *args, **kwargs)

    async def rmdir(self, *args, **kwargs):
        return await blocking_to_async(self.pool, os.rmdir, *args, **kwargs)

    async def remove(self, *args, **kwargs):
        return await blocking_to_async(self.pool, os.remove, *args, **kwargs)
