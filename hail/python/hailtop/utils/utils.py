import errno
import random
import logging
import asyncio
import aiohttp

log = logging.getLogger('hailtop.utils')


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


class AsyncWorkerPool:
    def __init__(self, parallelism):
        self._sem = asyncio.Semaphore(parallelism)
        self._count = 0
        self._done = asyncio.Event()

    async def _call(self, f, args, kwargs):
        async with self._sem:
            try:
                await f(*args, **kwargs)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception(f'worker pool caught exception')
            finally:
                assert self._count > 0
                self._count -= 1
                if self._count == 0:
                    self._done.set()

    async def call(self, f, *args, **kwargs):
        if self._count == 0:
            self._done.clear()
        self._count += 1
        asyncio.ensure_future(self._call(f, args, kwargs))

    async def wait(self):
        await self._done.wait()


async def request_retry_transient_errors(f, *args, **kwargs):
    delay = 0.1
    while True:
        try:
            return await f(*args, **kwargs)
        # observed exceptions:
        # aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host <host> ssl:None [Connect call failed ('<ip>', 80)]
        except aiohttp.ClientResponseError as e:
            # 408 request timeout, 503 service unavailable, 504 gateway timeout
            if e.status == 408 or e.status == 503 or e.status == 504:
                pass
            else:
                raise
        except aiohttp.ClientOSError as e:
            if e.errno == errno.ETIMEDOUT:
                pass
            else:
                log.exception('request failed ClientOSError errno {e.errno} {os.strerror(e.errno)}')
                raise
        except aiohttp.ServerTimeoutError:
            pass
        # exponentially back off, up to (expected) max of 30s
        delay = min(delay * 2, 60.0)
        t = delay * random.random()
        await asyncio.sleep(t)
