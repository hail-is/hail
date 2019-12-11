import errno
import random
import logging
import asyncio
import aiohttp
from aiohttp import web

log = logging.getLogger('hailtop.utils')


def grouped(n, ls):
    while len(ls) != 0:
        group = ls[:n]
        ls = ls[n:]
        yield group


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


async def bounded_gather(*pfs, parallelism=10, return_exceptions=False):
    gatherer = AsyncThrottledGather(*pfs,
                                    parallelism=parallelism,
                                    return_exceptions=return_exceptions)
    return await gatherer.wait()


class AsyncThrottledGather:
    def __init__(self, *pfs, parallelism=10, return_exceptions=False):
        self.count = len(pfs)
        self.n_finished = 0

        self._queue = asyncio.Queue()
        self._done = asyncio.Event()
        self._return_exceptions = return_exceptions

        self._results = [None] * len(pfs)
        self._errors = []

        self._workers = []
        for _ in range(parallelism):
            self._workers.append(asyncio.ensure_future(self._worker()))

        for i, pf in enumerate(pfs):
            self._queue.put_nowait((i, pf))

    def _cancel_workers(self):
        for worker in self._workers:
            worker.cancel()

    async def _worker(self):
        while True:
            i, pf = await self._queue.get()

            try:
                res = await pf()
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as err:  # pylint: disable=broad-except
                res = err
                if not self._return_exceptions:
                    self._errors.append(err)
                    self._done.set()
                    return

            self._results[i] = res
            self.n_finished += 1

            if self.n_finished == self.count:
                self._done.set()

    async def wait(self):
        if self.count > 0:
            await self._done.wait()

        self._cancel_workers()

        if self._errors:
            raise self._errors[0]

        return self._results


class AsyncWorkerPool:
    def __init__(self, parallelism, queue_size=1000):
        self._queue = asyncio.Queue(maxsize=queue_size)

        for _ in range(parallelism):
            asyncio.ensure_future(self._worker())

    async def _worker(self):
        while True:
            f, args, kwargs = await self._queue.get()
            try:
                await f(*args, **kwargs)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:  # pylint: disable=broad-except
                log.exception(f'worker pool caught exception')

    async def call(self, f, *args, **kwargs):
        await self._queue.put((f, args, kwargs))


def is_transient_error(e):
    # observed exceptions:
    #
    # aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host <host> ssl:None [Connect call failed ('<ip>', 80)]
    #
    # concurrent.futures._base.TimeoutError
    #   from aiohttp/helpers.py:585:TimerContext: raise asyncio.TimeoutError from None
    #
    # Connected call failed caused by:
    # OSError: [Errno 113] Connect call failed ('<ip>', 80)
    # 113 is EHOSTUNREACH: No route to host
    #
    # Fatal read error on socket transport
    # protocol: <asyncio.sslproto.SSLProtocol object at 0x12b47d320>
    # transport: <_SelectorSocketTransport fd=13 read=polling write=<idle, bufsize=0>>
    # Traceback (most recent call last):
    #   File "/anaconda3/lib/python3.7/asyncio/selector_events.py", line 812, in _read_ready__data_received
    #     data = self._sock.recv(self.max_size)
    # TimeoutError: [Errno 60] Operation timed out
    #
    # Traceback (most recent call last):
    #   ...
    #   File "/usr/local/lib/python3.6/dist-packages/aiohttp/client.py", line 505, in _request
    #     await resp.start(conn)
    #   File "/usr/local/lib/python3.6/dist-packages/aiohttp/client_reqrep.py", line 848, in start
    #     message, payload = await self._protocol.read()  # type: ignore  # noqa
    #   File "/usr/local/lib/python3.6/dist-packages/aiohttp/streams.py", line 592, in read
    #     await self._waiter
    # aiohttp.client_exceptions.ServerDisconnectedError: None
    #
    # during aiohttp request
    # aiohttp.client_exceptions.ClientOSError: [Errno 104] Connection reset by peer
    if isinstance(e, aiohttp.ClientResponseError):
        # nginx returns 502 if it cannot connect to the upstream server
        # 408 request timeout, 502 bad gateway, 503 service unavailable, 504 gateway timeout
        if e.status == 408 or e.status == 502 or e.status == 503 or e.status == 504:
            return True
    elif isinstance(e, aiohttp.ClientOSError):
        if (e.errno == errno.ETIMEDOUT or
                e.errno == errno.ECONNREFUSED or
                e.errno == errno.EHOSTUNREACH):
            return True
    elif isinstance(e, aiohttp.ServerTimeoutError):
        return True
    elif isinstance(e, aiohttp.ServerDisconnectedError):
        return True
    elif isinstance(e, asyncio.TimeoutError):
        return True
    elif isinstance(e, OSError):
        if (e.errno == errno.ETIMEDOUT or
                e.errno == errno.ECONNREFUSED or
                e.errno == errno.EHOSTUNREACH or
                e.errno == errno.ECONNRESET):
            return True
    return False


async def sleep_and_backoff(delay):
    # exponentially back off, up to (expected) max of 30s
    t = delay * random.random()
    await asyncio.sleep(t)
    return min(delay * 2, 60.0)


async def request_retry_transient_errors(session, method, url, **kwargs):
    delay = 0.1
    while True:
        try:
            return await session.request(method, url, **kwargs)
        except Exception as e:
            if is_transient_error(e):
                pass
            else:
                raise
        delay = await sleep_and_backoff(delay)


async def request_raise_transient_errors(session, method, url, **kwargs):
    try:
        return await session.request(method, url, **kwargs)
    except Exception as e:
        if is_transient_error(e):
            log.exception('request failed with transient exception: {method} {url}')
            raise web.HTTPServiceUnavailable()
        raise


async def collect_agen(agen):
    return [x async for x in agen]
