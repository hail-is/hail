from typing import Callable, TypeVar, Awaitable
import os
import errno
import random
import logging
import asyncio
import aiohttp
from aiohttp import web
import urllib3
import secrets
import socket
import requests
import google.auth.exceptions
import time

from .time import time_msecs

log = logging.getLogger('hailtop.utils')


RETRY_FUNCTION_SCRIPT = """function retry() {
    "$@" ||
        (sleep 2 && "$@") ||
        (sleep 5 && "$@");
}"""


def flatten(xxs):
    return [x for xs in xxs for x in xs]


def first_extant_file(*files):
    for f in files:
        if f is not None and os.path.isfile(f):
            return f
    return None


def secret_alnum_string(n=22, *, case=None):
    # 22 characters is math.log(62 ** 22, 2) == ~130 bits of randomness. OWASP
    # recommends at least 128 bits:
    # https://owasp.org/www-community/vulnerabilities/Insufficient_Session-ID_Length
    numbers = '0123456789'
    upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lower = 'abcdefghijklmnopqrstuvwxyz'
    if case is None:
        alphabet = numbers + upper + lower
    elif case == 'upper':
        alphabet = numbers + upper
    elif case == 'lower':
        alphabet = numbers + lower
    else:
        raise ValueError(f'invalid argument for case {case}')
    return ''.join([secrets.choice(alphabet) for _ in range(n)])


def grouped(n, ls):
    while len(ls) != 0:
        group = ls[:n]
        ls = ls[n:]
        yield group


def partition(k, ls):
    if k == 0:
        assert not ls
        return []

    assert ls
    assert k > 0
    n = len(ls)
    parts = [(n - i + k - 1) // k for i in range(k)]
    assert sum(parts) == n
    assert max(parts) - min(parts) <= 1

    def generator():
        start = 0
        for part in parts:
            yield ls[start:start + part]
            start += part
    return generator()


def unzip(lst):
    a = []
    b = []
    for x, y in lst:
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
                log.exception('worker pool caught exception')

    async def call(self, f, *args, **kwargs):
        await self._queue.put((f, args, kwargs))


class WaitableSharedPool:
    def __init__(self, worker_pool):
        self._worker_pool = worker_pool
        self._n_submitted = 0
        self._n_complete = 0
        self._waiting = False
        self._done = asyncio.Event()

    async def call(self, f, *args, **kwargs):
        assert not self._waiting
        self._n_submitted += 1

        async def invoke():
            try:
                await f(*args, **kwargs)
            finally:
                self._n_complete += 1
                if self._waiting and (self._n_complete == self._n_submitted):
                    self._done.set()

        await self._worker_pool.call(invoke)

    async def wait(self):
        assert not self._waiting
        self._waiting = True
        if self._n_complete == self._n_submitted:
            self._done.set()
        await self._done.wait()


RETRYABLE_HTTP_STATUS_CODES = {408, 500, 502, 503, 504}
if os.environ.get('HAIL_DONT_RETRY_500') == '1':
    RETRYABLE_HTTP_STATUS_CODES.remove(500)


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
    #
    # urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='www.googleapis.com', port=443): Read timed out. (read timeout=60)
    #
    # requests.exceptions.ConnectionError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))
    #
    # socket.timeout: The read operation timed out
    #
    # ConnectionResetError: [Errno 104] Connection reset by peer
    #
    # google.auth.exceptions.TransportError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))
    #
    # aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host batch.pr-6925-default-s24o4bgat8e8:80 ssl:None [Connect call failed ('10.36.7.86', 80)]
    if isinstance(e, aiohttp.ClientResponseError) and (
            e.status in RETRYABLE_HTTP_STATUS_CODES):
        # nginx returns 502 if it cannot connect to the upstream server
        # 408 request timeout, 500 internal server error, 502 bad gateway
        # 503 service unavailable, 504 gateway timeout
        return True
    if (isinstance(e, aiohttp.ClientOSError)
            and (e.errno == errno.ETIMEDOUT
                 or e.errno == errno.ECONNREFUSED
                 or e.errno == errno.EHOSTUNREACH
                 or e.errno == errno.ECONNRESET)):
        return True
    if isinstance(e, aiohttp.ServerTimeoutError):
        return True
    if isinstance(e, aiohttp.ServerDisconnectedError):
        return True
    if isinstance(e, asyncio.TimeoutError):
        return True
    if isinstance(e, aiohttp.client_exceptions.ClientConnectorError):
        return hasattr(e, 'os_error') and is_transient_error(e.os_error)
    if (isinstance(e, OSError)
            and (e.errno == errno.ETIMEDOUT
                 or e.errno == errno.ECONNREFUSED
                 or e.errno == errno.EHOSTUNREACH
                 or e.errno == errno.ECONNRESET)):
        return True
    if isinstance(e, urllib3.exceptions.ReadTimeoutError):
        return True
    if isinstance(e, requests.exceptions.ReadTimeout):
        return True
    if isinstance(e, requests.exceptions.ConnectionError):
        return True
    if isinstance(e, socket.timeout):
        return True
    if isinstance(e, ConnectionResetError):
        return True
    if isinstance(e, google.auth.exceptions.TransportError):
        return is_transient_error(e.__cause__)
    return False


async def sleep_and_backoff(delay):
    # exponentially back off, up to (expected) max of 30s
    t = delay * random.random()
    await asyncio.sleep(t)
    return min(delay * 2, 60.0)


def sync_sleep_and_backoff(delay):
    # exponentially back off, up to (expected) max of 30s
    t = delay * random.random()
    time.sleep(t)
    return min(delay * 2, 60.0)


def retry_all_errors(msg=None, error_logging_interval=10):
    async def _wrapper(f, *args, **kwargs):
        delay = 0.1
        errors = 0
        while True:
            try:
                return await f(*args, **kwargs)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception:
                errors += 1
                if msg and errors % error_logging_interval == 0:
                    log.exception(msg, stack_info=True)
            delay = await sleep_and_backoff(delay)
    return _wrapper


T = TypeVar('T')  # pylint: disable=invalid-name


async def retry_transient_errors(f: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
    delay = 0.1
    errors = 0
    while True:
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            if not is_transient_error(e):
                raise
            errors += 1
            if errors % 10 == 0:
                log.warning(f'encountered {errors} errors, most recent one was {e}', exc_info=True)
        delay = await sleep_and_backoff(delay)


def sync_retry_transient_errors(f, *args, **kwargs):
    delay = 0.1
    errors = 0
    while True:
        try:
            return f(*args, **kwargs)
        except Exception as e:
            errors += 1
            if errors % 10 == 0:
                log.warning(f'encountered {errors} errors, most recent one was {e}', exc_info=True)
            if is_transient_error(e):
                pass
            else:
                raise
        delay = sync_sleep_and_backoff(delay)


async def request_retry_transient_errors(session, method, url, **kwargs):
    return await retry_transient_errors(session.request, method, url, **kwargs)


async def request_raise_transient_errors(session, method, url, **kwargs):
    try:
        return await session.request(method, url, **kwargs)
    except Exception as e:
        if is_transient_error(e):
            log.exception('request failed with transient exception: {method} {url}')
            raise web.HTTPServiceUnavailable()
        raise


def retry_response_returning_functions(fun, *args, **kwargs):
    delay = 0.1
    errors = 0
    response = sync_retry_transient_errors(
        fun, *args, **kwargs)
    while response.status_code in RETRYABLE_HTTP_STATUS_CODES:
        errors += 1
        if errors % 10 == 0:
            log.warning(f'encountered {errors} bad status codes, most recent '
                        f'one was {response.status_code}', exc_info=True)
        response = sync_retry_transient_errors(
            fun, *args, **kwargs)
        delay = sync_sleep_and_backoff(delay)
    return response


async def collect_agen(agen):
    return [x async for x in agen]


async def retry_long_running(name, f, *args, **kwargs):
    delay_secs = 0.1
    while True:
        try:
            start_time = time_msecs()
            return await f(*args, **kwargs)
        except Exception:
            end_time = time_msecs()

            log.exception(f'in {name}')

            t = delay_secs * random.uniform(0.7, 1.3)
            await asyncio.sleep(t)

            ran_for_secs = (end_time - start_time) * 1000
            delay_secs = min(
                max(0.1, 2 * delay_secs - min(0, (ran_for_secs - t) / 2)),
                30.0)


async def run_if_changed(changed, f, *args, **kwargs):
    while True:
        changed.clear()
        should_wait = await f(*args, **kwargs)
        if should_wait:
            await changed.wait()


class LoggingTimerStep:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time_msecs()

    async def __aexit__(self, exc_type, exc, tb):
        finish_time = time_msecs()
        self.timer.timing[self.name] = finish_time - self.start_time


class LoggingTimer:
    def __init__(self, description, threshold_ms=None):
        self.description = description
        self.threshold_ms = threshold_ms
        self.timing = {}
        self.start_time = None

    def step(self, name):
        return LoggingTimerStep(self, name)

    async def __aenter__(self):
        self.start_time = time_msecs()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        finish_time = time_msecs()
        total = finish_time - self.start_time
        if self.threshold_ms is None or total > self.threshold_ms:
            self.timing['total'] = total
            log.info(f'{self.description} timing {self.timing}')
