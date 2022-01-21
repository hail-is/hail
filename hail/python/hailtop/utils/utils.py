from typing import Callable, TypeVar, Awaitable, Optional, Type, List, Dict, Iterable, Tuple
from typing_extensions import Literal
from types import TracebackType
import concurrent
import subprocess
import traceback
import sys
import os
import re
import errno
import random
import logging
import asyncio
import aiohttp
from aiohttp import web
import urllib
import urllib3
import secrets
import socket
import requests
import google.auth.exceptions
import google.api_core.exceptions
import botocore.exceptions
import time
import weakref
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

from .time import time_msecs

try:
    import aiodocker  # pylint: disable=import-error
except ModuleNotFoundError:
    aiodocker = None


log = logging.getLogger('hailtop.utils')


RETRY_FUNCTION_SCRIPT = """function retry() {
    "$@" ||
        (sleep 2 && "$@") ||
        (sleep 5 && "$@");
}"""


T = TypeVar('T')  # pylint: disable=invalid-name


def unpack_comma_delimited_inputs(inputs):
    return [s.strip()
            for steps in inputs
            for step in steps
            for s in step.split(',') if s.strip()]


def unpack_key_value_inputs(inputs):
    key_values = [i.split('=') for i in unpack_comma_delimited_inputs(inputs)]
    return {kv[0]: kv[1] for kv in key_values}


def flatten(xxs):
    return [x for xs in xxs for x in xs]


def filter_none(xs: Iterable[Optional[T]]) -> List[T]:
    return [x for x in xs if x is not None]


def first_extant_file(*files: Optional[str]) -> Optional[str]:
    for f in files:
        if f is not None and os.path.isfile(f):
            return f
    return None


def cost_str(cost):
    if cost is None:
        return None
    return f'${cost:.4f}'


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
    elif case == 'numbers':
        alphabet = numbers
    else:
        raise ValueError(f'invalid argument for case {case}')
    return ''.join([secrets.choice(alphabet) for _ in range(n)])


def digits_needed(i: int):
    assert i >= 0
    if i < 10:
        return 1
    return 1 + digits_needed(i // 10)


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


def async_to_blocking(coro: Awaitable[T]) -> T:
    return asyncio.get_event_loop().run_until_complete(coro)


async def blocking_to_async(thread_pool: concurrent.futures.Executor,
                            fun: Callable[..., T],
                            *args,
                            **kwargs) -> T:
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
            try:
                worker.cancel()
            except Exception:
                pass

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
        try:
            if self.count > 0:
                await self._done.wait()
        finally:
            self._cancel_workers()

        if self._errors:
            raise self._errors[0]

        return self._results


class AsyncWorkerPool:
    def __init__(self, parallelism, queue_size=1000):
        self._queue = asyncio.Queue(maxsize=queue_size)
        self.workers = weakref.WeakSet([
            asyncio.ensure_future(self._worker())
            for _ in range(parallelism)])

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

    def call_nowait(self, f, *args, **kwargs):
        self._queue.put_nowait((f, args, kwargs))

    def shutdown(self):
        for worker in self.workers:
            try:
                worker.cancel()
            except Exception:
                pass


class WaitableSharedPool:
    def __init__(self, worker_pool: AsyncWorkerPool):
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

    async def __aenter__(self) -> 'WaitableSharedPool':
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self.wait()


class WithoutSemaphore:
    def __init__(self, sema):
        self._sema = sema

    async def __aenter__(self) -> 'WithoutSemaphore':
        self._sema.release()
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self._sema.acquire()


class PoolShutdownError(Exception):
    pass


class OnlineBoundedGather2:
    '''`OnlineBoundedGather2` provides the capability to run background
    tasks with bounded parallelism.  It is a context manager, and
    waits for all background tasks to complete on exit.

    `OnlineBoundedGather2` supports cancellation of background tasks.
    When a background task raises `asyncio.CancelledError`, the task
    is considered complete and the pool and other background tasks
    continue runnning.

    If a background task fails (raises an exception besides
    `asyncio.CancelledError`), all running background tasks are
    cancelled and the the pool is shut down.  Subsequent calls to
    `OnlineBoundedGather2.call()` raise `PoolShutdownError`.

    Because the pool runs tasks in the background, multiple exceptions
    can occur simultaneously.  The first exception raised, whether by
    a background task or into the context manager exit, is raised by
    the context manager exit, and any further exceptions are logged
    and otherwise discarded.
    '''

    def __init__(self, sema: asyncio.Semaphore):
        self._counter = 0
        self._sema = sema
        self._pending: Optional[Dict[int, asyncio.Task]] = {}
        # done if there are no pending tasks (the tasks are all
        # complete), or if we've shutdown and the cancelled tasks are
        # complete
        self._done_event = asyncio.Event()
        # not pending tasks, so done
        self._done_event.set()
        self._exception: Optional[BaseException] = None

    async def _shutdown(self) -> None:
        '''Shut down the pool.

        Cancel all pending tasks and wait for them to complete.
        Subsequent calls to call will raise `PoolShutdownError`.
        '''

        if self._pending is None:
            return

        # shut down the pending tasks
        tasks = []
        for _, t in self._pending.items():
            if not t.done():
                t.cancel()
            tasks.append(t)
        self._pending = None

        if tasks:
            await asyncio.wait(tasks)

        self._done_event.set()

    def call(self, f, *args, **kwargs) -> asyncio.Task:
        '''Invoke a function as a background task.

        Return the task, which can be used to wait on (using
        `OnlineBoundedGather2.wait()`) or cancel the task (using
        `asyncio.Task.cancel()`).  Note, waiting on a task using
        `asyncio.wait()` directly can lead to deadlock.
        '''

        if self._pending is None:
            raise PoolShutdownError

        id = self._counter
        self._counter += 1

        async def run_and_cleanup():
            try:
                async with self._sema:
                    await f(*args, **kwargs)
            except asyncio.CancelledError:
                pass
            except:
                if self._exception is None:
                    _, exc, _ = sys.exc_info()
                    self._exception = exc
                    await asyncio.shield(self._shutdown())
                else:
                    log.info('discarding exception', exc_info=True)

            if self._pending is None:
                return
            del self._pending[id]
            if not self._pending:
                self._done_event.set()

        t = asyncio.create_task(run_and_cleanup())
        self._pending[id] = t
        self._done_event.clear()
        return t

    async def wait(self, tasks: List[asyncio.Task]) -> None:
        '''Wait for a list of tasks returned to complete.

        The tasks should be tasks returned from
        `OnlineBoundedGather2.call()`.  They can be a subset of the
        running tasks, `OnlineBoundedGather2.wait()` can be called
        multiple times, and additional tasks can be submitted to the
        pool after waiting.
        '''

        async with WithoutSemaphore(self._sema):
            await asyncio.wait(tasks)

    async def __aenter__(self) -> 'OnlineBoundedGather2':
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        if exc_val:
            if self._exception is None:
                self._exception = exc_val
                await self._shutdown()
            else:
                log.info('discarding exception', exc_info=exc_val)

        # wait for done and not pending _done_event.wait can return
        # when when there are pending jobs if the last job completed
        # (setting _done_event) and then more tasks were submitted
        async with WithoutSemaphore(self._sema):
            await self._done_event.wait()
        while self._pending:
            assert not self._done_event.is_set()
            async with WithoutSemaphore(self._sema):
                await self._done_event.wait()

        if self._exception:
            raise self._exception


async def bounded_gather2_return_exceptions(sema: asyncio.Semaphore, *pfs):
    '''Run the partial functions `pfs` as tasks with parallelism bounded
    by `sema`, which should be `asyncio.Semaphore` whose initial value
    is the desired level of parallelism.

    The return value is the list of partial function results as pairs:
    the pair `(value, None)` if the partial function returned value or
    `(None, exc)` if the partial function raised the exception `exc`.

    '''
    async def run_with_sema_return_exceptions(pf):
        try:
            async with sema:
                return (await pf(), None)
        except:
            _, exc, _ = sys.exc_info()
            return (None, exc)

    tasks = [asyncio.create_task(run_with_sema_return_exceptions(pf)) for pf in pfs]
    async with WithoutSemaphore(sema):
        return await asyncio.gather(*tasks)


async def bounded_gather2_raise_exceptions(sema: asyncio.Semaphore, *pfs, cancel_on_error: bool = False):
    '''Run the partial functions `pfs` as tasks with parallelism bounded
    by `sema`, which should be `asyncio.Semaphore` whose initial value
    is the level of parallelism.

    The return value is the list of partial function results.

    The first exception raised by a partial function is raised by
    bounded_gather2_raise_exceptions.

    If cancel_on_error is False (the default), the remaining partial
    functions continue to run with bounded parallelism.  If
    cancel_on_error is True, the unfinished tasks are all cancelled.

    '''
    async def run_with_sema(pf):
        async with sema:
            return await pf()

    tasks = [asyncio.create_task(run_with_sema(pf)) for pf in pfs]

    if not cancel_on_error:
        async with WithoutSemaphore(sema):
            return await asyncio.gather(*tasks)

    try:
        async with WithoutSemaphore(sema):
            return await asyncio.gather(*tasks)
    finally:
        _, exc, _ = sys.exc_info()
        if exc is not None:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                async with WithoutSemaphore(sema):
                    await asyncio.wait(tasks)


async def bounded_gather2(sema: asyncio.Semaphore, *pfs, return_exceptions: bool = False, cancel_on_error: bool = False):
    if return_exceptions:
        return await bounded_gather2_return_exceptions(sema, *pfs)
    return await bounded_gather2_raise_exceptions(sema, *pfs, cancel_on_error=cancel_on_error)


RETRYABLE_HTTP_STATUS_CODES = {408, 500, 502, 503, 504, 429}
if os.environ.get('HAIL_DONT_RETRY_500') == '1':
    RETRYABLE_HTTP_STATUS_CODES.remove(500)


class TransientError(Exception):
    pass


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
    #
    # OSError: [Errno 51] Connect call failed ('35.188.91.25', 443)
    # https://hail.zulipchat.com/#narrow/stream/223457-Batch-support/topic/ssl.20error
    import hailtop.aiocloud.aiogoogle.client.compute_client  # pylint: disable=import-outside-toplevel,cyclic-import
    import hailtop.httpx  # pylint: disable=import-outside-toplevel,cyclic-import
    if isinstance(e, aiohttp.ClientResponseError) and (
            e.status in RETRYABLE_HTTP_STATUS_CODES):
        # nginx returns 502 if it cannot connect to the upstream server
        # 408 request timeout, 500 internal server error, 502 bad gateway
        # 503 service unavailable, 504 gateway timeout
        # 429 "Temporarily throttled, too many requests"
        return True
    if (isinstance(e, hailtop.aiocloud.aiogoogle.client.compute_client.GCPOperationError)
            and 'QUOTA_EXCEEDED' in e.error_codes):
        return True
    if isinstance(e, hailtop.httpx.ClientResponseError) and (
            (e.status in RETRYABLE_HTTP_STATUS_CODES) or (
                e.status == 403 and 'rateLimitExceeded' in e.body)):
        return True
    if isinstance(e, aiohttp.ServerTimeoutError):
        return True
    if isinstance(e, aiohttp.ServerDisconnectedError):
        return True
    if isinstance(e, asyncio.TimeoutError):
        return True
    if isinstance(e, aiohttp.client_exceptions.ClientConnectorError):
        return hasattr(e, 'os_error') and is_transient_error(e.os_error)
    # appears to happen when the connection is lost prematurely, see:
    # https://github.com/aio-libs/aiohttp/issues/4581
    # https://github.com/aio-libs/aiohttp/blob/v3.7.4/aiohttp/client_proto.py#L85
    if (isinstance(e, aiohttp.ClientPayloadError)
            and e.args[0] == "Response payload is not completed"):
        return True
    if (isinstance(e, OSError)
            and e.errno in (errno.ETIMEDOUT,
                            errno.ECONNREFUSED,
                            errno.EHOSTUNREACH,
                            errno.ECONNRESET,
                            errno.ENETUNREACH,
                            errno.EPIPE,
                            errno.ETIMEDOUT
                            )):
        return True
    if isinstance(e, aiohttp.ClientOSError):
        # aiohttp/client_reqrep.py wraps all OSError instances with a ClientOSError
        return is_transient_error(e.__cause__)
    if isinstance(e, urllib3.exceptions.ReadTimeoutError):
        return True
    if isinstance(e, requests.exceptions.ReadTimeout):
        return True
    if isinstance(e, requests.exceptions.ConnectionError):
        return True
    if isinstance(e, socket.timeout):
        return True
    if isinstance(e, socket.gaierror):
        # socket.EAI_AGAIN: [Errno -3] Temporary failure in name resolution
        # socket.EAI_NONAME: [Errno 8] nodename nor servname provided, or not known
        return e.errno in (socket.EAI_AGAIN, socket.EAI_NONAME)
    if isinstance(e, ConnectionResetError):
        return True
    if isinstance(e, google.auth.exceptions.TransportError):
        return is_transient_error(e.__cause__)
    if isinstance(e, google.api_core.exceptions.GatewayTimeout):
        return True
    if isinstance(e, google.api_core.exceptions.ServiceUnavailable):
        return True
    if isinstance(e, botocore.exceptions.ConnectionClosedError):
        return True
    if aiodocker is not None and isinstance(e, aiodocker.exceptions.DockerError):
        # aiodocker.exceptions.DockerError: DockerError(500, 'Get https://gcr.io/v2/: net/http: request canceled (Client.Timeout exceeded while awaiting headers)')
        return e.status == 500 and 'Client.Timeout exceeded while awaiting headers' in e.message
    if isinstance(e, TransientError):
        return True
    return False


async def sleep_and_backoff(delay, max_delay=30.0):
    # exponentially back off, up to (expected) max_delay
    t = delay * random.uniform(0.9, 1.1)
    await asyncio.sleep(t)
    return min(delay * 2, max_delay)


def sync_sleep_and_backoff(delay):
    # exponentially back off, up to (expected) max of 30s
    t = delay * random.uniform(0.9, 1.1)
    time.sleep(t)
    return min(delay * 2, 30.0)


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


def retry_all_errors_n_times(max_errors=10, msg=None, error_logging_interval=10):
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
                if errors >= max_errors:
                    raise
            delay = await sleep_and_backoff(delay)
    return _wrapper


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
                st = ''.join(traceback.format_stack())
                log.warning(f'Encountered {errors} errors. My stack trace is {st}. Most recent error was {e}', exc_info=True)
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
                st = ''.join(traceback.format_stack())
                log.warning(f'Encountered {errors} errors. My stack trace is {st}. Most recent error was {e}', exc_info=True)
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
                        f'one was {response.status_code}')
        response = sync_retry_transient_errors(
            fun, *args, **kwargs)
        delay = sync_sleep_and_backoff(delay)
    return response


def external_requests_client_session(headers=None, timeout=5) -> requests.Session:
    session = requests.Session()
    adapter = TimeoutHTTPAdapter(max_retries=1, timeout=timeout)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    if headers:
        session.headers = headers
    return session


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, max_retries, timeout):
        self.max_retries = max_retries
        self.timeout = timeout
        super().__init__(max_retries=max_retries)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            retries=self.max_retries,
            timeout=self.timeout)


async def collect_agen(agen):
    return [x async for x in agen]


def dump_all_stacktraces():
    for t in asyncio.all_tasks():
        log.debug(t)
        t.print_stack()


async def retry_long_running(name, f, *args, **kwargs):
    delay_secs = 0.1
    while True:
        try:
            start_time = time_msecs()
            return await f(*args, **kwargs)
        except asyncio.CancelledError:
            raise
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


async def run_if_changed_idempotent(changed, f, *args, **kwargs):
    while True:
        should_wait = await f(*args, **kwargs)
        changed.clear()
        if should_wait:
            await changed.wait()


async def periodically_call(period, f, *args, **kwargs):
    async def loop():
        log.info(f'starting loop for {f.__name__}')
        while True:
            await f(*args, **kwargs)
            await asyncio.sleep(period)
    await retry_long_running(f.__name__, loop)


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


def url_basename(url: str) -> str:
    """Return the basename of the path of the URL `url`."""
    return os.path.basename(urllib.parse.urlparse(url).path)


def url_join(url: str, path: str) -> str:
    """Join the (relative or absolute) path `path` to the URL `url`."""
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(parsed._replace(path=os.path.join(parsed.path, path)))


def url_scheme(url: str) -> str:
    """Return scheme of `url`, or the empty string if there is no scheme."""
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme


def url_and_params(url: str) -> Tuple[str, Dict[str, str]]:
    """Strip the query parameters from `url` and parse them into a dictionary.
       Assumes that all query parameters are used only once, so have only one
       value.
    """
    parsed = urllib.parse.urlparse(url)
    params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
    without_query = urllib.parse.urlunparse(parsed._replace(query=''))
    return without_query, params


RegistryProvider = Literal['google', 'azure', 'dockerhub']


class ParsedDockerImageReference:
    def __init__(self, domain: str, path: str, tag: str, digest: str):
        self.domain = domain
        self.path = path
        self.tag = tag
        self.digest = digest

    def name(self) -> str:
        if self.domain:
            return self.domain + '/' + self.path
        return self.path

    def hosted_in(self, registry: RegistryProvider) -> bool:
        if registry == 'google':
            return self.domain is not None and (self.domain == 'gcr.io' or self.domain.endswith('docker.pkg.dev'))
        if registry == 'azure':
            return self.domain is not None and self.domain.endswith('azurecr.io')
        assert registry == 'dockerhub'
        return self.domain is None or self.domain == 'docker.io'

    def __str__(self):
        s = self.name()
        if self.tag is not None:
            s += ':'
            s += self.tag
        if self.digest is not None:
            s += '@'
            s += self.digest
        return s


# https://github.com/distribution/distribution/blob/v2.7.1/reference/reference.go
DOCKER_IMAGE_REFERENCE_REGEX = re.compile(r"(?:([^/]+)/)?([^:@]+)(?::([^@]+))?(?:@(.+))?")


def parse_docker_image_reference(reference_string: str) -> ParsedDockerImageReference:
    match = DOCKER_IMAGE_REFERENCE_REGEX.fullmatch(reference_string)
    if match is None:
        raise ValueError(f'could not parse {reference_string!r} as a docker image reference')
    domain, path, tag, digest = (match.group(i + 1) for i in range(4))
    return ParsedDockerImageReference(domain, path, tag, digest)


class Notice:
    def __init__(self):
        self.subscribers = []

    def subscribe(self):
        e = asyncio.Event()
        self.subscribers.append(e)
        return e

    def notify(self):
        for e in self.subscribers:
            e.set()


def find_spark_home():
    spark_home = os.environ.get('SPARK_HOME')
    if spark_home is None:
        find_spark_home = subprocess.run('find_spark_home.py',
                                         capture_output=True,
                                         check=False)
        if find_spark_home.returncode != 0:
            raise ValueError(f'''SPARK_HOME is not set and find_spark_home.py returned non-zero exit code:
STDOUT:
{find_spark_home.stdout}
STDERR:
{find_spark_home.stderr}''')
        spark_home = find_spark_home.stdout.decode().strip()
    return spark_home
