from typing import (Any, Callable, TypeVar, Awaitable, Mapping, Optional, Type, List, Dict, Iterable, Tuple,
                    Generic, cast, AsyncIterator, Iterator, Union)
from typing import Literal, Sequence
from typing_extensions import ParamSpec
from types import TracebackType
import concurrent.futures
import contextlib
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
import urllib.parse
import urllib3.exceptions
import secrets
import socket
import requests
import botocore.exceptions
import itertools
import time
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

from .time import time_msecs

try:
    import aiodocker  # pylint: disable=import-error
except ModuleNotFoundError:
    aiodocker = None  # type: ignore


log = logging.getLogger('hailtop.utils')


RETRY_FUNCTION_SCRIPT = """function retry() {
    "$@" ||
        (sleep 2 && "$@") ||
        (sleep 5 && "$@");
}"""


T = TypeVar('T')  # pylint: disable=invalid-name
U = TypeVar('U')  # pylint: disable=invalid-name
P = ParamSpec("P")


def unpack_comma_delimited_inputs(inputs: List[str]) -> List[str]:
    return [s.strip()
            for comma_separated_steps in inputs
            for s in comma_separated_steps.split(',') if s.strip()]


def unpack_key_value_inputs(inputs: List[str]) -> Dict[str, str]:
    key_values = [i.split('=') for i in unpack_comma_delimited_inputs(inputs)]
    return {kv[0]: kv[1] for kv in key_values}


def flatten(xxs: Iterable[Iterable[T]]) -> List[T]:
    return [x for xs in xxs for x in xs]


def filter_none(xs: Iterable[Optional[T]]) -> List[T]:
    return [x for x in xs if x is not None]


def first_extant_file(*files: Optional[str]) -> Optional[str]:
    for f in files:
        if f is not None and os.path.isfile(f):
            return f
    return None


def cost_str(cost: Optional[float]) -> Optional[str]:
    if cost is None:
        return None
    if cost == 0.0:
        return '$0.0000'
    if cost < 0.0001:
        return '<$0.0001'
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


def digits_needed(i: int) -> int:
    assert i >= 0
    if i < 10:
        return 1
    return 1 + digits_needed(i // 10)


def grouped(n: int, ls: Iterable[T]) -> Iterable[List[T]]:  # replace with itertools.batched in Python 3.12
    it = iter(ls)
    if n < 1:
        raise ValueError('invalid value for n: found {n}')
    while True:
        group = list(itertools.islice(it, n))
        if len(group) == 0:
            break
        yield group


def partition(k: int, ls: Sequence[T]) -> Iterable[Sequence[T]]:
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


def unzip(lst: Iterable[Tuple[T, U]]) -> Tuple[List[T], List[U]]:
    a = []
    b = []
    for x, y in lst:
        a.append(x)
        b.append(y)
    return a, b


def async_to_blocking(coro: Awaitable[T]) -> T:
    loop = asyncio.get_event_loop()
    task = asyncio.ensure_future(coro)
    try:
        return loop.run_until_complete(task)
    finally:
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc:
                raise exc
        else:
            task.cancel()


def ait_to_blocking(ait: AsyncIterator[T]) -> Iterator[T]:
    while True:
        try:
            yield async_to_blocking(ait.__anext__())
        except StopAsyncIteration:
            break


async def blocking_to_async(thread_pool: concurrent.futures.Executor,
                            fun: Callable[..., T],
                            *args,
                            **kwargs) -> T:
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, lambda: fun(*args, **kwargs))


async def bounded_gather(*pfs: Callable[[], Awaitable[T]],
                         parallelism: int = 10,
                         return_exceptions: bool = False
                         ) -> List[T]:
    gatherer = AsyncThrottledGather[T](*pfs,
                                       parallelism=parallelism,
                                       return_exceptions=return_exceptions)
    return await gatherer.wait()


class AsyncThrottledGather(Generic[T]):
    def __init__(self,
                 *pfs: Callable[[], Awaitable[T]],
                 parallelism: int = 10,
                 return_exceptions: bool = False):
        self.count = len(pfs)
        self.n_finished = 0

        self._queue: asyncio.Queue[Tuple[int, Callable[[], Awaitable[T]]]] = asyncio.Queue()
        self._done = asyncio.Event()
        self._return_exceptions = return_exceptions

        self._results: List[Union[T, Exception, None]] = [None] * len(pfs)
        self._errors: List[BaseException] = []

        self._workers: List[asyncio.Task] = []
        for _ in range(parallelism):
            self._workers.append(asyncio.create_task(self._worker()))

        for i, pf in enumerate(pfs):
            self._queue.put_nowait((i, pf))

    def _cancel_workers(self):
        for worker in self._workers:
            try:
                if worker.done() and not worker.cancelled():
                    exc = worker.exception()
                    if exc:
                        raise exc
                else:
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
                res = err  # type: ignore
                if not self._return_exceptions:
                    self._errors.append(err)
                    self._done.set()
                    return

            self._results[i] = res
            self.n_finished += 1

            if self.n_finished == self.count:
                self._done.set()

    async def wait(self) -> List[T]:
        try:
            if self.count > 0:
                await self._done.wait()
        finally:
            self._cancel_workers()

        if self._errors:
            raise self._errors[0]

        return cast(List[T], self._results)


class AsyncWorkerPool:
    def __init__(self, parallelism, queue_size=1000):
        self._queue: asyncio.Queue[Tuple[Callable, Tuple[Any, ...], Mapping[str, Any]]] = asyncio.Queue(maxsize=queue_size)
        self.workers = {asyncio.ensure_future(self._worker()) for _ in range(parallelism)}

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
        for _, t in self._pending.items():
            if t.done() and not t.cancelled():
                exc = t.exception()
                if exc:
                    raise exc
            else:
                t.cancel()
        self._pending = None

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
            retval = None
            try:
                async with self._sema:
                    retval = await f(*args, **kwargs)
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
                return retval
            del self._pending[id]
            if not self._pending:
                self._done_event.set()
            return retval

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


async def bounded_gather2_return_exceptions(
        sema: asyncio.Semaphore,
        *pfs: Callable[[], Awaitable[T]]
) -> List[Union[Tuple[T, None], Tuple[None, Optional[BaseException]]]]:
    '''Run the partial functions `pfs` as tasks with parallelism bounded
    by `sema`, which should be `asyncio.Semaphore` whose initial value
    is the desired level of parallelism.

    The return value is the list of partial function results as pairs:
    the pair `(value, None)` if the partial function returned value or
    `(None, exc)` if the partial function raised the exception `exc`.

    '''
    async def run_with_sema_return_exceptions(pf: Callable[[], Awaitable[T]]):
        try:
            async with sema:
                return (await pf(), None)
        except:
            _, exc, _ = sys.exc_info()
            return (None, exc)

    tasks = [asyncio.create_task(run_with_sema_return_exceptions(pf)) for pf in pfs]
    async with WithoutSemaphore(sema):
        return await asyncio.gather(*tasks)


async def bounded_gather2_raise_exceptions(
        sema: asyncio.Semaphore,
        *pfs: Callable[[], Awaitable[T]],
        cancel_on_error: bool = False
) -> List[T]:
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
    async def run_with_sema(pf: Callable[[], Awaitable[T]]):
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
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        raise exc
                else:
                    task.cancel()
            if tasks:
                async with WithoutSemaphore(sema):
                    await asyncio.wait(tasks)


async def bounded_gather2(
        sema: asyncio.Semaphore,
        *pfs: Callable[[], Awaitable[T]],
        return_exceptions: bool = False,
        cancel_on_error: bool = False
) -> List[T]:
    if return_exceptions:
        if cancel_on_error:
            raise ValueError('cannot request return_exceptions and cancel_on_error')
        return await bounded_gather2_return_exceptions(sema, *pfs)  # type: ignore
    return await bounded_gather2_raise_exceptions(sema, *pfs, cancel_on_error=cancel_on_error)


# nginx returns 502 if it cannot connect to the upstream server
#
# 408 request timeout
# 500 internal server error
# 502 bad gateway
# 503 service unavailable
# 504 gateway timeout
# 429 "Temporarily throttled, too many requests"
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
if os.environ.get('HAIL_DONT_RETRY_500') == '1':
    RETRYABLE_HTTP_STATUS_CODES.remove(500)

RETRYABLE_ERRNOS = {
    # these should match (where an equivalent exists) nettyRetryableErrorNumbers in
    # is/hail/services/package.scala
    errno.ETIMEDOUT,
    errno.ECONNREFUSED,
    errno.EHOSTUNREACH,
    errno.ECONNRESET,
    errno.ENETUNREACH,
    errno.EPIPE,
}


class TransientError(Exception):
    pass


RETRY_ONCE_BAD_REQUEST_ERROR_MESSAGES = {
    'User project specified in the request is invalid.',
    'Invalid grant: account not found',
}


def is_limited_retries_error(e: BaseException) -> bool:
    # An exception is a "retry once error" if a rare, known bug in a dependency or in a cloud
    # provider can manifest as this exception *and* that manifestation is indistinguishable from a
    # true error.
    import hailtop.httpx  # pylint: disable=import-outside-toplevel,cyclic-import
    if aiodocker is not None and isinstance(e, aiodocker.exceptions.DockerError):
        return (e.status == 404
                and 'azurecr.io' in e.message
                and 'not found: manifest unknown: ' in e.message)
    if isinstance(e, hailtop.httpx.ClientResponseError):
        return e.status == 400 and any(msg in e.body for msg in RETRY_ONCE_BAD_REQUEST_ERROR_MESSAGES)
    if isinstance(e, ConnectionResetError):
        return True
    if isinstance(e, ConnectionRefusedError):
        return True
    if e.__cause__ is not None:
        return is_limited_retries_error(e.__cause__)
    return False


def is_transient_error(e: BaseException) -> bool:
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
    #     message, payload = await self._protocol.read()  # type: ignore
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
    if (isinstance(e, aiohttp.ClientResponseError)
            and e.status in RETRYABLE_HTTP_STATUS_CODES):
        return True
    if (isinstance(e, hailtop.aiocloud.aiogoogle.client.compute_client.GCPOperationError)
            and e.error_codes is not None
            and 'QUOTA_EXCEEDED' in e.error_codes):
        return True
    if (isinstance(e, hailtop.httpx.ClientResponseError)
            and (e.status in RETRYABLE_HTTP_STATUS_CODES
                 or e.status == 403 and 'rateLimitExceeded' in e.body)):
        return True
    if isinstance(e, aiohttp.ServerTimeoutError):
        return True
    if isinstance(e, aiohttp.ServerDisconnectedError):
        return True
    if isinstance(e, asyncio.TimeoutError):
        return True
    if (isinstance(e, aiohttp.ClientConnectorError)
            and is_transient_error(e.os_error)):
        return True
    # appears to happen when the connection is lost prematurely, see:
    # https://github.com/aio-libs/aiohttp/issues/4581
    # https://github.com/aio-libs/aiohttp/blob/v3.7.4/aiohttp/client_proto.py#L85
    if (isinstance(e, aiohttp.ClientPayloadError)
            and e.args[0] == "Response payload is not completed"):
        return True
    if (isinstance(e, aiohttp.ClientOSError)
            and len(e.args) >= 2
            and 'sslv3 alert bad record mac' in e.args[1]):
        # aiohttp.client_exceptions.ClientOSError: [Errno 1] [SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:2548)
        #
        # This appears to be a symptom of Google rate-limiting as of 2023-10-15
        return True
    if isinstance(e, OSError) and e.errno in RETRYABLE_ERRNOS:
        return True
    if isinstance(e, urllib3.exceptions.ReadTimeoutError):
        return True
    if isinstance(e, requests.exceptions.ReadTimeout):
        return True
    if isinstance(e, requests.exceptions.ConnectionError):
        return True
    if isinstance(e, socket.timeout):
        return True
    if isinstance(e, socket.gaierror) and e.errno in (socket.EAI_AGAIN, socket.EAI_NONAME):
        # socket.EAI_AGAIN: [Errno -3] Temporary failure in name resolution
        # socket.EAI_NONAME: [Errno 8] nodename nor servname provided, or not known
        return True
    if isinstance(e, botocore.exceptions.ConnectionClosedError):
        return True
    if aiodocker is not None and isinstance(e, aiodocker.exceptions.DockerError):
        if e.status == 500 and 'Invalid repository name' in e.message:
            return False
        if e.status == 500 and 'Permission "artifactregistry.repositories.downloadArtifacts" denied on resource' in e.message:
            return False
        if e.status == 500 and 'denied: retrieving permissions failed' in e.message:
            return False
        # DockerError(500, "Head https://gcr.io/v2/genomics-tools/samtools/manifests/latest: unknown: Project 'project:genomics-tools' not found or deleted.")
        # DockerError(500, 'unknown: Tag v1.11.2 was deleted or has expired. To pull, revive via time machine')
        if e.status == 500 and 'unknown' in e.message:
            return False
        return e.status in RETRYABLE_HTTP_STATUS_CODES
    if isinstance(e, TransientError):
        return True
    if e.__cause__ is not None:
        return is_transient_error(e.__cause__)
    return False


def is_delayed_warning_error(e: BaseException) -> bool:
    if isinstance(e, aiohttp.ClientResponseError) and e.status in (503, 429):
        # 503 service unavailable
        # 429 "Temporarily throttled, too many requests"
        return True
    return False


LOG_2_MAX_MULTIPLIER = 30  # do not set larger than 30 to avoid BigInt arithmetic
DEFAULT_MAX_DELAY_MS = 60_000
DEFAULT_BASE_DELAY_MS = 1_000


def delay_ms_for_try(
    tries: int,
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS,
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS
) -> int:
    # Based on AWS' recommendations:
    # - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    # - https://github.com/aws/aws-sdk-java/blob/master/aws-java-sdk-core/src/main/java/com/amazonaws/retry/PredefinedBackoffStrategies.java
    multiplier = 1 << min(tries, LOG_2_MAX_MULTIPLIER)
    ceiling_for_delay_ms = base_delay_ms * multiplier
    proposed_delay_ms = ceiling_for_delay_ms // 2 + random.randrange(ceiling_for_delay_ms // 2 + 1)
    return min(proposed_delay_ms, max_delay_ms)


async def sleep_before_try(
    tries: int,
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS,
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS
):
    await asyncio.sleep(delay_ms_for_try(tries, base_delay_ms, max_delay_ms) / 1000.0)


def sync_sleep_before_try(
    tries: int,
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS,
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS
):
    time.sleep(delay_ms_for_try(tries, base_delay_ms, max_delay_ms) / 1000.0)


def retry_all_errors(msg: Optional[str] = None, error_logging_interval: int = 10):
    async def _wrapper(f: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        tries = 0
        while True:
            try:
                return await f(*args, **kwargs)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except KeyboardInterrupt:
                raise
            except Exception:
                tries += 1
                if msg and tries % error_logging_interval == 0:
                    log.exception(msg, stack_info=True)
            await sleep_before_try(tries)
    return _wrapper


def retry_all_errors_n_times(max_errors: int = 10, msg: Optional[str] = None, error_logging_interval: int = 10):
    async def _wrapper(f: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        tries = 0
        while True:
            try:
                return await f(*args, **kwargs)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except KeyboardInterrupt:
                raise
            except Exception:
                tries += 1
                if msg and tries % error_logging_interval == 0:
                    log.exception(msg, stack_info=True)
                if tries >= max_errors:
                    raise
            await sleep_before_try(tries)
    return _wrapper


async def retry_transient_errors(f: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
    return await retry_transient_errors_with_debug_string('', 0, f, *args, **kwargs)


async def retry_transient_errors_with_delayed_warnings(warning_delay_msecs: int, f: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
    return await retry_transient_errors_with_debug_string('', warning_delay_msecs, f, *args, **kwargs)


async def retry_transient_errors_with_debug_string(debug_string: str, warning_delay_msecs: int, f: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
    start_time = time_msecs()
    tries = 0
    while True:
        try:
            return await f(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            tries += 1
            delay = delay_ms_for_try(tries) / 1000.0
            if tries <= 5 and is_limited_retries_error(e):
                log.warning(
                    f'A limited retry error has occured. We will automatically retry '
                    f'{5 - tries} more times. Do not be alarmed. (next delay: '
                    f'{delay}s). The most recent error was {type(e)} {e}. {debug_string}'
                )
            elif not is_transient_error(e):
                raise
            else:
                log_warnings = (time_msecs() - start_time >= warning_delay_msecs) or not is_delayed_warning_error(e)
                if log_warnings and tries == 2:
                    log.warning(f'A transient error occured. We will automatically retry. Do not be alarmed. '
                                f'We have thus far seen {tries} transient errors (next delay: '
                                f'{delay}s). The most recent error was {type(e)} {e}. {debug_string}')
                elif log_warnings and tries % 10 == 0:
                    st = ''.join(traceback.format_stack())
                    log.warning(f'A transient error occured. We will automatically retry. '
                                f'We have thus far seen {tries} transient errors (next delay: '
                                f'{delay}s). The stack trace for this call is {st}. The most recent error was {type(e)} {e}. {debug_string}', exc_info=True)
        await asyncio.sleep(delay)


def sync_retry_transient_errors(f: Callable[..., T], *args, **kwargs) -> T:
    tries = 0
    while True:
        try:
            return f(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            tries += 1
            if tries % 10 == 0:
                st = ''.join(traceback.format_stack())
                log.warning(f'Encountered {tries} errors. My stack trace is {st}. Most recent error was {e}', exc_info=True)
            if is_transient_error(e):
                pass
            else:
                raise
        sync_sleep_before_try(tries)


def retry_response_returning_functions(fun, *args, **kwargs):
    tries = 0
    response = sync_retry_transient_errors(
        fun, *args, **kwargs)
    while response.status_code in RETRYABLE_HTTP_STATUS_CODES:
        tries += 1
        if tries % 10 == 0:
            log.warning(f'encountered {tries} bad status codes, most recent '
                        f'one was {response.status_code}')
        response = sync_retry_transient_errors(
            fun, *args, **kwargs)
        sync_sleep_before_try(tries)
    return response


def external_requests_client_session(headers: Optional[Dict[str, Any]] = None, timeout: int = 5) -> requests.Session:
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


async def collect_aiter(aiter: AsyncIterator[T]) -> List[T]:
    return [x async for x in aiter]


def dump_all_stacktraces():
    for t in asyncio.all_tasks():
        log.debug(t)
        t.print_stack()


async def retry_long_running(name: str, f: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
    delay_secs = 0.1
    while True:
        start_time = time_msecs()
        try:
            return await f(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
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


async def run_if_changed(changed: asyncio.Event, f: Callable[..., Awaitable[bool]], *args, **kwargs):
    while True:
        changed.clear()
        should_wait = await f(*args, **kwargs)
        # 0.5 is arbitrary, but should be short enough not to greatly
        # increase latency and long enough to reduce the impact of
        # wasteful spinning when `should_wait` is always true and the
        # event is constantly being set. This was instated to
        # avoid wasteful repetition of scheduling loops, but
        # might not always be desirable, especially in very low-latency batches.
        await asyncio.sleep(0.5)
        if should_wait:
            await changed.wait()


async def run_if_changed_idempotent(changed: asyncio.Event, f: Callable[..., Awaitable[bool]], *args, **kwargs):
    while True:
        should_wait = await f(*args, **kwargs)
        changed.clear()
        if should_wait:
            await changed.wait()


async def periodically_call(period: Union[int, float], f: Callable[..., Awaitable[Any]], *args, **kwargs):
    async def loop():
        log.info(f'starting loop for {f.__name__}')
        while True:
            await f(*args, **kwargs)
            await asyncio.sleep(period)
    await retry_long_running(f.__name__, loop)


async def periodically_call_with_dynamic_sleep(period: Callable[[], Union[int, float]], f, *args, **kwargs):
    async def loop():
        log.info(f'starting loop for {f.__name__}')
        while True:
            await f(*args, **kwargs)
            await asyncio.sleep(period())
    await retry_long_running(f.__name__, loop)


class LoggingTimerStep:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.start_time: Optional[int] = None

    async def __aenter__(self):
        self.start_time = time_msecs()

    async def __aexit__(self, exc_type, exc, tb):
        finish_time = time_msecs()
        assert self.start_time is not None
        self.timer.timing[self.name] = finish_time - self.start_time


class LoggingTimer:
    def __init__(self, description, threshold_ms=None):
        self.description = description
        self.threshold_ms = threshold_ms
        self.timing = {}
        self.start_time: Optional[int] = None

    def step(self, name):
        return LoggingTimerStep(self, name)

    async def __aenter__(self):
        self.start_time = time_msecs()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        finish_time = time_msecs()
        assert self.start_time is not None
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


def find_spark_home() -> str:
    spark_home = os.environ.get('SPARK_HOME')
    if spark_home is None:
        find_spark_home = subprocess.run('find_spark_home.py',
                                         capture_output=True,
                                         check=False)
        if find_spark_home.returncode != 0:
            raise ValueError(f'''SPARK_HOME is not set and find_spark_home.py returned non-zero exit code:
STDOUT:
{find_spark_home.stdout!r}
STDERR:
{find_spark_home.stderr!r}''')
        spark_home = find_spark_home.stdout.decode().strip()
    return spark_home


class Timings:
    def __init__(self):
        self.timings: Dict[str, Dict[str, int]] = {}

    @contextlib.contextmanager
    def step(self, name: str):
        assert name not in self.timings
        d: Dict[str, int] = {}
        self.timings[name] = d
        d['start_time'] = time_msecs()
        yield
        d['finish_time'] = time_msecs()
        d['duration'] = d['finish_time'] - d['start_time']

    def to_dict(self):
        return self.timings


def am_i_interactive() -> bool:
    """Determine if the current Python session is interactive.

    This should return True in IPython, a Python interpreter, and a Jupyter Notebook.

    """
    # https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
    return bool(getattr(sys, 'ps1', sys.flags.interactive))
