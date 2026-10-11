"""The one event loop that runs all of Hail's async work, on a dedicated thread.

Hail's sync API blocks on this loop rather than running the caller's own, so it works the same in
scripts and inside a running loop such as Jupyter's. It has to be a single loop for the life of the
process: clients and credentials are tied to the loop that created them.
"""

import asyncio
import inspect
import os
import sys
import threading
from typing import Any, Awaitable, Optional, TypeVar

T = TypeVar('T')

_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None


def _hail_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name='hail-event-loop', daemon=True)
            if hasattr(thread, '_ipykernel_parent_thread_ident'):
                # ipykernel sends a thread's output to the cell that created it. This thread outlives that
                # cell, so ask for the cell that is running instead. Deleting the attribute would stop
                # ipykernel from running the thread at all.
                thread._ipykernel_parent_thread_ident = None  # type: ignore[attr-defined]
            thread.start()
            _loop = loop
        return _loop


def _forget_loop_after_fork() -> None:
    # The loop's thread doesn't exist in a forked child, so the child starts its own.
    global _lock, _loop
    _lock = threading.Lock()
    _loop = None


if hasattr(os, 'register_at_fork'):  # not on Windows
    os.register_at_fork(after_in_child=_forget_loop_after_fork)


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def async_to_blocking(coro: Awaitable[T]) -> T:
    if sys.is_finalizing():
        # Daemon threads, the loop's included, are stopped by now, so the work would never run.
        _close(coro)
        raise RuntimeError('async_to_blocking cannot run while the interpreter is shutting down.')
    loop = _hail_loop()
    if _running_loop() is loop:
        _close(coro)
        raise RuntimeError(
            'async_to_blocking was called from a coroutine running on the Hail event loop, which would deadlock. '
            'Await the async version of this function instead.'
        )
    if not inspect.iscoroutine(coro):
        coro = _await(coro)

    started = threading.Event()
    finished = threading.Event()
    task: Optional[asyncio.Task[T]] = None

    def start() -> None:
        nonlocal task
        task = loop.create_task(coro)
        task.add_done_callback(lambda _: finished.set())
        started.set()

    # call_soon_threadsafe copies this thread's context, so the task sees the caller's context variables.
    loop.call_soon_threadsafe(start)
    try:
        _wait(finished)
    except KeyboardInterrupt:
        # Ctrl-C lands on this thread, not the loop's: cancel the task and let it clean up.
        _wait(started)
        assert task is not None
        loop.call_soon_threadsafe(task.cancel)
        _wait(finished)  # a second Ctrl-C stops waiting
        raise
    assert task is not None
    return task.result()


async def run_on_hail_loop(coro: Awaitable[T]) -> T:
    """Await `coro` on the Hail event loop, from whichever loop the caller is on.

    Async functions that create objects tied to the loop, such as clients, use this so that later
    `async_to_blocking` calls can use those objects.
    """
    loop = _hail_loop()
    if _running_loop() is loop:
        return await coro
    if not inspect.iscoroutine(coro):
        coro = _await(coro)
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))


def _close(awaitable: Awaitable[Any]) -> None:
    # Avoids a "coroutine was never awaited" warning.
    if inspect.iscoroutine(awaitable):
        awaitable.close()


async def _await(awaitable: Awaitable[T]) -> T:
    return await awaitable


def _wait(event: threading.Event) -> None:
    # A wait without a timeout can't be interrupted by Ctrl-C on Windows.
    while not event.wait(0.1):
        pass
