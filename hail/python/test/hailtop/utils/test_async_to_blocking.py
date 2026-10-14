import _thread
import asyncio
import contextvars
import os
import signal
import subprocess
import sys
import textwrap
import threading

import pytest

from hailtop.utils import async_to_blocking, run_on_hail_loop

pytestmark = pytest.mark.timeout(20)


def test_current_task_is_set_across_repeated_calls():
    async def needs_task():
        assert asyncio.current_task() is not None
        async with asyncio.timeout(5):
            return 'ok'

    # The first call used to take a different path from later ones, which broke on Python 3.14.
    for _ in range(3):
        assert async_to_blocking(needs_task()) == 'ok'


def test_can_be_called_from_inside_another_running_loop():
    async def answer():
        return 42

    async def caller():
        # e.g. a Jupyter cell calling Hail's sync API while the kernel's loop runs
        return async_to_blocking(answer())

    assert asyncio.run(caller()) == 42


def test_raises_what_the_work_raises():
    async def fails():
        raise ValueError('from the work')

    with pytest.raises(ValueError, match='from the work'):
        async_to_blocking(fails())


def test_accepts_awaitables_that_are_not_coroutines():
    class Answer:
        def __await__(self):
            return asyncio.sleep(0, 42).__await__()

    assert async_to_blocking(Answer()) == 42


def test_raises_instead_of_deadlocking_when_called_from_hail_async_code():
    async def inner():
        return 'unreachable'

    async def hail_async_code_calling_a_sync_wrapper():
        return async_to_blocking(inner())

    with pytest.raises(RuntimeError, match='Hail event loop'):
        async_to_blocking(hail_async_code_calling_a_sync_wrapper())


@pytest.fixture
def press_ctrl_c():
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    timers = []

    def press(after: float):
        timer = threading.Timer(after, _thread.interrupt_main)
        timers.append(timer)
        timer.start()

    try:
        yield press
    finally:
        for timer in timers:
            timer.cancel()
        signal.signal(signal.SIGINT, previous)


def test_ctrl_c_cancels_the_work_and_waits_for_it_to_clean_up(press_ctrl_c):
    cleaned_up = threading.Event()

    async def slow():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)
            cleaned_up.set()
            raise

    press_ctrl_c(after=0.2)
    with pytest.raises(KeyboardInterrupt):
        async_to_blocking(slow())
    assert cleaned_up.is_set()


def test_second_ctrl_c_stops_waiting_for_a_stuck_clean_up(press_ctrl_c):
    release = threading.Event()

    async def stuck_clean_up():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.to_thread(release.wait)
            raise

    press_ctrl_c(after=0.2)
    press_ctrl_c(after=0.6)
    try:
        with pytest.raises(KeyboardInterrupt):
            async_to_blocking(stuck_clean_up())
    finally:
        release.set()


_caller_var: contextvars.ContextVar[str] = contextvars.ContextVar('caller_var')


def test_work_sees_the_callers_context_variables():
    async def read():
        return _caller_var.get(None)

    token = _caller_var.set('set by caller')
    try:
        assert async_to_blocking(read()) == 'set by caller'
    finally:
        _caller_var.reset(token)


def test_works_in_a_forked_child():
    async def answer():
        return 42

    assert async_to_blocking(answer()) == 42  # the parent's loop thread now exists

    pid = os.fork()
    if pid == 0:
        signal.alarm(5)
        try:
            os._exit(0 if async_to_blocking(answer()) == 42 else 1)
        finally:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


async def _current_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def test_run_on_hail_loop_runs_on_the_loop_async_to_blocking_uses():
    # e.g. `await hl.init_batch()` in Jupyter must create its clients where later sync calls will use them
    hail_loop = await run_on_hail_loop(_current_loop())
    assert hail_loop is not asyncio.get_running_loop()
    assert hail_loop is async_to_blocking(_current_loop())


def test_run_on_hail_loop_awaits_directly_when_already_on_the_hail_loop():
    async def from_hail_async_code():
        return asyncio.get_running_loop(), await run_on_hail_loop(_current_loop())

    caller_loop, work_loop = async_to_blocking(from_hail_async_code())
    assert work_loop is caller_loop


def test_raises_instead_of_hanging_during_interpreter_shutdown():
    # The loop's daemon thread is stopped by then, e.g. when a __del__ runs during finalization.
    script = textwrap.dedent("""
        import asyncio
        from hailtop.utils import async_to_blocking

        class ClosesOnDel:
            def __del__(self):
                async_to_blocking(asyncio.sleep(0))

        async_to_blocking(asyncio.sleep(0))
        leaked = ClosesOnDel()
    """)
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert 'interpreter is shutting down' in result.stderr
