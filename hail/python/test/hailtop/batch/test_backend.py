import gc
import threading

import pytest

from hailtop.batch import LocalBackend
from hailtop.utils import async_to_blocking

pytestmark = pytest.mark.timeout(20)


class RecordsClose(LocalBackend):
    def __init__(self):
        super().__init__()
        self.closed = threading.Event()

    async def _async_close(self):
        await super()._async_close()
        self.closed.set()


@pytest.fixture
def no_automatic_gc():
    # So the cycle collector runs where the test runs it, not wherever an allocation happens to trigger it.
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def test_unclosed_backend_is_closed_when_its_last_reference_goes():
    b = RecordsClose()
    closed = b.closed
    del b
    assert closed.is_set()


def test_unclosed_backend_is_closed_when_its_last_reference_goes_on_the_hail_loop():
    b = RecordsClose()
    closed = b.closed
    holder = [b]
    del b

    async def drop_last_reference():
        holder.clear()

    async_to_blocking(drop_last_reference())
    assert closed.wait(5)


def test_unclosed_backend_is_closed_when_the_cycle_collector_runs_on_the_hail_loop(no_automatic_gc):
    b = RecordsClose()
    closed = b.closed
    b.cycle = b
    del b

    async def collect():
        gc.collect()

    async_to_blocking(collect())
    assert closed.wait(5)
