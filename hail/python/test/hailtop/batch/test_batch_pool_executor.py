import asyncio
import concurrent.futures
import time

import pytest

from hailtop.batch import BatchPoolExecutor, ServiceBackend
from hailtop.config import get_user_config
from hailtop.utils import sync_sleep_and_backoff
from hailtop.batch_client.client import BatchClient

PYTHON_DILL_IMAGE = 'hailgenetics/python-dill:3.7'


submitted_batch_ids = []


class RecordingServiceBackend(ServiceBackend):
    def _run(self, *args, **kwargs):
        b = super()._run(*args, **kwargs)
        submitted_batch_ids.append(b.id)
        return b


@pytest.fixture
def backend():
    return RecordingServiceBackend()


@pytest.fixture(scope='session', autouse=True)
def check_for_running_batches():
    yield
    billing_project = get_user_config().get('batch', 'billing_project', fallback=None)
    with BatchClient(billing_project=billing_project) as bc:
        for id in submitted_batch_ids:
            b = bc.get_batch(id)
            delay = 0.1
            while True:
                if b.status()['state'] != 'running':
                    break
                print(f'batch {b.id} is still running')
                delay = sync_sleep_and_backoff(delay)


def test_simple_map(backend):
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        actual = list(bpe.map(lambda x: x * 3, range(4)))
    assert [0, 3, 6, 9] == actual


def test_empty_map(backend):
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        actual = list(bpe.map(lambda x: x * 3, []))
    assert [] == actual


def test_simple_submit_result(backend):
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        future_twenty_one = bpe.submit(lambda: 7 * 3)
    assert 21 == future_twenty_one.result()


def test_cancel_future(backend):
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        def sleep_forever():
            while True:
                time.sleep(3600)

        future = bpe.submit(sleep_forever)
        was_cancelled = future.cancel()
    assert was_cancelled
    assert future.cancelled()


def test_cancel_future_after_shutdown_no_wait(backend):
    bpe = BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE)
    def sleep_forever():
        while True:
            time.sleep(3600)

    future = bpe.submit(sleep_forever)
    bpe.shutdown(wait=False)
    was_cancelled = future.cancel()
    assert was_cancelled
    assert future.cancelled()


def test_cancel_future_after_exit_no_wait_on_exit(backend):
    with BatchPoolExecutor(backend=backend, project='hail-vdc', wait_on_exit=False, image=PYTHON_DILL_IMAGE) as bpe:
        def sleep_forever():
            while True:
                time.sleep(3600)

        future = bpe.submit(sleep_forever)
    was_cancelled = future.cancel()
    assert was_cancelled
    assert future.cancelled()


def test_result_with_timeout(backend):
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        def sleep_forever():
            while True:
                time.sleep(3600)

        future = bpe.submit(sleep_forever)
        try:
            future.result(timeout=2)
        except asyncio.TimeoutError:
            pass
        else:
            assert False
        finally:
            future.cancel()


def test_map_chunksize(backend):
    row_args = [x
                for row in range(5)
                for x in [row, row, row, row, row]]
    col_args = [x
                for row in range(5)
                for x in list(range(5))]
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        multiplication_table = list(bpe.map(lambda x, y: x * y,
                                            row_args,
                                            col_args,
                                            chunksize=5))
    assert multiplication_table == [
        0,  0,  0,  0,  0,
        0,  1,  2,  3,  4,
        0,  2,  4,  6,  8,
        0,  3,  6,  9, 12,
        0,  4,  8, 12, 16]


def test_map_timeout(backend):
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        def sleep_forever():
            while True:
                time.sleep(3600)
        try:
            list(bpe.map(lambda _: sleep_forever(), range(5), timeout=2))
        except concurrent.futures.TimeoutError:
            pass
        else:
            assert False


def test_map_error_without_wait_no_error(backend):
    with BatchPoolExecutor(backend=backend, project='hail-vdc', wait_on_exit=False, image=PYTHON_DILL_IMAGE) as bpe:
        bpe.map(lambda _: time.sleep(10), range(5), timeout=2)


def test_exception_in_map(backend):
    def raise_value_error():
        raise ValueError('dead')
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        try:
            gen = bpe.map(lambda _: raise_value_error(), range(5))
            next(gen)
        except ValueError as exc:
            assert 'ValueError: dead' in exc.args[0]
        else:
            assert False


def test_exception_in_result(backend):
    def raise_value_error():
        raise ValueError('dead')
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        try:
            future = bpe.submit(raise_value_error)
            future.result()
        except ValueError as exc:
            assert 'ValueError: dead' in exc.args[0]
        else:
            assert False


def test_exception_in_exception(backend):
    def raise_value_error():
        raise ValueError('dead')
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        try:
            future = bpe.submit(raise_value_error)
            future.exception()
        except ValueError as exc:
            assert 'ValueError: dead' in exc.args[0]
        else:
            assert False


def test_no_exception_when_exiting_context(backend):
    def raise_value_error():
        raise ValueError('dead')
    with BatchPoolExecutor(backend=backend, project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        future = bpe.submit(raise_value_error)
    try:
        future.exception()
    except ValueError as exc:
        assert 'ValueError: dead' in exc.args[0]
    else:
        assert False


def test_bad_image_gives_good_error(backend):
    with BatchPoolExecutor(
            backend=backend,
            project='hail-vdc',
            image='hailgenetics/not-a-valid-image:123abc') as bpe:
        future = bpe.submit(lambda: 3)
    try:
        future.exception()
    except ValueError as exc:
        assert 'submitted job failed:' in exc.args[0]
    else:
        assert False


def test_call_result_after_timeout():
    with BatchPoolExecutor(project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        def sleep_forever():
            while True:
                time.sleep(3600)

        future = bpe.submit(sleep_forever)
        try:
            future.result(timeout=2)
        except asyncio.TimeoutError:
            try:
                future.result(timeout=2)
            except asyncio.TimeoutError:
                pass
            else:
                assert False
        else:
            assert False
        finally:
            future.cancel()


def test_basic_async_fun():
    with BatchPoolExecutor(project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        bpe.submit(asyncio.sleep, 1)


def test_async_fun_returns_value():
    async def foo(i, j):
        await asyncio.sleep(1)
        return i * j

    with BatchPoolExecutor(project='hail-vdc', image=PYTHON_DILL_IMAGE) as bpe:
        future = bpe.submit(foo, 2, 3)
        assert future.result() == 6
