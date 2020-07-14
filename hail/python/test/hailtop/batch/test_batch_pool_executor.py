import asyncio
import time
from hailtop.batch import BatchPoolExecutor


def test_simple_map():
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        actual = list(bpe.map(lambda x: x * 3, range(4)))
    assert [0, 3, 6, 9] == actual


def test_simple_submit_result():
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        future_twenty_one = bpe.submit(lambda: 7 * 3)
    assert 21 == future_twenty_one.result()


def test_cancel_future():
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        future = bpe.submit(sleep_forever)
        was_cancelled = future.cancel()
    assert was_cancelled
    assert future.cancelled()


def test_cancel_future_after_shutdown_no_wait():
    bpe = BatchPoolExecutor(project='hail-vdc')
    future = bpe.submit(sleep_forever)
    bpe.shutdown(wait=False)
    was_cancelled = future.cancel()
    assert was_cancelled
    assert future.cancelled()


def test_cancel_future_after_exit_no_wait_on_exit():
    with BatchPoolExecutor(project='hail-vdc', wait_on_exit=False) as bpe:
        future = bpe.submit(sleep_forever)
    was_cancelled = future.cancel()
    assert was_cancelled
    assert future.cancelled()


def test_result_with_timeout():
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        future = bpe.submit(sleep_forever)
        try:
            future.result(timeout=2)
        except asyncio.TimeoutError:
            pass
        else:
            assert False
        finally:
            future.cancel()


def test_map_chunksize():
    row_args = [x
                for row in range(5)
                for x in [row, row, row, row, row]]
    col_args = [x
                for row in range(5)
                for x in list(range(5))]
    with BatchPoolExecutor(project='hail-vdc') as bpe:
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


def test_map_timeout():
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        try:
            list(bpe.map(lambda _: sleep_forever(), range(5), timeout=2))
        except asyncio.TimeoutError:
            pass
        else:
            assert False


def test_map_error_without_wait_no_error():
    with BatchPoolExecutor(project='hail-vdc', wait_on_exit=False) as bpe:
        bpe.map(lambda _: time.sleep(10), range(5), timeout=2)


def test_exception_in_map():
    def raise_value_error():
        raise ValueError('dead')
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        try:
            gen = bpe.map(lambda _: raise_value_error(), range(5))
            next(gen)
        except ValueError as exc:
            assert 'ValueError: dead' in exc.args[0]
        else:
            assert False


def test_exception_in_result():
    def raise_value_error():
        raise ValueError('dead')
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        try:
            future = bpe.submit(raise_value_error)
            future.result()
        except ValueError as exc:
            assert 'ValueError: dead' in exc.args[0]
        else:
            assert False


def test_exception_in_exception():
    def raise_value_error():
        raise ValueError('dead')
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        try:
            future = bpe.submit(raise_value_error)
            future.exception()
        except ValueError as exc:
            assert 'ValueError: dead' in exc.args[0]
        else:
            assert False


def test_no_exception_when_exiting_context():
    def raise_value_error():
        raise ValueError('dead')
    with BatchPoolExecutor(project='hail-vdc') as bpe:
        future = bpe.submit(raise_value_error)
    try:
        future.exception()
    except ValueError as exc:
        assert 'ValueError: dead' in exc.args[0]
    else:
        assert False


def sleep_forever():
    while True:
        time.sleep(3600)
