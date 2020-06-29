import time
from hailtop.batch import BatchPoolExecutor


def test_simple_map():
    with BatchPoolExecutor() as bpe:
        actual = list(bpe.map(lambda x: x * 3, range(4)))
    assert [0, 3, 6, 9] == actual


def test_simple_submit_result():
    with BatchPoolExecutor() as bpe:
        future_twenty_one = bpe.submit(lambda: 7 * 3)
    assert 21 == future_twenty_one.result()


def test_cancel_future():
    with BatchPoolExecutor() as bpe:
        future = bpe.submit(sleep_forever)
        was_cancelled = future.cancel()
    assert was_cancelled


def test_cancel_future_after_shutdown_no_wait():
    bpe = BatchPoolExecutor()
    future = bpe.submit(sleep_forever)
    bpe.shutdown(wait=False)
    was_cancelled = future.cancel()
    assert was_cancelled


def test_cancel_future_after_shutdown_no_wait():
    bpe = BatchPoolExecutor()
    future = bpe.submit(sleep_forever)
    bpe.shutdown(wait=False)
    was_cancelled = future.cancel()
    assert was_cancelled
    assert future.cancelled()


def test_wait_with_timeout():
    with BatchPoolExecutor() as bpe:
        future = bpe.submit(sleep_forever)
        future.wait
    was_cancelled = future.cancel()
    assert was_cancelled
    assert future.cancelled()


def sleep_forever():
    while True:
        time.sleep(3600)
