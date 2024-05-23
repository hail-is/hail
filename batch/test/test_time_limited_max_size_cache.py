import asyncio

import pytest

from gear.time_limited_max_size_cache import TimeLimitedMaxSizeCache

one_second_ns = 1000 * 1000 * 1000
one_day_ns = 24 * 60 * 60 * one_second_ns


async def test_simple():
    async def load_secret(k: int):
        return k**2

    c = TimeLimitedMaxSizeCache(load_secret, one_day_ns, 2, 'test_cache')
    assert await c.lookup(3) == 9
    assert await c.lookup(10) == 100
    assert await c.lookup(3) == 9
    assert await c.lookup(3) == 9
    assert await c.lookup(10) == 100
    assert await c.lookup(4) == 16


async def test_num_slots():
    load_counts = 0

    async def load_secret_only_twice(k: int):
        nonlocal load_counts
        if load_counts == 2:
            raise ValueError("already loaded secret twice")
        load_counts += 1
        return k**2

    c = TimeLimitedMaxSizeCache(load_secret_only_twice, one_day_ns, 2, 'test_cache')
    assert await c.lookup(3) == 9
    assert load_counts == 1
    assert await c.lookup(10) == 100
    assert load_counts == 2
    assert await c.lookup(3) == 9
    assert await c.lookup(3) == 9
    assert await c.lookup(10) == 100

    assert load_counts == 2
    with pytest.raises(ValueError, match='^already loaded secret twice$'):
        await c.lookup(4)
    assert load_counts == 2


async def test_lifetime():
    load_counts = 0

    async def load_secret(k: int):
        nonlocal load_counts
        load_counts += 1
        return k**2

    c = TimeLimitedMaxSizeCache(load_secret, one_second_ns, 3, 'test_cache')
    assert await c.lookup(3) == 9
    assert load_counts == 1

    assert await c.lookup(10) == 100
    assert load_counts == 2

    await asyncio.sleep(2)  # seconds
    assert load_counts == 2

    assert await c.lookup(3) == 9
    assert load_counts == 3

    assert await c.lookup(5) == 25
    assert load_counts == 4

    assert await c.lookup(10) == 100
    assert load_counts == 5

    # Python should execute these lookups before the one second timeout on keys 3 and 10, but this
    # code is unavoidably a data race. This test can fail under extraordinary conditions which cause
    # a stall in the Python interpreter.
    assert await c.lookup(3) == 9
    assert load_counts == 5
    assert await c.lookup(10) == 100
    assert load_counts == 5


async def test_num_slots_deletes_oldest():
    load_counts = 0

    async def load_secret(k: int):
        nonlocal load_counts
        load_counts += 1
        return k**2

    c = TimeLimitedMaxSizeCache(load_secret, one_day_ns, 3, 'test_cache')
    assert await c.lookup(0) == 0
    assert load_counts == 1

    assert await c.lookup(1) == 1
    assert load_counts == 2

    assert await c.lookup(1) == 1
    assert load_counts == 2

    assert await c.lookup(2) == 4
    assert load_counts == 3

    assert await c.lookup(3) == 9
    assert load_counts == 4

    assert await c.lookup(3) == 9
    assert load_counts == 4

    assert await c.lookup(2) == 4
    assert load_counts == 4

    assert await c.lookup(1) == 1
    assert load_counts == 4

    assert await c.lookup(0) == 0
    assert load_counts == 5


async def test_exception_propagates_everywhere():
    async def boom(_: int):
        await asyncio.sleep(2)  # seconds
        raise ValueError('boom')

    c = TimeLimitedMaxSizeCache(boom, one_day_ns, 3, 'test_cache')
    x = asyncio.create_task(c.lookup(0))  # this task will sleep then boom
    y = asyncio.create_task(c.lookup(0))
    z = asyncio.create_task(c.lookup(0))

    with pytest.raises(ValueError, match='^boom$'):
        await x

    with pytest.raises(ValueError, match='^boom$'):
        await y

    with pytest.raises(ValueError, match='^boom$'):
        await z
