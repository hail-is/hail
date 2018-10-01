import pytest

import hail as hl
from timeit import default_timer as timer

def assert_time(f, max_duration):
    start = timer()
    x = f()
    end = timer()
    assert (start - end) < max_duration
    print(start - end)
    return x

def test_large_number_of_fields(tmpdir):
    mt = hl.utils.range_table(100)
    mt = mt.annotate(**{
        str(k): k for k in range(1000)
    })
    f = tmpdir.join("foo.mt")
    assert_time(lambda: mt.count(), 5)
    assert_time(lambda: mt.write(str(f)), 5)
    mt = assert_time(lambda: hl.read_table(str(f)), 5)
    assert_time(lambda: mt.count(), 5)
