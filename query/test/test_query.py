import pytest
import hail as hl
from hailtop.hailctl.dev.query import cli

def test_simple_table():
    t = hl.utils.range_table(50, 3)
    t = t.filter((t.idx % 3 == 0) | ((t.idx / 7) % 3 == 0))
    n = t.count()
    print(f'n {n}')
    assert n == 17

# FIXME(danking): disabled while I work on a fix
# def test_simple_shuffle():
#     expected = [hl.Struct(idx=i) for i in range(99, -1, -1)]
#     t = hl.utils.range_table(100)
#     actual = t.order_by(-t.idx).collect()
#     assert actual == expected
