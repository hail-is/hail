import hail as hl
from hail.backend import ServiceBackend


def test_count_range():
    assert isinstance(hl.current_backend(), ServiceBackend)
    assert hl.utils.range_table(1000)._force_count() == 1000
