import hail as hl


def test_count_range():
    assert hl.backend_type() == 'ServiceBackend'
    assert hl.utils.range_table(1000)._force_count() == 1000
