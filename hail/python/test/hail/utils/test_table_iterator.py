import hail as hl

from ..helpers import resource


def test_table_iterator_list_same_as_collecting_whole_table():
    actual = list(hl.utils.table_iterator(resource('backward_compatability/1.1.0/table/0.ht')))
    expected = hl.read_table(resource('backward_compatability/1.1.0/table/0.ht')).collect()
    assert actual == expected
