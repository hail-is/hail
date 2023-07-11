import hail as hl

from .helpers import run_in


@run_in('all')
def test_table_explode():
    hl.reset_global_randomness()
    ht = hl.utils.range_table(5)
    ht = ht.annotate(x = hl.range(hl.rand_int32(5)))
    ht = ht.explode('x')
    expected = [
        hl.Struct(idx=0, x=0),
        hl.Struct(idx=0, x=1),
        hl.Struct(idx=0, x=2),
        hl.Struct(idx=0, x=3),
        hl.Struct(idx=1, x=0),
        hl.Struct(idx=1, x=1),
        hl.Struct(idx=1, x=2),
        hl.Struct(idx=2, x=0),
        hl.Struct(idx=2, x=1),
        hl.Struct(idx=3, x=0),
        hl.Struct(idx=3, x=1),
        hl.Struct(idx=3, x=2),
        hl.Struct(idx=4, x=0),
        hl.Struct(idx=4, x=1),
        hl.Struct(idx=4, x=2)
    ]
    actual = ht.collect()
    assert expected == actual


@run_in('all')
def test_table_key_by():
    hl.reset_global_randomness()
    ht = hl.utils.range_table(5)
    ht = ht.annotate(x = hl.rand_int32(5))
    ht = ht.key_by('x')
    expected = [
        hl.Struct(idx=2, x=2),
        hl.Struct(idx=1, x=3),
        hl.Struct(idx=3, x=3),
        hl.Struct(idx=4, x=3),
        hl.Struct(idx=0, x=4)
    ]
    actual = ht.collect()
    assert expected == actual


@run_in('all')
def test_table_annotate():
    hl.reset_global_randomness()
    ht = hl.utils.range_table(5)
    ht = ht.annotate(x = hl.rand_int32(5))
    ht = ht.annotate(y = ht.x * 10)
    expected = [
        hl.Struct(idx=0, x=4, y=40),
        hl.Struct(idx=1, x=3, y=30),
        hl.Struct(idx=2, x=2, y=20),
        hl.Struct(idx=3, x=3, y=30),
        hl.Struct(idx=4, x=3, y=30),
    ]
    actual = ht.collect()
    assert expected == actual


@run_in('all')
def test_matrix_table_entries():
    hl.reset_global_randomness()
    mt = hl.utils.range_matrix_table(5, 2)
    mt = mt.annotate_entries(x = hl.rand_int32(5))
    expected = [
        hl.Struct(row_idx=0, col_idx=0, x=0),
        hl.Struct(row_idx=0, col_idx=1, x=3),
        hl.Struct(row_idx=1, col_idx=0, x=2),
        hl.Struct(row_idx=1, col_idx=1, x=4),
        hl.Struct(row_idx=2, col_idx=0, x=1),
        hl.Struct(row_idx=2, col_idx=1, x=4),
        hl.Struct(row_idx=3, col_idx=0, x=4),
        hl.Struct(row_idx=3, col_idx=1, x=2),
        hl.Struct(row_idx=4, col_idx=0, x=4),
        hl.Struct(row_idx=4, col_idx=1, x=4),
    ]
    actual = mt.entries().collect()
    assert expected == actual


@run_in('all')
def test_table_filter():
    hl.reset_global_randomness()
    ht = hl.utils.range_table(5)
    ht = ht.annotate(x = hl.rand_int32(5))
    ht = ht.filter(ht.x % 3 == 0)
    expected = [hl.Struct(idx=1, x=3), hl.Struct(idx=3, x=3), hl.Struct(idx=4, x=3)]
    actual = ht.collect()
    assert expected == actual


@run_in('all')
def test_table_key_by_aggregate():
    hl.reset_global_randomness()
    ht = hl.utils.range_table(5)
    ht = ht.annotate(x = hl.rand_int32(5))
    ht = ht.group_by(ht.x).aggregate(y=hl.agg.count())
    expected = [hl.Struct(x=2, y=1), hl.Struct(x=3, y=3), hl.Struct(x=4, y=1)]
    actual = ht.collect()
    assert expected == actual
