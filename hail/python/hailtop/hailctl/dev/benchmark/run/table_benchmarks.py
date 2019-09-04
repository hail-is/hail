from os import path
from tempfile import TemporaryDirectory
import hail as hl

from .utils import benchmark, resource


@benchmark
def table_key_by_shuffle():
    n = 1_000_000
    ht = hl.utils.range_table(n)
    ht = ht.key_by(x=n - ht.idx)
    ht._force_count()


@benchmark
def table_group_by_aggregate_sorted():
    n = 10_000_000
    ht = hl.utils.range_table(n)
    ht = ht.group_by(x=ht.idx // 1000).aggregate(y=hl.agg.count())
    ht._force_count()


@benchmark
def table_group_by_aggregate_unsorted():
    n = 10_000_000
    ht = hl.utils.range_table(n)
    ht = ht.group_by(x=(n - ht.idx) // 1000).aggregate(y=hl.agg.count())
    ht._force_count()


@benchmark
def table_range_force_count():
    hl.utils.range_table(100_000_000)._force_count()


@benchmark
def table_python_construction():
    n = 100
    ht = hl.utils.range_table(100)
    for i in range(n):
        ht = ht.annotate(**{f'x_{i}': 0})


@benchmark
def table_big_aggregate_compilation():
    n = 1_000
    ht = hl.utils.range_table(1)
    expr = tuple([hl.agg.fraction(ht.idx % i == 0) for i in range(n) if i > 0])
    ht.aggregate(expr)


@benchmark
def table_big_aggregate_compile_and_execute():
    n = 1_000
    m = 1_000_000
    ht = hl.utils.range_table(m)
    expr = tuple([hl.agg.fraction(ht.idx % i == 0) for i in range(n) if i > 0])
    ht.aggregate(expr)


def table_foreign_key_join(n1: int, n2: int):
    ht = hl.utils.range_table(n1)
    ht2 = hl.utils.range_table(n2)
    ht.annotate(x=ht2[(n1 - 1 - ht.idx) % n2])._force_count()


@benchmark
def table_foreign_key_join_same_cardinality():
    table_foreign_key_join(1_000_000, 1_000_000)


@benchmark
def table_foreign_key_join_left_higher_cardinality():
    table_foreign_key_join(1_000_000, 1_000)


@benchmark
def table_aggregate_array_sum():
    n = 10_000_000
    m = 100
    ht = hl.utils.range_table(n)
    ht.aggregate(hl.agg.array_sum(hl.range(0, m)))


@benchmark
def table_annotate_many_flat():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n)
    ht = ht.annotate(**{f'x{i}': i + ht.idx for i in range(m)})
    ht._force_count()


@benchmark
def table_annotate_many_nested_no_dependence():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n)
    for i in range(m):
        ht = ht.annotate(**{f'x{i}': i + ht.idx})
    ht._force_count()


@benchmark
def table_annotate_many_nested_dependence():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n).annotate(x0=1)
    for i in range(1, m):
        ht = ht.annotate(**{f'x{i}': i + ht[f'x{i - 1}']})
    ht._force_count()


@benchmark
def table_read_force_count_ints():
    ht = hl.read_table(resource('many_ints_table.ht'))
    ht._force_count()


@benchmark
def table_read_force_count_strings():
    ht = hl.read_table(resource('many_strings_table.ht'))
    ht._force_count()


@benchmark
def table_import_ints():
    hl.import_table(resource('many_ints_table.tsv.bgz'),
                    types={'idx': 'int',
                           **{f'i{i}': 'int' for i in range(5)},
                           **{f'array{i}': 'array<int>' for i in range(2)}}
                    )._force_count()


@benchmark
def table_import_strings():
    hl.import_table(resource('many_strings_table.tsv.bgz'))._force_count()


@benchmark
def table_aggregate_int_stats():
    ht = hl.read_table(resource('many_ints_table.ht'))
    ht.aggregate(tuple([*(hl.agg.stats(ht[f'i{i}']) for i in range(5)),
                        *(hl.agg.stats(hl.sum(ht[f'array{i}'])) for i in range(2)),
                        *(hl.agg.explode(lambda elt: hl.agg.stats(elt), ht[f'array{i}']) for i in range(2))]))


@benchmark
def table_aggregate_counter():
    ht = hl.read_table(resource('many_strings_table.ht'))
    ht.aggregate(hl.tuple([hl.agg.counter(ht[f'f{i}']) for i in range(8)]))


@benchmark
def table_take():
    ht = hl.read_table(resource('many_strings_table.ht'))
    ht.take(100)


@benchmark
def table_show():
    ht = hl.read_table(resource('many_strings_table.ht'))
    ht.show(100)


@benchmark
def table_expr_take():
    ht = hl.read_table(resource('many_strings_table.ht'))
    hl.tuple([ht.f1, ht.f2]).take(100)


@benchmark
def read_force_count_p1000():
    hl.read_table(resource('table_10M_par_1000.ht'))._force_count()


@benchmark
def read_force_count_p100():
    hl.read_table(resource('table_10M_par_100.ht'))._force_count()


@benchmark
def read_force_count_p10():
    hl.read_table(resource('table_10M_par_10.ht'))._force_count()


@benchmark
def write_range_table_p1000():
    with TemporaryDirectory() as tmpdir:
        ht = hl.utils.range_table(10_000_000, 1000)
        ht.write(path.join(tmpdir, 'tmp.ht'))


@benchmark
def write_range_table_p100():
    with TemporaryDirectory() as tmpdir:
        ht = hl.utils.range_table(10_000_000, 100)
        ht.write(path.join(tmpdir, 'tmp.ht'))


@benchmark
def write_range_table_p10():
    with TemporaryDirectory() as tmpdir:
        ht = hl.utils.range_table(10_000_000, 10)
        ht.write(path.join(tmpdir, 'tmp.ht'))


@benchmark
def read_with_index_p1000():
    rows = 10_000_000
    bins = 1_000
    width = rows // bins
    intervals = [hl.Interval(start=i, end=i + width) for i in range(0, rows, width)]
    ht = hl.read_table(resource('table_10M_par_10.ht'), _intervals=intervals)
    ht._force_count()


@benchmark
def union_p100_p100():
    ht1 = hl.read_table(resource('table_10M_par_100.ht'))
    ht2 = hl.read_table(resource('table_10M_par_100.ht'))
    ht1.union(ht2)._force_count()


@benchmark
def union_p1000_p1000():
    ht1 = hl.read_table(resource('table_10M_par_1000.ht'))
    ht2 = hl.read_table(resource('table_10M_par_1000.ht'))
    ht1.union(ht2)._force_count()


@benchmark
def union_p10_p1000():
    ht1 = hl.read_table(resource('table_10M_par_10.ht'))
    ht2 = hl.read_table(resource('table_10M_par_1000.ht'))
    ht1.union(ht2)._force_count()


@benchmark
def union_p1000_p10():
    ht1 = hl.read_table(resource('table_10M_par_1000.ht'))
    ht2 = hl.read_table(resource('table_10M_par_10.ht'))
    ht1.union(ht2)._force_count()


@benchmark
def union_p10_p100():
    ht1 = hl.read_table(resource('table_10M_par_10.ht'))
    ht2 = hl.read_table(resource('table_10M_par_100.ht'))
    ht1.union(ht2)._force_count()


@benchmark
def join_p100_p100():
    ht1 = hl.read_table(resource('table_10M_par_100.ht'))
    ht2 = hl.read_table(resource('table_10M_par_100.ht'))
    ht1.join(ht2)._force_count()


@benchmark
def join_p1000_p1000():
    ht1 = hl.read_table(resource('table_10M_par_1000.ht'))
    ht2 = hl.read_table(resource('table_10M_par_1000.ht'))
    ht1.join(ht2)._force_count()


@benchmark
def join_p10_p1000():
    ht1 = hl.read_table(resource('table_10M_par_10.ht'))
    ht2 = hl.read_table(resource('table_10M_par_1000.ht'))
    ht1.join(ht2)._force_count()


@benchmark
def join_p1000_p10():
    ht1 = hl.read_table(resource('table_10M_par_1000.ht'))
    ht2 = hl.read_table(resource('table_10M_par_10.ht'))
    ht1.join(ht2)._force_count()


@benchmark
def join_p10_p100():
    ht1 = hl.read_table(resource('table_10M_par_10.ht'))
    ht2 = hl.read_table(resource('table_10M_par_100.ht'))
    ht1.join(ht2)._force_count()


@benchmark
def join_p100_p10():
    ht1 = hl.read_table(resource('table_10M_par_100.ht'))
    ht2 = hl.read_table(resource('table_10M_par_10.ht'))
    ht1.join(ht2)._force_count()
