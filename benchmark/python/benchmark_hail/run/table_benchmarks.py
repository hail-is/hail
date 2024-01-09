from os import path
from tempfile import TemporaryDirectory

import hail as hl

from .resources import *
from .utils import benchmark


@benchmark()
def table_key_by_shuffle():
    n = 1_000_000
    ht = hl.utils.range_table(n)
    ht = ht.key_by(x=n - ht.idx)
    ht._force_count()


@benchmark()
def table_group_by_aggregate_sorted():
    n = 10_000_000
    ht = hl.utils.range_table(n)
    ht = ht.group_by(x=ht.idx // 1000).aggregate(y=hl.agg.count())
    ht._force_count()


@benchmark()
def table_group_by_aggregate_unsorted():
    n = 10_000_000
    ht = hl.utils.range_table(n)
    ht = ht.group_by(x=(n - ht.idx) // 1000).aggregate(y=hl.agg.count())
    ht._force_count()


@benchmark()
def table_range_force_count():
    hl.utils.range_table(100_000_000)._force_count()


@benchmark()
def table_range_join_1b_1k():
    ht1 = hl.utils.range_table(1_000_000_000)
    ht2 = hl.utils.range_table(1_000)
    ht1.join(ht2, 'inner').count()


@benchmark()
def table_range_join_1b_1b():
    ht1 = hl.utils.range_table(1_000_000_000)
    ht2 = hl.utils.range_table(1_000_000_000)
    ht1.join(ht2, 'inner').count()


@benchmark()
def table_python_construction():
    n = 100
    ht = hl.utils.range_table(100)
    for i in range(n):
        ht = ht.annotate(**{f'x_{i}': 0})


@benchmark()
def table_big_aggregate_compilation():
    n = 1_000
    ht = hl.utils.range_table(1)
    expr = tuple([hl.agg.fraction(ht.idx % i == 0) for i in range(n) if i > 0])
    ht.aggregate(expr)


@benchmark()
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


@benchmark()
def table_foreign_key_join_same_cardinality():
    table_foreign_key_join(1_000_000, 1_000_000)


@benchmark()
def table_foreign_key_join_left_higher_cardinality():
    table_foreign_key_join(1_000_000, 1_000)


@benchmark()
def table_aggregate_array_sum():
    n = 10_000_000
    m = 100
    ht = hl.utils.range_table(n)
    ht.aggregate(hl.agg.array_sum(hl.range(0, m)))


@benchmark()
def table_annotate_many_flat():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n, 16)
    ht = ht.annotate(**{f'x{i}': i + ht.idx for i in range(m)})
    ht._force_count()


@benchmark()
def table_annotate_many_nested_no_dependence():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n, 16)
    for i in range(m):
        ht = ht.annotate(**{f'x{i}': i + ht.idx})
    ht._force_count()


@benchmark()
def table_annotate_many_nested_dependence_constants():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n, 16).annotate(x0=1)
    for i in range(1, m):
        ht = ht.annotate(**{f'x{i}': i + ht[f'x{i - 1}']})
    ht._force_count()


@benchmark()
def table_annotate_many_nested_dependence():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n, 16)
    ht = ht.annotate(x0=ht.idx)
    for i in range(1, m):
        ht = ht.annotate(**{f'x{i}': i + ht[f'x{i - 1}']})
    ht._force_count()


@benchmark(args=many_ints_table.handle('ht'))
def table_read_force_count_ints(ht_path):
    ht = hl.read_table(ht_path)
    ht._force_count()


@benchmark(args=many_strings_table.handle('ht'))
def table_read_force_count_strings(ht_path):
    ht = hl.read_table(ht_path)
    ht._force_count()


@benchmark(args=many_ints_table.handle('tsv'))
def table_import_ints(tsv):
    hl.import_table(
        tsv, types={'idx': 'int', **{f'i{i}': 'int' for i in range(5)}, **{f'array{i}': 'array<int>' for i in range(2)}}
    )._force_count()


@benchmark(args=many_ints_table.handle('tsv'))
def table_import_ints_impute(tsv):
    hl.import_table(tsv, impute=True)._force_count()


@benchmark(args=many_strings_table.handle('tsv'))
def table_import_strings(tsv):
    hl.import_table(tsv)._force_count()


@benchmark(args=many_ints_table.handle('ht'))
def table_aggregate_int_stats(ht_path):
    ht = hl.read_table(ht_path)
    ht.aggregate(
        tuple([
            *(hl.agg.stats(ht[f'i{i}']) for i in range(5)),
            *(hl.agg.stats(hl.sum(ht[f'array{i}'])) for i in range(2)),
            *(hl.agg.explode(lambda elt: hl.agg.stats(elt), ht[f'array{i}']) for i in range(2)),
        ])
    )


@benchmark()
def table_range_means():
    ht = hl.utils.range_table(10_000_000, 16)
    ht = ht.annotate(m=hl.mean(hl.range(0, ht.idx % 1111)))
    ht._force_count()


@benchmark()
def table_range_array_range_force_count():
    ht = hl.utils.range_table(30).annotate(big_range=hl.range(100_000_000))
    ht._force_count()


@benchmark(args=random_doubles.handle('mt'))
def table_aggregate_approx_cdf(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt.aggregate_entries((
        hl.agg.approx_cdf(mt.x),
        hl.agg.approx_cdf(mt.x**2, k=500),
        hl.agg.approx_cdf(1 / mt.x, k=1000),
    ))


@benchmark(args=many_strings_table.handle('ht'))
def table_aggregate_counter(ht_path):
    ht = hl.read_table(ht_path)
    ht.aggregate(hl.tuple([hl.agg.counter(ht[f'f{i}']) for i in range(8)]))


@benchmark(args=many_strings_table.handle('ht'))
def table_aggregate_take_by_strings(ht_path):
    ht = hl.read_table(ht_path)
    ht.aggregate(hl.tuple([hl.agg.take(ht['f18'], 25, ordering=ht[f'f{i}']) for i in range(18)]))


@benchmark(args=many_ints_table.handle('ht'))
def table_aggregate_downsample_dense(ht_path):
    ht = hl.read_table(ht_path)
    ht.aggregate(tuple([hl.agg.downsample(ht[f'i{i}'], ht['i3'], label=hl.str(ht['i4'])) for i in range(3)]))


@benchmark()
def table_aggregate_downsample_worst_case():
    ht = hl.utils.range_table(250_000_000, 8)
    ht.aggregate(hl.agg.downsample(ht.idx, -ht.idx))


# @benchmark FIXME: this needs fixtures to accurately measure downsample (rather than randomness)
def table_aggregate_downsample_sparse():
    ht = hl.utils.range_table(250_000_000, 8)
    ht.aggregate(hl.agg.downsample(hl.rand_norm() ** 5, hl.rand_norm() ** 5))


@benchmark(args=many_ints_table.handle('ht'))
def table_aggregate_linreg(ht_path):
    ht = hl.read_table(ht_path)
    ht.aggregate(hl.agg.array_agg(lambda i: hl.agg.linreg(ht.i0 + i, [ht.i1, ht.i2, ht.i3, ht.i4]), hl.range(75)))


@benchmark(args=many_strings_table.handle('ht'))
def table_take(ht_path):
    ht = hl.read_table(ht_path)
    ht.take(100)


@benchmark(args=many_strings_table.handle('ht'))
def table_show(ht_path):
    ht = hl.read_table(ht_path)
    ht.show(100)


@benchmark(args=many_strings_table.handle('ht'))
def table_expr_take(ht_path):
    ht = hl.read_table(ht_path)
    hl.tuple([ht.f1, ht.f2]).take(100)


@benchmark(args=many_partitions_tables.handle(1000))
def read_force_count_p1000(path):
    hl.read_table(path)._force_count()


@benchmark(args=many_partitions_tables.handle(100))
def read_force_count_p100(path):
    hl.read_table(path)._force_count()


@benchmark(args=many_partitions_tables.handle(10))
def read_force_count_p10(path):
    hl.read_table(path)._force_count()


@benchmark()
def write_range_table_p1000():
    with TemporaryDirectory() as tmpdir:
        ht = hl.utils.range_table(10_000_000, 1000)
        ht.write(path.join(tmpdir, 'tmp.ht'))


@benchmark()
def write_range_table_p100():
    with TemporaryDirectory() as tmpdir:
        ht = hl.utils.range_table(10_000_000, 100)
        ht.write(path.join(tmpdir, 'tmp.ht'))


@benchmark()
def write_range_table_p10():
    with TemporaryDirectory() as tmpdir:
        ht = hl.utils.range_table(10_000_000, 10)
        ht.write(path.join(tmpdir, 'tmp.ht'))


@benchmark(args=many_partitions_tables.handle(10))
def read_with_index_p1000(path):
    rows = 10_000_000
    bins = 1_000
    width = rows // bins
    intervals = [hl.Interval(start=i, end=i + width) for i in range(0, rows, width)]
    ht = hl.read_table(path, _intervals=intervals)
    ht._force_count()


@benchmark(args=many_partitions_tables.handle(100))
def union_p100_p100(path_100):
    ht1 = hl.read_table(path_100)
    ht2 = hl.read_table(path_100)
    ht1.union(ht2)._force_count()


@benchmark(args=many_partitions_tables.handle(1000))
def union_p1000_p1000(path_1000):
    ht1 = hl.read_table(path_1000)
    ht2 = hl.read_table(path_1000)
    ht1.union(ht2)._force_count()


@benchmark(args=(many_partitions_tables.handle(10), many_partitions_tables.handle(1000)))
def union_p10_p1000(path_10, path_1000):
    ht1 = hl.read_table(path_10)
    ht2 = hl.read_table(path_1000)
    ht1.union(ht2)._force_count()


@benchmark(args=(many_partitions_tables.handle(10), many_partitions_tables.handle(1000)))
def union_p1000_p10(path_10, path_1000):
    ht1 = hl.read_table(path_10)
    ht2 = hl.read_table(path_1000)
    ht2.union(ht1)._force_count()


@benchmark(args=(many_partitions_tables.handle(10), many_partitions_tables.handle(100)))
def union_p10_p100(path_10, path_100):
    ht1 = hl.read_table(path_10)
    ht2 = hl.read_table(path_100)
    ht1.union(ht2)._force_count()


@benchmark(args=many_partitions_tables.handle(100))
def join_p100_p100(path_100):
    ht1 = hl.read_table(path_100)
    ht2 = hl.read_table(path_100)
    ht1.join(ht2)._force_count()


@benchmark(args=many_partitions_tables.handle(1000))
def join_p1000_p1000(path_1000):
    ht1 = hl.read_table(path_1000)
    ht2 = hl.read_table(path_1000)
    ht1.join(ht2)._force_count()


@benchmark(args=(many_partitions_tables.handle(10), many_partitions_tables.handle(1000)))
def join_p10_p1000(path_10, path_1000):
    ht1 = hl.read_table(path_10)
    ht2 = hl.read_table(path_1000)
    ht1.join(ht2)._force_count()


@benchmark(args=(many_partitions_tables.handle(10), many_partitions_tables.handle(1000)))
def join_p1000_p10(path_10, path_1000):
    ht1 = hl.read_table(path_10)
    ht2 = hl.read_table(path_1000)
    ht2.join(ht1)._force_count()


@benchmark(args=(many_partitions_tables.handle(10), many_partitions_tables.handle(100)))
def join_p10_p100(path_10, path_100):
    ht1 = hl.read_table(path_10)
    ht2 = hl.read_table(path_100)
    ht1.join(ht2)._force_count()


@benchmark(args=(many_partitions_tables.handle(10), many_partitions_tables.handle(100)))
def join_p100_p10(path_10, path_100):
    ht1 = hl.read_table(path_10)
    ht2 = hl.read_table(path_100)
    ht2.join(ht1)._force_count()


@benchmark(args=gnomad_dp_sim.handle())
def group_by_collect_per_row(path):
    ht = hl.read_matrix_table(path).localize_entries('e', 'c')
    ht.group_by(*ht.key).aggregate(value=hl.agg.collect(ht.row_value))._force_count()


@benchmark(args=gnomad_dp_sim.handle())
def group_by_take_rekey(path):
    ht = hl.read_matrix_table(path).localize_entries('e', 'c')
    ht.group_by(k=hl.int(ht.row_idx / 50)).aggregate(value=hl.agg.take(ht.row_value, 1))._force_count()


@benchmark()
def table_scan_sum_1k_partitions():
    ht = hl.utils.range_table(1000000, n_partitions=1000)
    ht = ht.annotate(x=hl.scan.sum(ht.idx))
    ht._force_count()


@benchmark()
def table_scan_prev_non_null():
    ht = hl.utils.range_table(100000000, n_partitions=10)
    ht = ht.annotate(x=hl.range(0, ht.idx % 25))
    ht = ht.annotate(y=hl.scan._prev_nonnull(ht.row))
    ht._force_count()


@benchmark()
def test_map_filter_region_memory():
    high_mem_table = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(100_000_000))
    high_mem_table = high_mem_table.filter(high_mem_table.idx % 2 == 0)
    assert high_mem_table._force_count() == 15


@benchmark()
def test_head_and_tail_region_memory():
    high_mem_table = hl.utils.range_table(100).annotate(big_array=hl.zeros(100_000_000))
    high_mem_table = high_mem_table.head(30)
    high_mem_table._force_count()


@benchmark()
def test_inner_join_region_memory():
    high_mem_table = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(50_000_000))
    high_mem_table2 = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(50_000_000))
    joined = high_mem_table.join(high_mem_table2)
    joined._force_count()


@benchmark()
def test_left_join_region_memory():
    high_mem_table = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(50_000_000))
    high_mem_table2 = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(50_000_000))
    joined = high_mem_table.join(high_mem_table2, how='left')
    joined._force_count()
