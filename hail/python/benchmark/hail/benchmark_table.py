import pytest

import hail as hl
from benchmark.hail.fixtures import many_partitions_ht
from benchmark.tools import benchmark


@benchmark()
def benchmark_table_key_by_shuffle():
    n = 1_000_000
    ht = hl.utils.range_table(n)
    ht = ht.key_by(x=n - ht.idx)
    ht._force_count()


@benchmark()
def benchmark_table_group_by_aggregate_sorted():
    n = 10_000_000
    ht = hl.utils.range_table(n)
    ht = ht.group_by(x=ht.idx // 1000).aggregate(y=hl.agg.count())
    ht._force_count()


@benchmark()
def benchmark_table_group_by_aggregate_unsorted():
    n = 10_000_000
    ht = hl.utils.range_table(n)
    ht = ht.group_by(x=(n - ht.idx) // 1000).aggregate(y=hl.agg.count())
    ht._force_count()


@benchmark()
def benchmark_table_range_force_count():
    hl.utils.range_table(100_000_000)._force_count()


@benchmark()
def benchmark_table_range_join_1b_1k():
    ht1 = hl.utils.range_table(1_000_000_000)
    ht2 = hl.utils.range_table(1_000)
    ht1.join(ht2, 'inner').count()


@benchmark()
def benchmark_table_range_join_1b_1b():
    ht1 = hl.utils.range_table(1_000_000_000)
    ht2 = hl.utils.range_table(1_000_000_000)
    ht1.join(ht2, 'inner').count()


@benchmark()
def benchmark_table_python_construction():
    n = 100
    ht = hl.utils.range_table(100)
    for i in range(n):
        ht = ht.annotate(**{f'x_{i}': 0})


@benchmark()
def benchmark_table_big_aggregate_compilation():
    n = 1_000
    ht = hl.utils.range_table(1)
    expr = tuple([hl.agg.fraction(ht.idx % i == 0) for i in range(n) if i > 0])
    ht.aggregate(expr)


@benchmark()
def benchmark_table_big_aggregate_compile_and_execute():
    n = 1_000
    m = 1_000_000
    ht = hl.utils.range_table(m)
    expr = tuple([hl.agg.fraction(ht.idx % i == 0) for i in range(n) if i > 0])
    ht.aggregate(expr)


@benchmark()
@pytest.mark.parametrize('m, n', [(1_000_000, 1_000_000), (1_000_000, 1_000)])
def benchmark_table_foreign_key_join(m, n):
    ht = hl.utils.range_table(m)
    ht2 = hl.utils.range_table(n)
    ht.annotate(x=ht2[(m - 1 - ht.idx) % n])._force_count()


@benchmark()
def benchmark_table_aggregate_array_sum():
    n = 10_000_000
    m = 100
    ht = hl.utils.range_table(n)
    ht.aggregate(hl.agg.array_sum(hl.range(0, m)))


@benchmark()
def benchmark_table_annotate_many_flat():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n, 16)
    ht = ht.annotate(**{f'x{i}': i + ht.idx for i in range(m)})
    ht._force_count()


@benchmark()
def benchmark_table_annotate_many_nested_no_dependence():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n, 16)
    for i in range(m):
        ht = ht.annotate(**{f'x{i}': i + ht.idx})
    ht._force_count()


@benchmark()
def benchmark_table_annotate_many_nested_dependence_constants():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n, 16).annotate(x0=1)
    for i in range(1, m):
        ht = ht.annotate(**{f'x{i}': i + ht[f'x{i - 1}']})
    ht._force_count()


@benchmark()
def benchmark_table_annotate_many_nested_dependence():
    n = 1_000_000
    m = 100
    ht = hl.utils.range_table(n, 16)
    ht = ht.annotate(x0=ht.idx)
    for i in range(1, m):
        ht = ht.annotate(**{f'x{i}': i + ht[f'x{i - 1}']})
    ht._force_count()


@benchmark()
def benchmark_table_read_force_count_ints(many_ints_ht):
    ht = hl.read_table(str(many_ints_ht))
    ht._force_count()


@benchmark()
def benchmark_table_read_force_count_strings(many_strings_ht):
    ht = hl.read_table(str(many_strings_ht))
    ht._force_count()


@benchmark()
def benchmark_table_import_ints(many_ints_tsv):
    hl.import_table(
        str(many_ints_tsv),
        types={'idx': 'int', **{f'i{i}': 'int' for i in range(5)}, **{f'array{i}': 'array<int>' for i in range(2)}},
    )._force_count()


@benchmark()
def benchmark_table_import_ints_impute(many_ints_tsv):
    hl.import_table(str(many_ints_tsv), impute=True)._force_count()


@benchmark()
def benchmark_table_import_strings(many_strings_tsv):
    hl.import_table(str(many_strings_tsv))._force_count()


@benchmark()
def benchmark_table_aggregate_int_stats(many_ints_ht):
    ht = hl.read_table(str(many_ints_ht))
    ht.aggregate(
        tuple([
            *(hl.agg.stats(ht[f'i{i}']) for i in range(5)),
            *(hl.agg.stats(hl.sum(ht[f'array{i}'])) for i in range(2)),
            *(hl.agg.explode(lambda elt: hl.agg.stats(elt), ht[f'array{i}']) for i in range(2)),
        ])
    )


@benchmark()
def benchmark_table_range_means():
    ht = hl.utils.range_table(10_000_000, 16)
    ht = ht.annotate(m=hl.mean(hl.range(0, ht.idx % 1111)))
    ht._force_count()


@benchmark()
def benchmark_table_range_array_range_force_count():
    ht = hl.utils.range_table(30).annotate(big_range=hl.range(100_000_000))
    ht._force_count()


@benchmark()
def benchmark_table_aggregate_approx_cdf(random_doubles_mt):
    mt = hl.read_matrix_table(str(random_doubles_mt))
    mt.aggregate_entries((
        hl.agg.approx_cdf(mt.x),
        hl.agg.approx_cdf(mt.x**2, k=500),
        hl.agg.approx_cdf(1 / mt.x, k=1000),
    ))


@benchmark()
def benchmark_table_aggregate_counter(many_strings_ht):
    ht = hl.read_table(str(many_strings_ht))
    ht.aggregate(hl.tuple([hl.agg.counter(ht[f'f{i}']) for i in range(8)]))


@benchmark()
def benchmark_table_aggregate_take_by_strings(many_strings_ht):
    ht = hl.read_table(str(many_strings_ht))
    ht.aggregate(hl.tuple([hl.agg.take(ht['f18'], 25, ordering=ht[f'f{i}']) for i in range(18)]))


@benchmark()
def benchmark_table_aggregate_downsample_dense(many_ints_ht):
    ht = hl.read_table(str(many_ints_ht))
    ht.aggregate(tuple([hl.agg.downsample(ht[f'i{i}'], ht['i3'], label=hl.str(ht['i4'])) for i in range(3)]))


@benchmark()
def benchmark_table_aggregate_downsample_worst_case():
    ht = hl.utils.range_table(250_000_000, 8)
    ht.aggregate(hl.agg.downsample(ht.idx, -ht.idx))


@benchmark()
@pytest.mark.skip(reason='FIXME: this needs fixtures to accurately measure downsample (rather than randomness')
def benchmark_table_aggregate_downsample_sparse():
    ht = hl.utils.range_table(250_000_000, 8)
    ht.aggregate(hl.agg.downsample(hl.rand_norm() ** 5, hl.rand_norm() ** 5))


@benchmark()
def benchmark_table_aggregate_linreg(many_ints_ht):
    ht = hl.read_table(str(many_ints_ht))
    ht.aggregate(hl.agg.array_agg(lambda i: hl.agg.linreg(ht.i0 + i, [ht.i1, ht.i2, ht.i3, ht.i4]), hl.range(75)))


@benchmark()
def benchmark_table_take(many_strings_ht):
    ht = hl.read_table(str(many_strings_ht))
    ht.take(100)


@benchmark()
def benchmark_table_show(many_strings_ht):
    ht = hl.read_table(str(many_strings_ht))
    ht.show(100)


@benchmark()
def benchmark_table_expr_take(many_strings_ht):
    ht = hl.read_table(str(many_strings_ht))
    hl.tuple([ht.f1, ht.f2]).take(100)


@benchmark()
def benchmark_read_force_count_partitions(many_partitions_ht):
    hl.read_table(str(many_partitions_ht))._force_count()


@benchmark()
@pytest.mark.parametrize('n,n_partitions', [(10_000_000, 1000), (10_000_000, 100), (10_000_000, 10)])
def benchmark_write_range_table(tmp_path, n, n_partitions):
    ht = hl.utils.range_table(n, n_partitions)
    ht.write(str(tmp_path / 'tmp.ht'))


@benchmark()
@pytest.mark.parametrize('many_partitions_ht', [1_000], indirect=True)
def benchmark_read_with_index(many_partitions_ht):
    rows = 10_000_000
    bins = 1_000
    width = rows // bins
    intervals = [hl.Interval(start=i, end=i + width) for i in range(0, rows, width)]
    ht = hl.read_table(str(many_partitions_ht), _intervals=intervals)
    ht._force_count()


many_partitions_ht1 = many_partitions_ht
many_partitions_ht2 = many_partitions_ht


@benchmark()
def benchmark_union_partitions_table(many_partitions_ht1, many_partitions_ht2):
    ht1 = hl.read_table(str(many_partitions_ht1))
    ht2 = hl.read_table(str(many_partitions_ht2))
    ht1.union(ht2)._force_count()


@benchmark()
def benchmark_join_partitions_table(many_partitions_ht1, many_partitions_ht2):
    ht1 = hl.read_table(str(many_partitions_ht1))
    ht2 = hl.read_table(str(many_partitions_ht2))
    ht1.join(ht2)._force_count()


@benchmark()
def benchmark_group_by_collect_per_row(gnomad_dp_sim):
    ht = hl.read_matrix_table(str(gnomad_dp_sim)).localize_entries('e', 'c')
    ht.group_by(*ht.key).aggregate(value=hl.agg.collect(ht.row_value))._force_count()


@benchmark()
def benchmark_group_by_take_rekey(gnomad_dp_sim):
    ht = hl.read_matrix_table(str(gnomad_dp_sim)).localize_entries('e', 'c')
    ht.group_by(k=hl.int(ht.row_idx / 50)).aggregate(value=hl.agg.take(ht.row_value, 1))._force_count()


@benchmark()
def benchmark_table_scan_sum_1k_partitions():
    ht = hl.utils.range_table(1000000, n_partitions=1000)
    ht = ht.annotate(x=hl.scan.sum(ht.idx))
    ht._force_count()


@benchmark()
def benchmark_table_scan_prev_non_null():
    ht = hl.utils.range_table(100000000, n_partitions=10)
    ht = ht.annotate(x=hl.range(0, ht.idx % 25))
    ht = ht.annotate(y=hl.scan._prev_nonnull(ht.row))
    ht._force_count()


@benchmark()
def benchmark_test_map_filter_region_memory():
    high_mem_table = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(100_000_000))
    high_mem_table = high_mem_table.filter(high_mem_table.idx % 2 == 0)
    assert high_mem_table._force_count() == 15


@benchmark()
def benchmark_test_head_and_tail_region_memory():
    high_mem_table = hl.utils.range_table(100).annotate(big_array=hl.zeros(100_000_000))
    high_mem_table = high_mem_table.head(30)
    high_mem_table._force_count()


@benchmark()
def benchmark_test_inner_join_region_memory():
    high_mem_table = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(50_000_000))
    high_mem_table2 = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(50_000_000))
    joined = high_mem_table.join(high_mem_table2)
    joined._force_count()


@benchmark()
def benchmark_test_left_join_region_memory():
    high_mem_table = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(50_000_000))
    high_mem_table2 = hl.utils.range_table(30).naive_coalesce(1).annotate(big_array=hl.zeros(50_000_000))
    joined = high_mem_table.join(high_mem_table2, how='left')
    joined._force_count()
