import hail as hl

from os import path
from tempfile import TemporaryDirectory
from .utils import benchmark, resource


@benchmark
def table_key_by_shuffle():
    N = 1_000_000
    ht = hl.utils.range_table(N)
    ht = ht.key_by(x=N - ht.idx)
    ht._force_count()


@benchmark
def table_group_by_aggregate_sorted():
    N = 10_000_000
    ht = hl.utils.range_table(N)
    ht = ht.group_by(x=ht.idx // 1000).aggregate(y=hl.agg.count())
    ht._force_count()


@benchmark
def table_group_by_aggregate_unsorted():
    N = 10_000_000
    ht = hl.utils.range_table(N)
    ht = ht.group_by(x=(N - ht.idx) // 1000).aggregate(y=hl.agg.count())
    ht._force_count()


@benchmark
def table_range_force_count():
    hl.utils.range_table(100_000_000)._force_count()


@benchmark
def table_python_construction():
    N = 100
    ht = hl.utils.range_table(100)
    for i in range(N):
        ht = ht.annotate(**{f'x_{i}': 0})


@benchmark
def table_big_aggregate_compilation():
    N = 1_000
    ht = hl.utils.range_table(1)
    expr = tuple([hl.agg.fraction(ht.idx % i == 0) for i in range(N) if i > 0])
    ht.aggregate(expr)


@benchmark
def table_big_aggregate_compile_and_execute():
    N = 1_000
    M = 1_000_000
    ht = hl.utils.range_table(M)
    expr = tuple([hl.agg.fraction(ht.idx % i == 0) for i in range(N) if i > 0])
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
    N = 10_000_000
    M = 100
    ht = hl.utils.range_table(N)
    ht.aggregate(hl.agg.array_sum(hl.range(0, M)))

@benchmark
def table_annotate_many_flat():
    N = 1_000_000
    M = 100
    ht = hl.utils.range_table(N)
    ht = ht.annotate(**{f'x{i}': i + ht.idx for i in range(M)})
    ht._force_count()


@benchmark
def table_annotate_many_nested_no_dependence():
    N = 1_000_000
    M = 100
    ht = hl.utils.range_table(N)
    for i in range(M):
        ht = ht.annotate(**{f'x{i}': i + ht.idx})
    ht._force_count()


@benchmark
def table_annotate_many_nested_dependence():
    N = 1_000_000
    M = 100
    ht = hl.utils.range_table(N).annotate(x0=1)
    for i in range(1, M):
        ht = ht.annotate(**{f'x{i}': i + ht[f'x{i - 1}']})
    ht._force_count()

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
