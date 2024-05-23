import hail as hl

from .resources import *
from .utils import benchmark


@benchmark(args=profile_25.handle('mt'))
def shuffle_key_rows_by_mt(mt_path):
    mt = hl.read_matrix_table(mt_path)
    mt = mt.annotate_rows(reversed_position_locus=hl.struct(contig=mt.locus.contig, position=-mt.locus.position))
    mt = mt.key_rows_by(mt.reversed_position_locus)
    mt._force_count_rows()


@benchmark()
def shuffle_order_by_10m_int():
    t = hl.utils.range_table(10_000_000, n_partitions=100)
    t = t.order_by(-t.idx)
    t._force_count()


@benchmark()
def shuffle_key_rows_by_4096_byte_rows():
    mt = hl.utils.range_matrix_table(100_000, (1 << 12) // 4)
    mt = mt.annotate_entries(entry=mt.row_idx * mt.col_idx)
    mt = mt.key_rows_by(backward_rows_idx=-mt.row_idx)
    mt._force_count_rows()


@benchmark()
def shuffle_key_rows_by_65k_byte_rows():
    mt = hl.utils.range_matrix_table(10_000, (1 << 16) // 4)
    mt = mt.annotate_entries(entry=mt.row_idx * mt.col_idx)
    mt = mt.key_rows_by(backward_rows_idx=-mt.row_idx)
    mt._force_count_rows()


@benchmark(args=many_ints_table.handle('ht'))
def shuffle_key_by_aggregate_bad_locality(ht_path):
    ht = hl.read_table(ht_path)
    ht.group_by(x=ht.i0 % 1000).aggregate(c=hl.agg.count(), m=hl.agg.mean(ht.i2))._force_count()


@benchmark(args=many_ints_table.handle('ht'))
def shuffle_key_by_aggregate_good_locality(ht_path):
    ht = hl.read_table(ht_path)
    divisor = 7_500_000 / 51  # should ensure each partition never overflows default buffer size
    ht.group_by(x=ht.idx // divisor).aggregate(c=hl.agg.count(), m=hl.agg.mean(ht.i2))._force_count()
