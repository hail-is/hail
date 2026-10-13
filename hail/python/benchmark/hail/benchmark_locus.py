import pytest

import hail as hl
from benchmark.hail.fixtures import (
    MULTI_CONTIG_N_PARTITIONS,
    MULTI_CONTIG_N_ROWS,
    MULTI_CONTIG_SWITCH_RATES,
)
from benchmark.hail.utils import multi_contig_table


def profile25_rows(profile25_mt):
    return hl.read_matrix_table(str(profile25_mt)).rows()


def profile25_intervals(n=1_000):
    # profile25's loci lie on contig 22 between 16.05Mb and 18.09Mb. Each
    # interval covers the first half of its share of that span.
    start, stop = 16_000_000, 18_100_000
    step = (stop - start) // n
    return [
        hl.Interval(hl.Locus('22', s, 'GRCh37'), hl.Locus('22', s + step // 2, 'GRCh37'))
        for s in range(start, stop, step)
    ]


def test_write_profile_rows(profile25_mt, tmp_path):
    profile25_rows(profile25_mt).write(str(tmp_path / 'tmp.ht'))


def test_order_by_locus(profile25_rows_scrambled):
    ht = hl.read_table(str(profile25_rows_scrambled))
    ht.order_by('locus')._force_count()


def test_key_by_locus(profile25_rows_scrambled):
    ht = hl.read_table(str(profile25_rows_scrambled))
    ht.key_by('locus')._force_count()


def test_join_locus_keyed(profile25_mt):
    ht1, ht2 = (profile25_rows(profile25_mt).key_by('locus').select() for _ in range(2))
    ht1.join(ht2)._force_count()


def test_filter_intervals_many(profile25_mt):
    ht = profile25_rows(profile25_mt)
    hl.filter_intervals(ht, profile25_intervals())._force_count()


def test_interval_join_locus(profile25_mt):
    ht = profile25_rows(profile25_mt)
    intervals = hl.Table.parallelize(
        hl.literal(profile25_intervals()).map(lambda i: hl.struct(interval=i)),
        key='interval',
    )
    ht.annotate(in_interval=hl.is_defined(intervals[ht.locus]))._force_count()


def test_group_by_contig(profile25_rows_scrambled):
    ht = hl.read_table(str(profile25_rows_scrambled))
    ht.group_by(contig=ht.locus.contig).aggregate(n=hl.agg.count())._force_count()


def test_filter_contig_eq(profile25_rows_scrambled):
    ht = hl.read_table(str(profile25_rows_scrambled))
    ht.filter(ht.locus.contig == '22')._force_count()


@pytest.mark.parametrize('contig_switch_rate', MULTI_CONTIG_SWITCH_RATES)
def test_write_multi_contig(contig_switch_rate, tmp_path):
    ht = multi_contig_table(MULTI_CONTIG_N_ROWS, contig_switch_rate, MULTI_CONTIG_N_PARTITIONS)
    ht.write(str(tmp_path / 'tmp.ht'))


def test_read_force_count_multi_contig(multi_contig_ht):
    hl.read_table(str(multi_contig_ht))._force_count()


def test_order_by_locus_multi_contig(multi_contig_ht):
    ht = hl.read_table(str(multi_contig_ht))
    ht.order_by('locus')._force_count()


def test_key_by_locus_multi_contig(multi_contig_ht):
    ht = hl.read_table(str(multi_contig_ht))
    ht.key_by('locus')._force_count()


def test_group_by_contig_multi_contig(multi_contig_ht):
    ht = hl.read_table(str(multi_contig_ht))
    ht.group_by(contig=ht.locus.contig).aggregate(n=hl.agg.count())._force_count()


def test_filter_contig_eq_multi_contig(multi_contig_ht):
    ht = hl.read_table(str(multi_contig_ht))
    ht.filter(ht.locus.contig == 'chr1')._force_count()
