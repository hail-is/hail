from os import path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import hail as hl
from .utils import benchmark, resource


@benchmark
def matrix_table_decode_and_count():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt._force_count_rows()


@benchmark
def matrix_table_array_arithmetic():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt = mt.filter_rows(mt.alleles.length() == 2)
    mt.select_entries(dosage=hl.pl_dosage(mt.PL)).select_rows()._force_count_rows()


@benchmark
def matrix_table_entries_table():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.entries()._force_count()


@benchmark
def matrix_table_entries_table_no_key():
    mt = hl.read_matrix_table(resource('profile.mt')).key_rows_by().key_cols_by()
    mt.entries()._force_count()


@benchmark
def matrix_table_rows_force_count():
    ht = hl.read_matrix_table(resource('profile.mt')).rows().key_by()
    ht._force_count()


@benchmark
def matrix_table_show():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.show(100)

@benchmark
def matrix_table_rows_show():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.rows().show(100)

@benchmark
def matrix_table_cols_show():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.cols().show(100)

@benchmark
def matrix_table_take_entry():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.GT.take(100)

@benchmark
def matrix_table_take_row():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.info.AF.take(100)

@benchmark
def matrix_table_take_col():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.s.take(100)

@benchmark
def write_range_matrix_table_p100():
    with TemporaryDirectory() as tmpdir:
        mt = hl.utils.range_matrix_table(n_rows=1_000_000, n_cols=10, n_partitions=100)
        mt = mt.annotate_entries(x=mt.col_idx + mt.row_idx)
        mt.write(path.join(tmpdir, 'tmp.mt'))


@benchmark
def matrix_table_rows_is_transition():
    ht = hl.read_matrix_table(resource('profile.mt')).rows().key_by()
    ht.select(is_snp=hl.is_snp(ht.alleles[0], ht.alleles[1]))._force_count()


@benchmark
def matrix_table_filter_entries():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.filter_entries((mt.GQ > 8) & (mt.DP > 2))._force_count_rows()


@benchmark
def matrix_table_filter_entries_unfilter():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.filter_entries((mt.GQ > 8) & (mt.DP > 2)).unfilter_entries()._force_count_rows()


def many_aggs(mt):
    aggs = [
        hl.agg.count_where(mt.GT.is_hom_ref()),
        hl.agg.count_where(mt.GT.is_het()),
        hl.agg.count_where(mt.GT.is_hom_var()),
        hl.agg.count_where(mt.GT.is_non_ref()),
        hl.agg.count_where(mt.GT.n_alt_alleles() == 2),
        hl.agg.count_where(mt.GT.phased),
        hl.agg.count_where(mt.GT.is_haploid()),
        hl.agg.count_where(mt.GT.is_diploid()),
        hl.agg.count_where(mt.GT.ploidy == 2),
        hl.agg.fraction(mt.AD[0] > 0),
        hl.agg.fraction(mt.AD[0] < 0),
        hl.agg.fraction(mt.AD.length() < 0),
        hl.agg.fraction(mt.AD.length() > 0),
        hl.agg.fraction(mt.PL[0] > 0),
        hl.agg.fraction(mt.PL[0] < 0),
        hl.agg.fraction(mt.PL.length() < 0),
        hl.agg.fraction(mt.PL.length() > 0),
        hl.agg.fraction(mt.GQ < 0),
        hl.agg.fraction(mt.GQ > 0),
        hl.agg.fraction(mt.GQ % 2 == 0),
        hl.agg.fraction(mt.GQ % 2 != 0),
        hl.agg.fraction(mt.GQ / 5 < 10),
        hl.agg.fraction(mt.GQ / 5 <= 10),
        hl.agg.fraction(mt.GQ / 5 > 10),
        hl.agg.fraction(mt.GQ / 5 >= 10),
        hl.agg.fraction(mt.DP < 0),
        hl.agg.fraction(mt.DP > 0),
        hl.agg.fraction(mt.DP % 2 == 0),
        hl.agg.fraction(mt.DP % 2 != 0),
        hl.agg.fraction(mt.DP / 5 < 10),
        hl.agg.fraction(mt.DP / 5 <= 10),
        hl.agg.fraction(mt.DP / 5 > 10),
        hl.agg.fraction(mt.DP / 5 >= 10),
    ]
    return {f'x{i}': expr for i, expr in enumerate(aggs)}


@benchmark
def matrix_table_many_aggs_row_wise():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt = mt.annotate_rows(**many_aggs(mt))
    mt.rows()._force_count()


@benchmark
def matrix_table_many_aggs_col_wise():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt = mt.annotate_cols(**many_aggs(mt))
    mt.cols()._force_count()


@benchmark
def matrix_table_aggregate_entries():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.aggregate_entries(hl.agg.stats(mt.GQ))


@benchmark
def matrix_table_call_stats_star_star():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.annotate_rows(**hl.agg.call_stats(mt.GT, mt.alleles))._force_count_rows()


# @benchmark never finishes
def gnomad_coverage_stats():
    mt = hl.read_matrix_table(resource('gnomad_dp_simulation.mt'))

    def get_coverage_expr(mt):
        cov_arrays = hl.literal({
            x:
                [1, 1, 1, 1, 1, 1, 1, 1, 0] if x >= 50
                else [1, 1, 1, 1, 1, 1, 1, 0, 0] if x >= 30
                else ([1] * (i + 2)) + ([0] * (7 - i))
            for i, x in enumerate(range(5, 100, 5))
        })

        return hl.bind(
            lambda array_expr: hl.struct(
                **{
                    f'over_{x}': hl.int32(array_expr[i]) for i, x in enumerate([1, 5, 10, 15, 20, 25, 30, 50, 100])
                }
            ),
            hl.agg.array_sum(hl.case()
                             .when(mt.x >= 100, [1, 1, 1, 1, 1, 1, 1, 1, 1])
                             .when(mt.x >= 5, cov_arrays[mt.x - (mt.x % 5)])
                             .when(mt.x >= 1, [1, 0, 0, 0, 0, 0, 0, 0, 0])
                             .default([0, 0, 0, 0, 0, 0, 0, 0, 0])))

    mt = mt.annotate_rows(mean=hl.agg.mean(mt.x),
                          median=hl.median(hl.agg.collect(mt.x)),
                          **get_coverage_expr(mt))
    mt.rows()._force_count()


@benchmark
def gnomad_coverage_stats_optimized():
    mt = hl.read_matrix_table(resource('gnomad_dp_simulation.mt'))
    mt = mt.annotate_rows(mean=hl.agg.mean(mt.x),
                          count_array=hl.rbind(hl.agg.counter(hl.min(100, mt.x)),
                                               lambda c: hl.range(0, 100).map(lambda i: c.get(i, 0))))
    mt = mt.annotate_rows(median=hl.rbind(hl.sum(mt.count_array) / 2, lambda s: hl.find(lambda x: x > s,
                                                                                        hl.array_scan(
                                                                                            lambda i, j: i + j,
                                                                                            0,
                                                                                            mt.count_array))),
                          **{f'above_{x}': hl.sum(mt.count_array[x:]) for x in [1, 5, 10, 15, 20, 25, 30, 50, 100]}
                          )
    mt.rows()._force_count()


@benchmark
def per_row_stats_star_star():
    mt = hl.read_matrix_table(resource('gnomad_dp_simulation.mt'))
    mt.annotate_rows(**hl.agg.stats(mt.x))._force_count_rows()


@benchmark
def read_decode_gnomad_coverage():
    hl.read_matrix_table(resource('gnomad_dp_simulation.mt'))._force_count_rows()


@benchmark
def import_bgen_force_count_just_gp():
    mt = hl.import_bgen(resource('sim_ukb.bgen'),
                        sample_file=resource('sim_ukb.sample'),
                        entry_fields=['GP'],
                        n_partitions=8)
    mt._force_count_rows()


@benchmark
def import_bgen_force_count_all():
    mt = hl.import_bgen(resource('sim_ukb.bgen'),
                        sample_file=resource('sim_ukb.sample'),
                        entry_fields=['GT', 'GP', 'dosage'],
                        n_partitions=8)
    mt._force_count_rows()


@benchmark
def import_bgen_info_score():
    mt = hl.import_bgen(resource('sim_ukb.bgen'),
                        sample_file=resource('sim_ukb.sample'),
                        entry_fields=['GP'],
                        n_partitions=8)
    mt = mt.annotate_rows(info_score=hl.agg.info_score(mt.GP))
    mt.rows().select('info_score')._force_count()


@benchmark
def import_bgen_filter_count():
    mt = hl.import_bgen(resource('sim_ukb.bgen'),
                        sample_file=resource('sim_ukb.sample'),
                        entry_fields=['GT', 'GP'],
                        n_partitions=8)
    mt = mt.filter_rows(mt.alleles == ['A', 'T'])
    mt._force_count_rows()

@benchmark
def export_range_matrix_table_entry_field_p100():
    with NamedTemporaryFile() as f:
        mt = hl.utils.range_matrix_table(n_rows=1_000_000, n_cols=10, n_partitions=100)
        mt = mt.annotate_entries(x=mt.col_idx + mt.row_idx)
        mt.x.export(f.name)


@benchmark
def export_range_matrix_table_row_p100():
    with NamedTemporaryFile() as f:
        mt = hl.utils.range_matrix_table(n_rows=1_000_000, n_cols=10, n_partitions=100)
        mt.row.export(f.name)


@benchmark
def export_range_matrix_table_col_p100():
    with NamedTemporaryFile() as f:
        mt = hl.utils.range_matrix_table(n_rows=1_000_000, n_cols=10, n_partitions=100)
        mt.col.export(f.name)
