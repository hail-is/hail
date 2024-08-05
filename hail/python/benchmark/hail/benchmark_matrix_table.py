import pytest

import hail as hl
from benchmark.tools import benchmark


@benchmark()
def benchmark_matrix_table_decode_and_count(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt._force_count_rows()


@benchmark()
def benchmark_matrix_table_decode_and_count_just_gt(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt)).select_entries('GT')
    mt._force_count_rows()


@benchmark()
def benchmark_matrix_table_array_arithmetic(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.filter_rows(mt.alleles.length() == 2)
    mt.select_entries(dosage=hl.pl_dosage(mt.PL)).select_rows()._force_count_rows()


@benchmark()
def benchmark_matrix_table_entries_table(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.entries()._force_count()


@benchmark()
def benchmark_matrix_table_entries_table_no_key(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt)).key_rows_by().key_cols_by()
    mt.entries()._force_count()


@benchmark()
def benchmark_matrix_table_rows_force_count(profile25_mt):
    ht = hl.read_matrix_table(str(profile25_mt)).rows().key_by()
    ht._force_count()


@benchmark()
def benchmark_matrix_table_show(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.show(100)


@benchmark()
def benchmark_matrix_table_rows_show(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.rows().show(100)


@benchmark()
def benchmark_matrix_table_cols_show(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.cols().show(100)


@benchmark()
def benchmark_matrix_table_take_entry(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.GT.take(100)


@benchmark()
def benchmark_matrix_table_entries_show(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.entries().show()


@benchmark()
def benchmark_matrix_table_take_row(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.info.AF.take(100)


@benchmark()
def benchmark_matrix_table_take_col(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.s.take(100)


@benchmark()
def benchmark_write_range_matrix_table_p100(tmp_path):
    mt = hl.utils.range_matrix_table(n_rows=1_000_000, n_cols=10, n_partitions=100)
    mt = mt.annotate_entries(x=mt.col_idx + mt.row_idx)
    mt.write(str(tmp_path / 'tmp.mt'))


@benchmark()
def benchmark_write_profile_mt(profile25_mt, tmp_path):
    hl.read_matrix_table(str(profile25_mt)).write(str(tmp_path / 'tmp.mt'))


@benchmark()
def benchmark_matrix_table_rows_is_transition(profile25_mt):
    ht = hl.read_matrix_table(str(profile25_mt)).rows().key_by()
    ht.select(is_snp=hl.is_snp(ht.alleles[0], ht.alleles[1]))._force_count()


@benchmark()
def benchmark_matrix_table_filter_entries(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.filter_entries((mt.GQ > 8) & (mt.DP > 2))._force_count_rows()


@benchmark()
def benchmark_matrix_table_filter_entries_unfilter(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.filter_entries((mt.GQ > 8) & (mt.DP > 2)).unfilter_entries()._force_count_rows()


@benchmark()
def benchmark_matrix_table_nested_annotate_rows_annotate_entries(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.annotate_rows(r0=mt.info.AF[0] + 1)
    mt = mt.annotate_entries(e0=mt.GQ + 5)
    for i in range(1, 20):
        mt = mt.annotate_rows(**{f'r{i}': mt[f'r{i-1}'] + 1})
        mt = mt.annotate_entries(**{f'e{i}': mt[f'e{i-1}'] + 1})
    mt._force_count_rows()


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


@benchmark()
def benchmark_matrix_table_many_aggs_row_wise(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.annotate_rows(**many_aggs(mt))
    mt.rows()._force_count()


@benchmark()
def benchmark_matrix_table_many_aggs_col_wise(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.annotate_cols(**many_aggs(mt))
    mt.cols()._force_count()


@benchmark()
def benchmark_matrix_table_aggregate_entries(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.aggregate_entries(hl.agg.stats(mt.GQ))


@benchmark()
def benchmark_matrix_table_call_stats_star_star(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt.annotate_rows(**hl.agg.call_stats(mt.GT, mt.alleles))._force_count_rows()


@benchmark()
@pytest.mark.skip(reason='never finishes')
def benchmark_gnomad_coverage_stats(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))

    def get_coverage_expr(mt):
        cov_arrays = hl.literal({
            x: [1, 1, 1, 1, 1, 1, 1, 1, 0]
            if x >= 50
            else [1, 1, 1, 1, 1, 1, 1, 0, 0]
            if x >= 30
            else ([1] * (i + 2)) + ([0] * (7 - i))
            for i, x in enumerate(range(5, 100, 5))
        })

        return hl.bind(
            lambda array_expr: hl.struct(**{
                f'over_{x}': hl.int32(array_expr[i]) for i, x in enumerate([1, 5, 10, 15, 20, 25, 30, 50, 100])
            }),
            hl.agg.array_sum(
                hl.case()
                .when(mt.x >= 100, [1, 1, 1, 1, 1, 1, 1, 1, 1])
                .when(mt.x >= 5, cov_arrays[mt.x - (mt.x % 5)])
                .when(mt.x >= 1, [1, 0, 0, 0, 0, 0, 0, 0, 0])
                .default([0, 0, 0, 0, 0, 0, 0, 0, 0])
            ),
        )

    mt = mt.annotate_rows(mean=hl.agg.mean(mt.x), median=hl.median(hl.agg.collect(mt.x)), **get_coverage_expr(mt))
    mt.rows()._force_count()


@benchmark()
def gnomad_coverage_stats_optimized(gnomad_dp_sim):
    mt = hl.read_matrix_table(str(gnomad_dp_sim))
    mt = mt.annotate_rows(
        mean=hl.agg.mean(mt.x),
        count_array=hl.rbind(hl.agg.counter(hl.min(100, mt.x)), lambda c: hl.range(0, 100).map(lambda i: c.get(i, 0))),
    )
    mt = mt.annotate_rows(
        median=hl.rbind(
            hl.sum(mt.count_array) / 2,
            lambda s: hl.find(lambda x: x > s, hl.array_scan(lambda i, j: i + j, 0, mt.count_array)),
        ),
        **{f'above_{x}': hl.sum(mt.count_array[x:]) for x in [1, 5, 10, 15, 20, 25, 30, 50, 100]},
    )
    mt.rows()._force_count()


@benchmark()
def benchmark_per_row_stats_star_star(gnomad_dp_sim):
    mt = hl.read_matrix_table(str(gnomad_dp_sim))
    mt.annotate_rows(**hl.agg.stats(mt.x))._force_count_rows()


@benchmark()
def benchmark_read_decode_gnomad_coverage(gnomad_dp_sim):
    hl.read_matrix_table(str(gnomad_dp_sim))._force_count_rows()


@benchmark()
def benchmark_import_bgen_force_count_just_gp(sim_ukb_bgen, sim_ukb_sample):
    mt = hl.import_bgen(str(sim_ukb_bgen), sample_file=str(sim_ukb_sample), entry_fields=['GP'], n_partitions=8)
    mt._force_count_rows()


@benchmark()
def benchmark_import_bgen_force_count_all(sim_ukb_bgen, sim_ukb_sample):
    mt = hl.import_bgen(
        str(sim_ukb_bgen), sample_file=str(sim_ukb_sample), entry_fields=['GT', 'GP', 'dosage'], n_partitions=8
    )
    mt._force_count_rows()


@benchmark()
def benchmark_import_bgen_info_score(sim_ukb_bgen, sim_ukb_sample):
    mt = hl.import_bgen(str(sim_ukb_bgen), sample_file=str(sim_ukb_sample), entry_fields=['GP'], n_partitions=8)
    mt = mt.annotate_rows(info_score=hl.agg.info_score(mt.GP))
    mt.rows().select('info_score')._force_count()


@benchmark()
def benchmark_import_bgen_filter_count(sim_ukb_bgen, sim_ukb_sample):
    mt = hl.import_bgen(str(sim_ukb_bgen), sample_file=str(sim_ukb_sample), entry_fields=['GT', 'GP'], n_partitions=8)
    mt = mt.filter_rows(mt.alleles == ['A', 'T'])
    mt._force_count_rows()


@benchmark()
def benchmark_export_range_matrix_table_entry_field_p100(tmp_path):
    mt = hl.utils.range_matrix_table(n_rows=1_000_000, n_cols=10, n_partitions=100)
    mt = mt.annotate_entries(x=mt.col_idx + mt.row_idx)
    mt.x.export(str(tmp_path / 'result.txt'))


@benchmark()
def benchmark_export_range_matrix_table_row_p100(tmp_path):
    mt = hl.utils.range_matrix_table(n_rows=1_000_000, n_cols=10, n_partitions=100)
    mt.row.export(str(tmp_path / 'result.txt'))


@benchmark()
def benchmark_export_range_matrix_table_col_p100(tmp_path):
    mt = hl.utils.range_matrix_table(n_rows=1_000_000, n_cols=10, n_partitions=100)
    mt.col.export(str(tmp_path / 'result.txt'))


@benchmark()
def benchmark_large_range_matrix_table_sum():
    mt = hl.utils.range_matrix_table(n_cols=500000, n_rows=10000, n_partitions=2500)
    mt = mt.annotate_entries(x=mt.col_idx + mt.row_idx)
    mt.annotate_cols(foo=hl.agg.sum(mt.x))._force_count_cols()


@benchmark()
def benchmark_kyle_sex_specific_qc(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    mt = mt.annotate_cols(sex=hl.if_else(hl.rand_bool(0.5), 'Male', 'Female'))
    (num_males, num_females) = mt.aggregate_cols((
        hl.agg.count_where(mt.sex == 'Male'),
        hl.agg.count_where(mt.sex == 'Female'),
    ))
    mt = mt.annotate_rows(
        male_hets=hl.agg.count_where(mt.GT.is_het() & (mt.sex == 'Male')),
        male_homvars=hl.agg.count_where(mt.GT.is_hom_var() & (mt.sex == 'Male')),
        male_calls=hl.agg.count_where(hl.is_defined(mt.GT) & (mt.sex == 'Male')),
        female_hets=hl.agg.count_where(mt.GT.is_het() & (mt.sex == 'Female')),
        female_homvars=hl.agg.count_where(mt.GT.is_hom_var() & (mt.sex == 'Female')),
        female_calls=hl.agg.count_where(hl.is_defined(mt.GT) & (mt.sex == 'Female')),
    )

    mt = mt.annotate_rows(
        call_rate=(
            hl.case()
            .when(mt.locus.in_y_nonpar(), (mt.male_calls / num_males))
            .when(mt.locus.in_x_nonpar(), (mt.male_calls + 2 * mt.female_calls) / (num_males + 2 * num_females))
            .default((mt.male_calls + mt.female_calls) / (num_males + num_females))
        ),
        AC=(
            hl.case()
            .when(mt.locus.in_y_nonpar(), mt.male_homvars)
            .when(mt.locus.in_x_nonpar(), mt.male_homvars + mt.female_hets + 2 * mt.female_homvars)
            .default(mt.male_hets + 2 * mt.male_homvars + mt.female_hets + 2 * mt.female_homvars)
        ),
        AN=(
            hl.case()
            .when(mt.locus.in_y_nonpar(), mt.male_calls)
            .when(mt.locus.in_x_nonpar(), mt.male_calls + 2 * mt.female_calls)
            .default(2 * mt.male_calls + 2 * mt.female_calls)
        ),
    )

    mt.rows()._force_count()


@benchmark()
def benchmark_matrix_table_scan_count_rows_2():
    mt = hl.utils.range_matrix_table(n_rows=200_000_000, n_cols=10, n_partitions=16)
    mt = mt.annotate_rows(x=hl.scan.count())
    mt._force_count_rows()


@benchmark()
def benchmark_matrix_table_scan_count_cols_2():
    mt = hl.utils.range_matrix_table(n_cols=10_000_000, n_rows=10)
    mt = mt.annotate_cols(x=hl.scan.count())
    mt._force_count_rows()


@benchmark()
def benchmark_matrix_multi_write_nothing(tmp_path):
    mt = hl.utils.range_matrix_table(1, 1, n_partitions=1)
    mts = [mt] * 1000
    hl.experimental.write_matrix_tables(mts, str(tmp_path / 'multi-write'))


@benchmark()
def benchmark_mt_localize_and_collect(profile25_mt):
    mt = hl.read_matrix_table(str(profile25_mt))
    ht = mt.localize_entries("ent")
    ht.head(150).collect()


@benchmark()
def benchmark_mt_group_by_memory_usage(random_doubles_mt):
    mt = hl.read_matrix_table(str(random_doubles_mt))
    mt = mt.group_rows_by(new_idx=mt.row_idx % 3).aggregate(x=hl.agg.mean(mt.x))
    mt._force_count_rows()
