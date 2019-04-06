from benchmark.utils import resource, benchmark

import hail as hl


@benchmark
def matrix_table_decode_and_count():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt._force_count_rows()

@benchmark
def matrix_table_array_arithmetic():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt = mt.filter_rows(mt.alleles.length() == 2)
    mt.select_entries(dosage = hl.pl_dosage(mt.PL)).select_rows()._force_count_rows()


@benchmark
def matrix_table_entries_table():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt.entries()._force_count()

@benchmark
def matrix_table_rows_force_count():
    ht = hl.read_matrix_table(resource('profile.mt')).rows().key_by()
    ht._force_count()

@benchmark
def matrix_table_rows_is_transition():
    ht = hl.read_matrix_table(resource('profile.mt')).rows().key_by()
    ht.select(is_snp = hl.is_snp(ht.alleles[0], ht.alleles[1]))._force_count()

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