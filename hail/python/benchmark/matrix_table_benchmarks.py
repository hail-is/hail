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