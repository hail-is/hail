from benchmark.utils import resource, benchmark

import hail as hl


@benchmark
def matrix_table_decode_and_count():
    mt = hl.read_matrix_table(resource('profile.mt'))
    mt._force_count_rows()
