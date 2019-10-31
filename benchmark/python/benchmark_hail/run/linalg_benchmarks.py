import hail as hl

from .utils import benchmark


@benchmark
def block_matrix_nested_multiply():
    bm = hl.linalg.BlockMatrix.random(8 * 1024, 8 * 1024).checkpoint(hl.utils.new_temp_file(suffix='bm'))
    path = hl.utils.new_temp_file(suffix='bm')
    ((bm @ bm) @ bm @ bm @ (bm @ bm)).write(path, overwrite=True)
