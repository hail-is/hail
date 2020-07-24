import hail as hl

from .utils import benchmark


@benchmark()
def block_matrix_nested_multiply():
    bm = hl.linalg.BlockMatrix.random(8 * 1024, 8 * 1024).checkpoint(hl.utils.new_temp_file(extension='bm'))
    path = hl.utils.new_temp_file(extension='bm')
    ((bm @ bm) @ bm @ bm @ (bm @ bm)).write(path, overwrite=True)


@benchmark()
def make_ndarray_bench():
    ht = hl.utils.range_table(200_000)
    ht = ht.annotate(x=hl.nd.array(hl.range(200_000)))
    ht._force_count()


@benchmark()
def ndarray_addition_benchmark():
    arr = hl.nd.ones((1024, 1024))
    hl.eval(arr + arr)


@benchmark()
def ndarray_matmul_int64_benchmark():
    arr = hl.nd.arange(1024 * 1024).map(hl.int64).reshape((1024, 1024))
    hl.eval(arr @ arr)


@benchmark()
def ndarray_matmul_float64_benchmark():
    arr = hl.nd.arange(1024 * 1024).map(hl.float64).reshape((1024, 1024))
    hl.eval(arr @ arr)


@benchmark()
def blockmatrix_write_from_entry_expr_range_mt():
    mt = hl.utils.range_matrix_table(40_000, 40_000, n_partitions=4)
    path = hl.utils.new_temp_file(extension='bm')
    hl.linalg.BlockMatrix.write_from_entry_expr(mt.row_idx + mt.col_idx, path)


@benchmark()
def blockmatrix_write_from_entry_expr_range_mt_standardize():
    mt = hl.utils.range_matrix_table(40_000, 40_000, n_partitions=4)
    path = hl.utils.new_temp_file(extension='bm')
    hl.linalg.BlockMatrix.write_from_entry_expr(mt.row_idx + mt.col_idx, path, mean_inpute=True, center=True,
                                                normalize=True)
