import hail as hl

from .utils import benchmark, recursive_delete


@benchmark()
def block_matrix_nested_multiply():
    bm = hl.linalg.BlockMatrix.random(8 * 1024, 8 * 1024).checkpoint(hl.utils.new_temp_file(extension='bm'))
    path = hl.utils.new_temp_file(extension='bm')
    ((bm @ bm) @ bm @ bm @ (bm @ bm)).write(path, overwrite=True)
    return lambda: recursive_delete(path)


@benchmark()
def make_ndarray_bench_2():
    ht = hl.utils.range_table(200_000)
    ht = ht.annotate(x=hl.nd.array(hl.range(ht.idx)))
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
    return lambda: recursive_delete(path)


@benchmark()
def blockmatrix_write_from_entry_expr_range_mt_standardize():
    mt = hl.utils.range_matrix_table(40_000, 40_000, n_partitions=4)
    path = hl.utils.new_temp_file(extension='bm')
    hl.linalg.BlockMatrix.write_from_entry_expr(
        mt.row_idx + mt.col_idx, path, mean_impute=True, center=True, normalize=True
    )
    return lambda: recursive_delete(path)


@benchmark()
def sum_table_of_ndarrays():
    ht = hl.utils.range_table(400).annotate(nd=hl.nd.ones((4096, 4096)))
    ht.aggregate(hl.agg.ndarray_sum(ht.nd))


@benchmark()
def block_matrix_to_matrix_table_row_major():
    mt = hl.utils.range_matrix_table(20_000, 20_000, n_partitions=4)
    bm = hl.linalg.BlockMatrix.from_entry_expr(mt.row_idx + mt.col_idx)
    bm.to_matrix_table_row_major()._force_count_rows()


@benchmark()
def king():
    mt = hl.balding_nichols_model(6, n_variants=10000, n_samples=4096)
    path = hl.utils.new_temp_file(extension='mt')
    hl.king(mt.GT).write(path, overwrite=True)
