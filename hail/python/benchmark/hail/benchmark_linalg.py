import hail as hl
from benchmark.tools import benchmark


@benchmark()
def benchmark_block_matrix_nested_multiply(tmp_path):
    bm = hl.linalg.BlockMatrix.random(8 * 1024, 8 * 1024)
    bm = bm.checkpoint(str(tmp_path / 'checkpoint.mt'))
    bm = (bm @ bm) @ bm @ bm @ (bm @ bm)
    bm.write(str(tmp_path / 'result.mt'), overwrite=True)


@benchmark()
def benchmark_make_ndarray():
    ht = hl.utils.range_table(200_000)
    ht = ht.annotate(x=hl.nd.array(hl.range(ht.idx)))
    ht._force_count()


@benchmark()
def benchmark_ndarray_addition():
    arr = hl.nd.ones((1024, 1024))
    hl.eval(arr + arr)


@benchmark()
def benchmark_ndarray_matmul_int64():
    arr = hl.nd.arange(1024 * 1024).map(hl.int64).reshape((1024, 1024))
    hl.eval(arr @ arr)


@benchmark()
def benchmark_ndarray_matmul_float64():
    arr = hl.nd.arange(1024 * 1024).map(hl.float64).reshape((1024, 1024))
    hl.eval(arr @ arr)


@benchmark()
def benchmark_blockmatrix_write_from_entry_expr_range_mt(tmp_path):
    mt = hl.utils.range_matrix_table(40_000, 40_000, n_partitions=4)
    path = str(tmp_path / 'result.bm')
    hl.linalg.BlockMatrix.write_from_entry_expr(mt.row_idx + mt.col_idx, path)


@benchmark()
def benchmark_blockmatrix_write_from_entry_expr_range_mt_standardize(tmp_path):
    mt = hl.utils.range_matrix_table(40_000, 40_000, n_partitions=4)
    path = str(tmp_path / 'result.bm')
    hl.linalg.BlockMatrix.write_from_entry_expr(
        mt.row_idx + mt.col_idx, path, mean_impute=True, center=True, normalize=True
    )


@benchmark()
def benchmark_sum_table_of_ndarrays():
    ht = hl.utils.range_table(400).annotate(nd=hl.nd.ones((4096, 4096)))
    ht.aggregate(hl.agg.ndarray_sum(ht.nd))


@benchmark()
def benchmark_block_matrix_to_matrix_table_row_major():
    mt = hl.utils.range_matrix_table(20_000, 20_000, n_partitions=4)
    bm = hl.linalg.BlockMatrix.from_entry_expr(mt.row_idx + mt.col_idx)
    bm.to_matrix_table_row_major()._force_count_rows()


@benchmark()
def benchmark_king(tmp_path):
    mt = hl.balding_nichols_model(6, n_variants=10000, n_samples=4096)
    path = str(tmp_path / 'result.mt')
    hl.king(mt.GT).write(path, overwrite=True)
