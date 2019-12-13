import hail as hl

from .utils import benchmark


@benchmark()
def block_matrix_nested_multiply():
    bm = hl.linalg.BlockMatrix.random(8 * 1024, 8 * 1024).checkpoint(hl.utils.new_temp_file(suffix='bm'))
    path = hl.utils.new_temp_file(suffix='bm')
    ((bm @ bm) @ bm @ bm @ (bm @ bm)).write(path, overwrite=True)


@benchmark()
def make_ndarray_bench():
    ht = hl.utils.range_table(200_000)
    ht = ht.annotate(x=hl._nd.array(hl.range(200_000)))
    ht._force_count()

@benchmark()
def ndarray_addition_benchmark():
    arr = hl._nd.ones((1024, 1024))
    hl.eval(arr + arr)

@benchmark()
def ndarray_matmul_int64_benchmark():
    arr = hl._nd.arange(1024 * 1024).map(hl.int64).reshape((1024, 1024))
    hl.eval(arr @ arr)

@benchmark()
def ndarray_matmul_float64_benchmark():
    arr = hl._nd.arange(1024 * 1024).map(hl.float64).reshape((1024, 1024))
    hl.eval(arr @ arr)
