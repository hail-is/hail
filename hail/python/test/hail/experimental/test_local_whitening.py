import numpy as np
from numpy.random import default_rng
from hail.methods.pca import _make_tsm
import hail as hl
from ..helpers import *

def naive_whiten(X, w):
    m, n = np.shape(X)
    Xw = np.zeros((m, n))
    for j in range(n):
        Q, _ = np.linalg.qr(X[:, max(j - w, 0): j])
        Xw[:, j] = X[:, j] - Q @ (Q.T @ X[:, j])
    return Xw.T


def run_local_whitening_test(vec_size, num_rows, chunk_size, window_size, partition_size, initial_num_partitions):
    rng = default_rng()
    data = rng.normal(size=(num_rows, vec_size))
    mt = hl.utils.range_matrix_table(num_rows, vec_size, n_partitions=initial_num_partitions)
    mt = mt.annotate_globals(data=data)
    mt = mt.annotate_entries(x=mt.data[mt.row_idx, mt.col_idx])

    tsm = _make_tsm(mt.x, chunk_size, partition_size=partition_size, whiten_window_size=window_size)
    ht = tsm.block_table
    whitened_hail = np.vstack(ht.aggregate(hl.agg.collect(tsm.block_expr)))
    whitened_naive = naive_whiten(data.T, window_size)
    np.testing.assert_allclose(whitened_hail, whitened_naive, rtol=5e-05)

@test_timeout(local=5 * 60, batch=12 * 60)
def test_local_whitening():
    run_local_whitening_test(
        vec_size=100,
        num_rows=10000,
        chunk_size=32,
        window_size=64,
        partition_size=32 * 40,
        initial_num_partitions=50)

@test_timeout(local=5 * 60, batch=12 * 60)
def test_local_whitening_singleton_final_partition():
    run_local_whitening_test(
        vec_size=100,
        num_rows=32 * 40 * 8 + 1,  # = 10241
        chunk_size=32,
        window_size=64,
        partition_size=32 * 40,
        initial_num_partitions=50)
