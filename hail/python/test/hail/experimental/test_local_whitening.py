import numpy as np
from numpy.random import default_rng
from hail.experimental import whiten
import hail as hl
import unittest
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

def naive_whiten(X, w):
    m, n = np.shape(X)
    Xw = np.zeros((m, n))
    for j in range(n):
        Q, _ = np.linalg.qr(X[:, max(j - w, 0): j])
        Xw[:,j] = X[:,j] - Q @ (Q.T @ X[:,j])
    return Xw.T

class Tests(unittest.TestCase):

    def run_local_whitening_test(self, vec_size, num_rows, chunk_size, window_size, partition_size, initial_num_partitions):
        rng = default_rng()
        data = rng.normal(size=(num_rows, vec_size))
        mt = hl.utils.range_matrix_table(num_rows, vec_size, n_partitions=initial_num_partitions)
        mt = mt.annotate_globals(data=data)
        mt = mt.annotate_entries(x=mt.data[mt.row_idx, mt.col_idx])

        ht = whiten(mt.x, chunk_size, window_size, partition_size)
        whitened_hail = np.vstack(ht.aggregate(hl.agg.collect(ht.ndarray.T)))
        whitened_naive = naive_whiten(data.T, window_size)
        np.testing.assert_allclose(whitened_hail, whitened_naive)

    def test_local_whitening(self):
        self.run_local_whitening_test(
            vec_size=100,
            num_rows=10000,
            chunk_size=32,
            window_size=64,
            partition_size=32*40,
            initial_num_partitions=50)