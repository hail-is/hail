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

    def test_local_whitening(self):
        m = 10000
        n = 100
        w = 64
        chunk_size = 32
        n_partitions = 50
        chunks_per_partition = 40
        rng = default_rng()
        data = rng.normal(size=(m, n))
        mt = hl.utils.range_matrix_table(m, n, n_partitions=n_partitions)
        mt = mt.annotate_globals(data=data)
        mt = mt.annotate_entries(x=mt.data[mt.row_idx, mt.col_idx])

        ht = whiten(mt.x, chunk_size, w, chunks_per_partition * chunk_size)
        whitened_hail = np.vstack(ht.aggregate(hl.agg.collect(ht.ndarray.T)))
        whitened_naive = naive_whiten(data.T, w)
        np.testing.assert_allclose(whitened_hail, whitened_naive)