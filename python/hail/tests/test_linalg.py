import unittest

import hail as hl
from hail.linalg import BlockMatrix
from .utils import resource, startTestHailContext, stopTestHailContext
import numpy as np

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext

class Tests(unittest.TestCase):
    _dataset = None

    def get_dataset(self):
        if Tests._dataset is None:
            Tests._dataset = hl.split_multi_hts(hl.import_vcf(resource('sample.vcf')))
        return Tests._dataset

    def test_from_entry_expr(self):
        mt = self.get_dataset()
        mt = mt.annotate_entries(x = hl.or_else(mt.GT.n_alt_alleles(), 0)).cache()

        a1 = BlockMatrix.from_entry_expr(hl.or_else(mt.GT.n_alt_alleles(), 0), block_size=32).to_numpy()
        a2 = BlockMatrix.from_entry_expr(mt.x, block_size=32).to_numpy()
        a3 = BlockMatrix.from_entry_expr(hl.float64(mt.x), block_size=32).to_numpy()

        self.assertTrue(np.allclose(a1, a2))
        self.assertTrue(np.allclose(a1, a3))


    def test_to_from_numpy(self):
        data = np.random.rand(110)

        a = data.reshape(10, 11)
        bm = BlockMatrix._create_block_matrix(10, 11, data.tolist(), row_major=True, block_size=4)

        self.assertTrue(np.allclose(bm.to_numpy(), a))
        self.assertTrue(np.allclose(BlockMatrix.from_numpy(a, block_size=5).to_numpy(), a))

        self.assertTrue(np.allclose(bm.T.to_numpy(), a.T))
        self.assertTrue(np.allclose(BlockMatrix.from_numpy(a.T).to_numpy(), a.T))