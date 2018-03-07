import unittest

import hail as hl
from hail.linalg import BlockMatrix
from .utils import resource
import numpy as np

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