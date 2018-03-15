import unittest

import hail as hl
from hail.linalg import BlockMatrix
from hail.utils import new_temp_file
from .utils import resource, startTestHailContext, stopTestHailContext
import numpy as np
import tempfile

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

        self.assertTrue(np.array_equal(a1, a2))
        self.assertTrue(np.array_equal(a1, a3))

        path = new_temp_file()
        BlockMatrix.write_from_entry_expr(mt.x, path, block_size=32)
        a4 = BlockMatrix.read(path).to_numpy()
        self.assertTrue(np.array_equal(a1, a4))

    def test_to_from_numpy(self):
        n_rows = 10
        n_cols = 11
        data = np.random.rand(n_rows * n_cols)

        bm = BlockMatrix._create_block_matrix(n_rows, n_cols, data.tolist(), row_major=True, block_size=4)
        a = data.reshape((n_rows, n_cols))

        with tempfile.NamedTemporaryFile() as bm_f:
            with tempfile.NamedTemporaryFile() as a_f:
                bm.tofile(bm_f.name)
                a.tofile(a_f.name)

                a1 = bm.to_numpy()
                a2 = BlockMatrix.from_numpy(a, block_size=5).to_numpy()
                a3 = np.fromfile(bm_f.name).reshape((n_rows, n_cols))
                a4 = BlockMatrix.fromfile(a_f.name, n_rows, n_cols, block_size=3).to_numpy()
                a5 = BlockMatrix.fromfile(bm_f.name, n_rows, n_cols).to_numpy()

                self.assertTrue(np.array_equal(a1, a))
                self.assertTrue(np.array_equal(a2, a))
                self.assertTrue(np.array_equal(a3, a))
                self.assertTrue(np.array_equal(a4, a))
                self.assertTrue(np.array_equal(a5, a))

        bmT = bm.T
        aT = a.T

        with tempfile.NamedTemporaryFile() as bmT_f:
            with tempfile.NamedTemporaryFile() as aT_f:
                bmT.tofile(bmT_f.name)
                aT.tofile(aT_f.name)

                aT1 = bmT.to_numpy()
                aT2 = BlockMatrix.from_numpy(aT).to_numpy()
                aT3 = np.fromfile(bmT_f.name).reshape((n_cols, n_rows))
                aT4 = BlockMatrix.fromfile(aT_f.name, n_cols, n_rows).to_numpy()
                aT5 = BlockMatrix.fromfile(bmT_f.name, n_cols, n_rows).to_numpy()

                self.assertTrue(np.array_equal(aT1, aT))
                self.assertTrue(np.array_equal(aT2, aT))
                self.assertTrue(np.array_equal(aT3, aT))
                self.assertTrue(np.array_equal(aT4, aT))
                self.assertTrue(np.array_equal(aT5, aT))