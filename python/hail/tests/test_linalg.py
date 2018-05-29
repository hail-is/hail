import unittest

import hail as hl
from hail.linalg import BlockMatrix
from hail.utils import new_temp_file, new_local_temp_dir, local_path_uri, FatalError
from .utils import resource, startTestHailContext, stopTestHailContext
import numpy as np
import tempfile

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


class Tests(unittest.TestCase):
    _dataset = None

    @staticmethod
    def get_dataset():
        if Tests._dataset is None:
            Tests._dataset = hl.split_multi_hts(hl.import_vcf(resource('sample.vcf')))
        return Tests._dataset

    @staticmethod
    def _np_matrix(a):
        if isinstance(a, BlockMatrix):
            return np.matrix(a.to_numpy())
        else:
            return np.matrix(a)

    def _assert_eq(self, a, b):
        self.assertTrue(np.array_equal(self._np_matrix(a), self._np_matrix(b)))

    def _assert_close(self, a, b):
        self.assertTrue(np.allclose(self._np_matrix(a), self._np_matrix(b)))

    def test_from_entry_expr(self):
        mt = self.get_dataset()
        mt = mt.annotate_entries(x=hl.or_else(mt.GT.n_alt_alleles(), 0)).cache()

        a1 = BlockMatrix.from_entry_expr(hl.or_else(mt.GT.n_alt_alleles(), 0), block_size=32).to_numpy()
        a2 = BlockMatrix.from_entry_expr(mt.x, block_size=32).to_numpy()
        a3 = BlockMatrix.from_entry_expr(hl.float64(mt.x), block_size=32).to_numpy()

        self._assert_eq(a1, a2)
        self._assert_eq(a1, a3)

        path = new_temp_file()
        BlockMatrix.write_from_entry_expr(mt.x, path, block_size=32)
        a4 = BlockMatrix.read(path).to_numpy()
        self._assert_eq(a1, a4)

    def test_to_from_numpy(self):
        n_rows = 10
        n_cols = 11
        data = np.random.rand(n_rows * n_cols)

        bm = BlockMatrix._create(n_rows, n_cols, data.tolist(), row_major=True, block_size=4)
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

                self._assert_eq(a1, a)
                self._assert_eq(a2, a)
                self._assert_eq(a3, a)
                self._assert_eq(a4, a)
                self._assert_eq(a5, a)

        bmt = bm.T
        at = a.T

        with tempfile.NamedTemporaryFile() as bmt_f:
            with tempfile.NamedTemporaryFile() as at_f:
                bmt.tofile(bmt_f.name)
                at.tofile(at_f.name)

                at1 = bmt.to_numpy()
                at2 = BlockMatrix.from_numpy(at).to_numpy()
                at3 = np.fromfile(bmt_f.name).reshape((n_cols, n_rows))
                at4 = BlockMatrix.fromfile(at_f.name, n_cols, n_rows).to_numpy()
                at5 = BlockMatrix.fromfile(bmt_f.name, n_cols, n_rows).to_numpy()

                self._assert_eq(at1, at)
                self._assert_eq(at2, at)
                self._assert_eq(at3, at)
                self._assert_eq(at4, at)
                self._assert_eq(at5, at)

    def test_promote(self):
        nx = np.matrix([[2.0]])
        nc = np.matrix([[1.0], [2.0]])
        nr = np.matrix([[1.0, 2.0, 3.0]])
        nm = np.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        e = 2
        x = BlockMatrix.from_numpy(nx)
        c = BlockMatrix.from_numpy(nc)
        r = BlockMatrix.from_numpy(nr)
        m = BlockMatrix.from_numpy(nm)

        nct, nrt, nmt = nc.T, nr.T, nm.T
        ct, rt, mt = c.T, r.T, m.T

        good = [(x, x),  (x, c),  (x, r),  (x, m), (x, e),
                (c, x),  (c, c),           (c, m), (c, e),
                (r, x),           (r, r),  (r, m), (r, e),
                (m, x),  (m, c),  (m, r),  (m, m), (m, e),
                (x, nx), (x, nc), (x, nr), (x, nm),
                (c, nx), (c, nc),          (c, nm),
                (r, nx),          (r, nr), (r, nm),
                (m, nx), (m, nc), (m, nr), (m, nm)]

        bad = [(c, r), (r, c), (c, ct), (r, rt),
               (c, rt), (c, mt), (ct, r), (ct, m),
               (r, ct), (r, mt), (rt, c), (rt, m),
               (m, ct), (m, rt), (m, mt), (mt, c), (mt, r), (mt, m),
               (c, nr), (r, nc), (c, nct), (r, nrt),
               (c, nrt), (c, nmt), (ct, nr), (ct, nm),
               (r, nct), (r, nmt), (rt, nc), (rt, nm),
               (m, nct), (m, nrt), (m, nmt), (mt, nc), (mt, nr), (mt, nm)]

        for (a, b) in good:
            a._promote(b, '')

        for (a, b) in bad:
            self.assertRaises(ValueError,
                              lambda: a._promote(b, ''))

    def test_elementwise_ops(self):
        nx = np.matrix([[2.0]])
        nc = np.matrix([[1.0], [2.0]])
        nr = np.matrix([[1.0, 2.0, 3.0]])
        nm = np.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        e = 2.0
        x = BlockMatrix.from_numpy(nx)
        c = BlockMatrix.from_numpy(nc)
        r = BlockMatrix.from_numpy(nr)
        m = BlockMatrix.from_numpy(nm)

        self.assertRaises(TypeError,
                          lambda: x + np.array(['one'], dtype=str))

        self._assert_eq(+m, 0 + m)
        self._assert_eq(-m, 0 - m)

        # addition
        self._assert_eq(x + e, nx + e)
        self._assert_eq(c + e, nc + e)
        self._assert_eq(r + e, nr + e)
        self._assert_eq(m + e, nm + e)

        self._assert_eq(x + e, e + x)
        self._assert_eq(c + e, e + c)
        self._assert_eq(r + e, e + r)
        self._assert_eq(m + e, e + m)

        self._assert_eq(x + x, 2 * x)
        self._assert_eq(c + c, 2 * c)
        self._assert_eq(r + r, 2 * r)
        self._assert_eq(m + m, 2 * m)

        self._assert_eq(x + c, np.matrix([[3.0], [4.0]]))
        self._assert_eq(x + r, np.matrix([[3.0, 4.0, 5.0]]))
        self._assert_eq(x + m, np.matrix([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]))
        self._assert_eq(c + m, np.matrix([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]))
        self._assert_eq(r + m, np.matrix([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]]))
        self._assert_eq(x + c, c + x)
        self._assert_eq(x + r, r + x)
        self._assert_eq(x + m, m + x)
        self._assert_eq(c + m, m + c)
        self._assert_eq(r + m, m + r)

        self._assert_eq(x + nx, x + x)
        self._assert_eq(x + nc, x + c)
        self._assert_eq(x + nr, x + r)
        self._assert_eq(x + nm, x + m)
        self._assert_eq(c + nx, c + x)
        self._assert_eq(c + nc, c + c)
        self._assert_eq(c + nm, c + m)
        self._assert_eq(r + nx, r + x)
        self._assert_eq(r + nr, r + r)
        self._assert_eq(r + nm, r + m)
        self._assert_eq(m + nx, m + x)
        self._assert_eq(m + nc, m + c)
        self._assert_eq(m + nr, m + r)
        self._assert_eq(m + nm, m + m)

        # subtraction
        self._assert_eq(x - e, nx - e)
        self._assert_eq(c - e, nc - e)
        self._assert_eq(r - e, nr - e)
        self._assert_eq(m - e, nm - e)

        self._assert_eq(x - e, -(e - x))
        self._assert_eq(c - e, -(e - c))
        self._assert_eq(r - e, -(e - r))
        self._assert_eq(m - e, -(e - m))

        self._assert_eq(x - x, np.zeros((1, 1)))
        self._assert_eq(c - c, np.zeros((2, 1)))
        self._assert_eq(r - r, np.zeros((1, 3)))
        self._assert_eq(m - m, np.zeros((2, 3)))

        self._assert_eq(x - c, np.matrix([[1.0], [0.0]]))
        self._assert_eq(x - r, np.matrix([[1.0, 0.0, -1.0]]))
        self._assert_eq(x - m, np.matrix([[1.0, 0.0, -1.0], [-2.0, -3.0, -4.0]]))
        self._assert_eq(c - m, np.matrix([[0.0, -1.0, -2.0], [-2.0, -3.0, -4.0]]))
        self._assert_eq(r - m, np.matrix([[0.0, 0.0, 0.0], [-3.0, -3.0, -3.0]]))
        self._assert_eq(x - c, -(c - x))
        self._assert_eq(x - r, -(r - x))
        self._assert_eq(x - m, -(m - x))
        self._assert_eq(c - m, -(m - c))
        self._assert_eq(r - m, -(m - r))

        self._assert_eq(x - nx, x - x)
        self._assert_eq(x - nc, x - c)
        self._assert_eq(x - nr, x - r)
        self._assert_eq(x - nm, x - m)
        self._assert_eq(c - nx, c - x)
        self._assert_eq(c - nc, c - c)
        self._assert_eq(c - nm, c - m)
        self._assert_eq(r - nx, r - x)
        self._assert_eq(r - nr, r - r)
        self._assert_eq(r - nm, r - m)
        self._assert_eq(m - nx, m - x)
        self._assert_eq(m - nc, m - c)
        self._assert_eq(m - nr, m - r)
        self._assert_eq(m - nm, m - m)

        # multiplication
        self._assert_eq(x * e, nx * e)
        self._assert_eq(c * e, nc * e)
        self._assert_eq(r * e, nr * e)
        self._assert_eq(m * e, nm * e)

        self._assert_eq(x * e, e * x)
        self._assert_eq(c * e, e * c)
        self._assert_eq(r * e, e * r)
        self._assert_eq(m * e, e * m)

        self._assert_eq(x * x, x ** 2)
        self._assert_eq(c * c, c ** 2)
        self._assert_eq(r * r, r ** 2)
        self._assert_eq(m * m, m ** 2)

        self._assert_eq(x * c, np.matrix([[2.0], [4.0]]))
        self._assert_eq(x * r, np.matrix([[2.0, 4.0, 6.0]]))
        self._assert_eq(x * m, np.matrix([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]))
        self._assert_eq(c * m, np.matrix([[1.0, 2.0, 3.0], [8.0, 10.0, 12.0]]))
        self._assert_eq(r * m, np.matrix([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]))
        self._assert_eq(x * c, c * x)
        self._assert_eq(x * r, r * x)
        self._assert_eq(x * m, m * x)
        self._assert_eq(c * m, m * c)
        self._assert_eq(r * m, m * r)

        self._assert_eq(x * nx, x * x)
        self._assert_eq(x * nc, x * c)
        self._assert_eq(x * nr, x * r)
        self._assert_eq(x * nm, x * m)
        self._assert_eq(c * nx, c * x)
        self._assert_eq(c * nc, c * c)
        self._assert_eq(c * nm, c * m)
        self._assert_eq(r * nx, r * x)
        self._assert_eq(r * nr, r * r)
        self._assert_eq(r * nm, r * m)
        self._assert_eq(m * nx, m * x)
        self._assert_eq(m * nc, m * c)
        self._assert_eq(m * nr, m * r)
        self._assert_eq(m * nm, m * m)

        # division
        self._assert_close(x / e, nx / e)
        self._assert_close(c / e, nc / e)
        self._assert_close(r / e, nr / e)
        self._assert_close(m / e, nm / e)

        self._assert_close(x / e, 1 / (e / x))
        self._assert_close(c / e, 1 / (e / c))
        self._assert_close(r / e, 1 / (e / r))
        self._assert_close(m / e, 1 / (e / m))

        self._assert_close(x / x, np.ones((1, 1)))
        self._assert_close(c / c, np.ones((2, 1)))
        self._assert_close(r / r, np.ones((1, 3)))
        self._assert_close(m / m, np.ones((2, 3)))

        self._assert_close(x / c, np.matrix([[2 / 1.0], [2 / 2.0]]))
        self._assert_close(x / r, np.matrix([[2 / 1.0, 2 / 2.0, 2 / 3.0]]))
        self._assert_close(x / m, np.matrix([[2 / 1.0, 2 / 2.0, 2 / 3.0], [2 / 4.0, 2 / 5.0, 2 / 6.0]]))
        self._assert_close(c / m, np.matrix([[1 / 1.0, 1 / 2.0, 1 / 3.0], [2 / 4.0, 2 / 5.0, 2 / 6.0]]))
        self._assert_close(r / m, np.matrix([[1 / 1.0, 2 / 2.0, 3 / 3.0], [1 / 4.0, 2 / 5.0, 3 / 6.0]]))
        self._assert_close(x / c, 1 / (c / x))
        self._assert_close(x / r, 1 / (r / x))
        self._assert_close(x / m, 1 / (m / x))
        self._assert_close(c / m, 1 / (m / c))
        self._assert_close(r / m, 1 / (m / r))

        self._assert_close(x / nx, x / x)
        self._assert_close(x / nc, x / c)
        self._assert_close(x / nr, x / r)
        self._assert_close(x / nm, x / m)
        self._assert_close(c / nx, c / x)
        self._assert_close(c / nc, c / c)
        self._assert_close(c / nm, c / m)
        self._assert_close(r / nx, r / x)
        self._assert_close(r / nr, r / r)
        self._assert_close(r / nm, r / m)
        self._assert_close(m / nx, m / x)
        self._assert_close(m / nc, m / c)
        self._assert_close(m / nr, m / r)
        self._assert_close(m / nm, m / m)

    def test_special_elementwise_ops(self):
        nm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = BlockMatrix.from_numpy(nm)

        self._assert_close(m ** 3, nm ** 3)

        self._assert_close(m.sqrt(), np.sqrt(nm))

        self._assert_close(m.log(), np.log(nm))

        self._assert_close((m - 4).abs(), np.abs(nm - 4))

    def test_matrix_ops(self):
        nm = np.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = BlockMatrix.from_numpy(nm, block_size=2)

        self._assert_eq(m.T, nm.T)
        self._assert_eq(m.T, nm.T)

        self._assert_eq(m @ m.T, nm @ nm.T)
        self._assert_eq(m @ nm.T, nm @ nm.T)

        self._assert_eq(m.T @ m, nm.T @ nm)
        self._assert_eq(m.T @ nm, nm.T @ nm)

        self.assertRaises(ValueError, lambda: m @ m)
        self.assertRaises(ValueError, lambda: m @ nm)

        self._assert_eq(m.diagonal(), np.array([1.0, 5.0]))
        self._assert_eq(m.T.diagonal(), np.array([1.0, 5.0]))
        self._assert_eq((m @ m.T).diagonal(), np.array([14.0, 77.0]))

    def test_fill(self):
        nd = np.ones((3, 5))
        bm = BlockMatrix.fill(3, 5, 1.0)
        bm2 = BlockMatrix.fill(3, 5, 1.0, block_size=2)

        self.assertTrue(bm.block_size == BlockMatrix.default_block_size())
        self.assertTrue(bm2.block_size == 2)
        self._assert_eq(bm, nd)
        self._assert_eq(bm2, nd)

    def test_sum(self):
        def sums_agree(bm, nd):
            self.assertAlmostEqual(bm.sum(), np.sum(nd))
            self._assert_close(bm.sum(axis=0), np.sum(nd, axis=0, keepdims=True))
            self._assert_close(bm.sum(axis=1), np.sum(nd, axis=1, keepdims=True))

        nd = np.random.normal(size=(11, 13))
        bm = BlockMatrix.from_numpy(nd, block_size=3)

        nd2 = np.zeros(shape=(5, 7))
        nd2[2, 4] = 1.0
        nd2[2, 5] = 2.0
        nd2[3, 4] = 3.0
        nd2[3, 5] = 4.0
        bm2 = BlockMatrix.from_numpy(nd2, block_size=2).sparsify_rectangles([[2, 4, 4, 6]])

        nd3 = np.zeros(shape=(5, 7))
        bm3 = BlockMatrix.fill(5, 7, value=0.0, block_size=2).sparsify_rectangles([])

        sums_agree(bm, nd)
        sums_agree(bm2, nd2)
        sums_agree(bm3, nd3)

    def test_slicing(self):
        nd = np.array(np.arange(0, 80, dtype=float)).reshape(8, 10)
        bm = BlockMatrix.from_numpy(nd, block_size=3)

        for indices in [(0, 0), (5, 7), (-3, 9), (-8, -10)]:
            self._assert_eq(bm[indices], nd[indices])

        for indices in [(0, slice(3, 4)),
                        (1, slice(3, 4)),
                        (-8, slice(3, 4)),
                        (-1, slice(3, 4))]:
            self._assert_eq(bm[indices], np.expand_dims(nd[indices], 0))

        for indices in [(slice(3, 4), 0),
                        (slice(3, 4), 1),
                        (slice(3, 4), -8),
                        (slice(3, 4), -1)]:
            self._assert_eq(bm[indices], np.expand_dims(nd[indices], 1))

        for indices in [(slice(0, 8), slice(0, 10)),
                        (slice(0, 8, 2), slice(0, 10, 2)),
                        (slice(2, 4), slice(5, 7)),
                        (slice(-8, -1), slice(-10, -1)),
                        (slice(-8, -1, 2), slice(-10, -1, 2)),
                        (slice(None, 4, 1), slice(None, 4, 1)),
                        (slice(4, None), slice(4, None)),
                        (slice(None, None), slice(None, None))]:
            self._assert_eq(bm[indices], nd[indices])

        self.assertRaises(ValueError, lambda: bm[0, ])

        self.assertRaises(ValueError, lambda: bm[9, 0])
        self.assertRaises(ValueError, lambda: bm[-9, 0])
        self.assertRaises(ValueError, lambda: bm[0, 11])
        self.assertRaises(ValueError, lambda: bm[0, -11])

        self.assertRaises(ValueError, lambda: bm[::-1, 0])
        self.assertRaises(ValueError, lambda: bm[0, ::-1])

        self.assertRaises(ValueError, lambda: bm[:0, 0])
        self.assertRaises(ValueError, lambda: bm[0, :0])

        self.assertRaises(ValueError, lambda: bm[0:9, 0])
        self.assertRaises(ValueError, lambda: bm[-9:, 0])
        self.assertRaises(ValueError, lambda: bm[:-9, 0])

        self.assertRaises(ValueError, lambda: bm[0, :11])
        self.assertRaises(ValueError, lambda: bm[0, -11:])
        self.assertRaises(ValueError, lambda: bm[0, :-11])

        bm2 = bm.sparsify_row_intervals([0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(bm2[0, 1], 1.0)
        self.assertEqual(bm2[0, 2], 0.0)
        self.assertEqual(bm2[0, 9], 0.0)

        nd2 = np.zeros(shape=(8, 10))
        nd2[0, 1] = 1.0
        self._assert_eq(bm2[:, :], nd2)

        self._assert_eq(bm2[:, 1], nd2[:, 1:2])
        self._assert_eq(bm2[1, :], nd2[1:2, :])
        self._assert_eq(bm2[0:5, 0:5], nd2[0:5, 0:5])

    def test_sparsify_row_intervals(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)

        self._assert_eq(
            bm.sparsify_row_intervals(
                starts=[1, 0, 2, 2],
                stops= [2, 0, 3, 4]),
            np.array([[ 0.,  2.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.],
                      [ 0.,  0., 11.,  0.],
                      [ 0.,  0., 15., 16.]]))

        self._assert_eq(
            bm.sparsify_row_intervals(
                starts=[1, 0, 2, 2],
                stops= [2, 0, 3, 4],
                blocks_only=True),
            np.array([[ 1.,  2.,  0.,  0.],
                      [ 5.,  6.,  0.,  0.],
                      [ 0.,  0., 11., 12.],
                      [ 0.,  0., 15., 16.]]))

        nd2 = np.random.normal(size=(8, 10))
        bm2 = BlockMatrix.from_numpy(nd2, block_size=3)

        for bounds in [[[0, 1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 3, 4, 5, 6, 7, 8]],
                       [[0, 0, 5, 3, 4, 5, 8, 2],
                        [9, 0, 5, 3, 4, 5, 9, 5]],
                       [[0, 5, 10, 8, 7, 6, 5, 4],
                        [0, 5, 10, 9, 8, 7, 6, 5]]]:
            starts, stops = bounds
            actual = bm2.sparsify_row_intervals(starts, stops, blocks_only=False).to_numpy()
            expected = nd2.copy()
            for i in range(0, 8):
                for j in range(0, starts[i]):
                    expected[i, j] = 0.0
                for j in range(stops[i], 10):
                    expected[i, j] = 0.0
            self._assert_eq(actual, expected)

    def test_sparsify_band(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)

        self._assert_eq(
            bm.sparsify_band(lower=-1, upper=2),
            np.array([[ 1.,  2.,  3.,  0.],
                      [ 5.,  6.,  7.,  8.],
                      [ 0., 10., 11., 12.],
                      [ 0.,  0., 15., 16.]]))

        self._assert_eq(
            bm.sparsify_band(lower=0, upper=0, blocks_only=True),
            np.array([[ 1.,  2.,  0.,  0.],
                      [ 5.,  6.,  0.,  0.],
                      [ 0.,  0., 11., 12.],
                      [ 0.,  0., 15., 16.]]))

        nd2 = np.arange(0, 80, dtype=float).reshape(8, 10)
        bm2 = BlockMatrix.from_numpy(nd2, block_size=3)

        for bounds in [[0, 0], [1, 1], [2, 2], [-5, 5], [-7, 0], [0, 9], [-100, 100]]:
            lower, upper = bounds
            actual = bm2.sparsify_band(lower, upper, blocks_only=False).to_numpy()
            mask = np.fromfunction(lambda i, j: (lower <= j - i) * (j - i <= upper), (8, 10))
            self._assert_eq(actual, nd2 * mask)

    def test_sparsify_triangle(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)

        self.assertFalse(bm.is_sparse)
        self.assertTrue(bm.sparsify_triangle().is_sparse)

        self._assert_eq(
            bm.sparsify_triangle(),
            np.array([[ 1.,  2.,  3.,  4.],
                      [ 0.,  6.,  7.,  8.],
                      [ 0.,  0., 11., 12.],
                      [ 0.,  0.,  0., 16.]]))

        self._assert_eq(
            bm.sparsify_triangle(lower=True),
            np.array([[ 1.,  0.,  0.,  0.],
                      [ 5.,  6.,  0.,  0.],
                      [ 9., 10., 11.,  0.],
                      [13., 14., 15., 16.]]))

        self._assert_eq(
            bm.sparsify_triangle(blocks_only=True),
            np.array([[ 1.,  2.,  3.,  4.],
                      [ 5.,  6.,  7.,  8.],
                      [ 0.,  0., 11., 12.],
                      [ 0.,  0., 15., 16.]]))

    def test_sparsify_rectangles(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)

        self._assert_eq(
            bm.sparsify_rectangles([[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4]]),
            np.array([[ 1.,  2.,  3.,  4.],
                      [ 5.,  6.,  7.,  8.],
                      [ 9., 10.,  0.,  0.],
                      [13., 14.,  0.,  0.]]))

        self._assert_eq(bm.sparsify_rectangles([]), np.zeros(shape=(4, 4)))
                        
    def test_export_rectangles(self):
        nd = np.arange(0, 80, dtype=float).reshape(8, 10)

        rects1 = [[0, 1, 0, 1], [4, 5, 7, 8]]

        rects2 = [[4, 5, 0, 10], [0, 8, 4, 5]]

        rects3 = [[0, 1, 0, 1], [1, 2, 1, 2], [2, 3, 2, 3],
                  [3, 5, 3, 6], [3, 6, 3, 7], [3, 7, 3, 8],
                  [4, 5, 0, 10], [0, 8, 4, 5], [0, 8, 0, 10]]

        for rects in [rects1, rects2, rects3]:
            for block_size in [3, 4, 10]:
                bm_uri = new_temp_file()
                rect_path = new_local_temp_dir()
                rect_uri = local_path_uri(rect_path)

                (BlockMatrix.from_numpy(nd, block_size=block_size)
                    .sparsify_rectangles(rects)
                    .write(bm_uri, force_row_major=True))

                BlockMatrix.export_rectangles(bm_uri, rect_uri, rects)

                for (i, r) in enumerate(rects):
                    file = rect_path + '/rect-' + str(i) + '_' + '-'.join(map(str, r))
                    expected = nd[r[0]:r[1], r[2]:r[3]]
                    actual = np.reshape(np.loadtxt(file), (r[1] - r[0], r[3] - r[2]))
                    self._assert_eq(expected, actual)

        bm_uri = new_temp_file()
        rect_uri = new_temp_file()

        (BlockMatrix.from_numpy(nd, block_size=5)
            .sparsify_rectangles([[0, 1, 0, 1]])
            .write(bm_uri, force_row_major=True))

        with self.assertRaises(FatalError) as e:
            BlockMatrix.export_rectangles(bm_uri, rect_uri, [[5, 6, 5, 6]])
            self.assertEquals(e.msg, 'block (1, 1) missing for rectangle 0 with bounds [5, 6, 5, 6]')
