import unittest

import hail as hl
from hail.linalg import BlockMatrix
from hail.utils import new_temp_file, FatalError
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
    def np_matrix(a):
        if isinstance(a, BlockMatrix):
            return np.matrix(a.to_numpy())
        else:
            return np.matrix(a)

    def test_from_entry_expr(self):
        mt = self.get_dataset()
        mt = mt.annotate_entries(x=hl.or_else(mt.GT.n_alt_alleles(), 0)).cache()

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

                self.assertTrue(np.array_equal(a1, a))
                self.assertTrue(np.array_equal(a2, a))
                self.assertTrue(np.array_equal(a3, a))
                self.assertTrue(np.array_equal(a4, a))
                self.assertTrue(np.array_equal(a5, a))

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

                self.assertTrue(np.array_equal(at1, at))
                self.assertTrue(np.array_equal(at2, at))
                self.assertTrue(np.array_equal(at3, at))
                self.assertTrue(np.array_equal(at4, at))
                self.assertTrue(np.array_equal(at5, at))

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
        def assert_eq(a, b):
            self.assertTrue(np.array_equal(self.np_matrix(a), self.np_matrix(b)))

        def assert_close(a, b):
            self.assertTrue(np.allclose(self.np_matrix(a), self.np_matrix(b)))

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

        assert_eq(+m, 0 + m)
        assert_eq(-m, 0 - m)

        # addition
        assert_eq(x + e, nx + e)
        assert_eq(c + e, nc + e)
        assert_eq(r + e, nr + e)
        assert_eq(m + e, nm + e)

        assert_eq(x + e, e + x)
        assert_eq(c + e, e + c)
        assert_eq(r + e, e + r)
        assert_eq(m + e, e + m)

        assert_eq(x + x, 2 * x)
        assert_eq(c + c, 2 * c)
        assert_eq(r + r, 2 * r)
        assert_eq(m + m, 2 * m)

        assert_eq(x + c, np.matrix([[3.0], [4.0]]))
        assert_eq(x + r, np.matrix([[3.0, 4.0, 5.0]]))
        assert_eq(x + m, np.matrix([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]))
        assert_eq(c + m, np.matrix([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]))
        assert_eq(r + m, np.matrix([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]]))
        assert_eq(x + c, c + x)
        assert_eq(x + r, r + x)
        assert_eq(x + m, m + x)
        assert_eq(c + m, m + c)
        assert_eq(r + m, m + r)

        assert_eq(x + nx, x + x)
        assert_eq(x + nc, x + c)
        assert_eq(x + nr, x + r)
        assert_eq(x + nm, x + m)
        assert_eq(c + nx, c + x)
        assert_eq(c + nc, c + c)
        assert_eq(c + nm, c + m)
        assert_eq(r + nx, r + x)
        assert_eq(r + nr, r + r)
        assert_eq(r + nm, r + m)
        assert_eq(m + nx, m + x)
        assert_eq(m + nc, m + c)
        assert_eq(m + nr, m + r)
        assert_eq(m + nm, m + m)

        # subtraction
        assert_eq(x - e, nx - e)
        assert_eq(c - e, nc - e)
        assert_eq(r - e, nr - e)
        assert_eq(m - e, nm - e)

        assert_eq(x - e, -(e - x))
        assert_eq(c - e, -(e - c))
        assert_eq(r - e, -(e - r))
        assert_eq(m - e, -(e - m))

        assert_eq(x - x, np.zeros((1, 1)))
        assert_eq(c - c, np.zeros((2, 1)))
        assert_eq(r - r, np.zeros((1, 3)))
        assert_eq(m - m, np.zeros((2, 3)))

        assert_eq(x - c, np.matrix([[1.0], [0.0]]))
        assert_eq(x - r, np.matrix([[1.0, 0.0, -1.0]]))
        assert_eq(x - m, np.matrix([[1.0, 0.0, -1.0], [-2.0, -3.0, -4.0]]))
        assert_eq(c - m, np.matrix([[0.0, -1.0, -2.0], [-2.0, -3.0, -4.0]]))
        assert_eq(r - m, np.matrix([[0.0, 0.0, 0.0], [-3.0, -3.0, -3.0]]))
        assert_eq(x - c, -(c - x))
        assert_eq(x - r, -(r - x))
        assert_eq(x - m, -(m - x))
        assert_eq(c - m, -(m - c))
        assert_eq(r - m, -(m - r))

        assert_eq(x - nx, x - x)
        assert_eq(x - nc, x - c)
        assert_eq(x - nr, x - r)
        assert_eq(x - nm, x - m)
        assert_eq(c - nx, c - x)
        assert_eq(c - nc, c - c)
        assert_eq(c - nm, c - m)
        assert_eq(r - nx, r - x)
        assert_eq(r - nr, r - r)
        assert_eq(r - nm, r - m)
        assert_eq(m - nx, m - x)
        assert_eq(m - nc, m - c)
        assert_eq(m - nr, m - r)
        assert_eq(m - nm, m - m)

        # multiplication
        assert_eq(x * e, nx * e)
        assert_eq(c * e, nc * e)
        assert_eq(r * e, nr * e)
        assert_eq(m * e, nm * e)

        assert_eq(x * e, e * x)
        assert_eq(c * e, e * c)
        assert_eq(r * e, e * r)
        assert_eq(m * e, e * m)

        assert_eq(x * x, x ** 2)
        assert_eq(c * c, c ** 2)
        assert_eq(r * r, r ** 2)
        assert_eq(m * m, m ** 2)

        assert_eq(x * c, np.matrix([[2.0], [4.0]]))
        assert_eq(x * r, np.matrix([[2.0, 4.0, 6.0]]))
        assert_eq(x * m, np.matrix([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]))
        assert_eq(c * m, np.matrix([[1.0, 2.0, 3.0], [8.0, 10.0, 12.0]]))
        assert_eq(r * m, np.matrix([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]))
        assert_eq(x * c, c * x)
        assert_eq(x * r, r * x)
        assert_eq(x * m, m * x)
        assert_eq(c * m, m * c)
        assert_eq(r * m, m * r)

        assert_eq(x * nx, x * x)
        assert_eq(x * nc, x * c)
        assert_eq(x * nr, x * r)
        assert_eq(x * nm, x * m)
        assert_eq(c * nx, c * x)
        assert_eq(c * nc, c * c)
        assert_eq(c * nm, c * m)
        assert_eq(r * nx, r * x)
        assert_eq(r * nr, r * r)
        assert_eq(r * nm, r * m)
        assert_eq(m * nx, m * x)
        assert_eq(m * nc, m * c)
        assert_eq(m * nr, m * r)
        assert_eq(m * nm, m * m)

        # division
        assert_close(x / e, nx / e)
        assert_close(c / e, nc / e)
        assert_close(r / e, nr / e)
        assert_close(m / e, nm / e)

        assert_close(x / e, 1 / (e / x))
        assert_close(c / e, 1 / (e / c))
        assert_close(r / e, 1 / (e / r))
        assert_close(m / e, 1 / (e / m))

        assert_close(x / x, np.ones((1, 1)))
        assert_close(c / c, np.ones((2, 1)))
        assert_close(r / r, np.ones((1, 3)))
        assert_close(m / m, np.ones((2, 3)))

        assert_close(x / c, np.matrix([[2 / 1.0], [2 / 2.0]]))
        assert_close(x / r, np.matrix([[2 / 1.0, 2 / 2.0, 2 / 3.0]]))
        assert_close(x / m, np.matrix([[2 / 1.0, 2 / 2.0, 2 / 3.0], [2 / 4.0, 2 / 5.0, 2 / 6.0]]))
        assert_close(c / m, np.matrix([[1 / 1.0, 1 / 2.0, 1 / 3.0], [2 / 4.0, 2 / 5.0, 2 / 6.0]]))
        assert_close(r / m, np.matrix([[1 / 1.0, 2 / 2.0, 3 / 3.0], [1 / 4.0, 2 / 5.0, 3 / 6.0]]))
        assert_close(x / c, 1 / (c / x))
        assert_close(x / r, 1 / (r / x))
        assert_close(x / m, 1 / (m / x))
        assert_close(c / m, 1 / (m / c))
        assert_close(r / m, 1 / (m / r))

        assert_close(x / nx, x / x)
        assert_close(x / nc, x / c)
        assert_close(x / nr, x / r)
        assert_close(x / nm, x / m)
        assert_close(c / nx, c / x)
        assert_close(c / nc, c / c)
        assert_close(c / nm, c / m)
        assert_close(r / nx, r / x)
        assert_close(r / nr, r / r)
        assert_close(r / nm, r / m)
        assert_close(m / nx, m / x)
        assert_close(m / nc, m / c)
        assert_close(m / nr, m / r)
        assert_close(m / nm, m / m)

        # exponentiation
        assert_close(m ** 3, m * m * m)

        # sqrt
        assert_close(m.sqrt(), m ** 0.5)

    def test_matrix_ops(self):
        def assert_eq(a, b):
            self.assertTrue(np.array_equal(self.np_matrix(a), self.np_matrix(b)))

        nm = np.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        m = BlockMatrix.from_numpy(nm)

        assert_eq(m.T, nm.T)
        assert_eq(m.T, nm.T)

        assert_eq(m @ m.T, nm @ nm.T)
        assert_eq(m @ nm.T, nm @ nm.T)

        assert_eq(m.T @ m, nm.T @ nm)
        assert_eq(m.T @ nm, nm.T @ nm)

        self.assertRaises(ValueError, lambda: m @ m)
        self.assertRaises(ValueError, lambda: m @ nm)

        assert_eq(m.diagonal(), np.array([1.0, 5.0]))
        assert_eq(m.T.diagonal(), np.array([1.0, 5.0]))
        assert_eq((m @ m.T).diagonal(), np.array([14.0, 77.0]))

    def test_slicing(self):
        nd = np.array(np.arange(0, 80, dtype=float)).reshape(8, 10)
        bm = BlockMatrix.from_numpy(nd, block_size=3)

        for indices in [(0, 0), (5, 7), (-3, 9), (-8, -10)]:
            self.assertTrue(np.array_equal(bm[indices], nd[indices]))

        for indices in [(0, slice(3, 4)),
                        (1, slice(3, 4)),
                        (-8, slice(3, 4)),
                        (-1, slice(3, 4))]:
            self.assertTrue(np.array_equal(bm[indices].to_numpy(), np.expand_dims(nd[indices], 0)))

        for indices in [(slice(3, 4), 0),
                        (slice(3, 4), 1),
                        (slice(3, 4), -8),
                        (slice(3, 4), -1)]:
            self.assertTrue(np.array_equal(bm[indices].to_numpy(), np.expand_dims(nd[indices], 1)))

        for indices in [(slice(0, 8), slice(0, 10)),
                        (slice(0, 8, 2), slice(0, 10, 2)),
                        (slice(2, 4), slice(5, 7)),
                        (slice(-8, -1), slice(-10, -1)),
                        (slice(-8, -1, 2), slice(-10, -1, 2)),
                        (slice(None, 4, 1), slice(None, 4, 1)),
                        (slice(4, None), slice(4, None)),
                        (slice(None, None), slice(None, None))]:
            self.assertTrue(np.array_equal(bm[indices].to_numpy(), nd[indices]))

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
        self.assertTrue(np.array_equal(bm2[:, :].to_numpy(), nd2))

        self.assertRaises(FatalError, lambda: bm2[:, 0])
        self.assertRaises(FatalError, lambda: bm2[0, :])
        self.assertRaises(FatalError, lambda: bm2[0:1, 0:1])

    def test_sparsify(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)

        self.assertTrue(np.array_equal(
            bm.sparsify_row_intervals(
                starts=[1, 0, 2, 2],
                stops= [2, 0, 3, 4]).to_numpy(),
            np.array([[ 0.,  2.,  0.,  0.],
                      [ 0.,  0.,  0.,  0.],
                      [ 0.,  0., 11.,  0.],
                      [ 0.,  0., 15., 16.]])))

        self.assertTrue(np.array_equal(
            bm.sparsify_row_intervals(
                starts=[1, 0, 2, 2],
                stops= [2, 0, 3, 4],
                blocks_only=True).to_numpy(),
            np.array([[ 1.,  2.,  0.,  0.],
                      [ 5.,  6.,  0.,  0.],
                      [ 0.,  0., 11., 12.],
                      [ 0.,  0., 15., 16.]])))
