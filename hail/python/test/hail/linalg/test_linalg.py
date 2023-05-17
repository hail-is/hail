import traceback

import pytest

import hail as hl
from hail.linalg import BlockMatrix
from hail.utils import local_path_uri, FatalError, HailUserError
from ..helpers import *
import numpy as np
import math
from hail.expr.expressions import ExpressionException


def sparsify_numpy(np_mat, block_size, blocks_to_sparsify):
    n_rows, n_cols = np_mat.shape
    target_mat = np.zeros((n_rows, n_cols))
    n_block_rows = math.ceil(n_rows / block_size)
    n_block_cols = math.ceil(n_cols / block_size)
    n_rows_last_block = block_size if (n_rows % block_size) == 0 else n_rows % block_size
    n_cols_last_block = block_size if (n_cols % block_size) == 0 else n_cols % block_size

    for block in blocks_to_sparsify:
        block_row_idx = block % n_block_rows
        block_col_idx = block // n_block_rows
        rows_to_copy = block_size if block_row_idx != (n_block_rows - 1) else n_rows_last_block
        cols_to_copy = block_size if block_col_idx != (n_block_cols - 1) else n_cols_last_block
        starting_row_idx = block_row_idx * block_size
        starting_col_idx = block_col_idx * block_size

        a = starting_row_idx
        b = starting_row_idx + rows_to_copy
        c = starting_col_idx
        d = starting_col_idx + cols_to_copy
        target_mat[a:b, c:d] = np_mat[a:b, c:d]

    return target_mat


class BatchedAsserts():

    def __init__(self, batch_size=32):
        self._batch_size = batch_size

    def __enter__(self):
        self._a_list = []
        self._b_list = []
        self._comparison_funcs = []
        self._tbs = []
        return self

    def _assert_agree(self, a, b, f):
        self._a_list.append(a)
        self._b_list.append(b)
        self._comparison_funcs.append(f)
        self._tbs.append(traceback.format_stack())

    def assert_eq(self, a, b):
        self._assert_agree(a, b, np.testing.assert_equal)

    def assert_close(self, a, b):
        self._assert_agree(a, b, np.testing.assert_allclose)

    def __exit__(self, *exc):
        a_list = self._a_list
        b_list = self._b_list
        comparisons = self._comparison_funcs
        tb = self._tbs
        assert len(a_list) == len(b_list)
        assert len(a_list) == len(comparisons)
        assert len(a_list) == len(tb)

        all_bms = {}

        a_results = []
        for i, a in enumerate(a_list):
            if isinstance(a, BlockMatrix):
                all_bms[(0, i)] = a.to_ndarray()
                a_results.append(None)
            else:
                a_results.append(np.array(a))

        b_results = []
        for i, b in enumerate(b_list):
            if isinstance(b, BlockMatrix):
                all_bms[(1, i)] = b.to_ndarray()
                b_results.append(None)
            else:
                b_results.append(np.array(b))

        bm_keys = list(all_bms.keys())

        vals = []
        batch_size = self._batch_size
        n_bms = len(bm_keys)
        for batch_start in range(0, n_bms, batch_size):
            vals.extend(list(hl.eval(tuple([all_bms[k] for k in bm_keys[batch_start:batch_start + batch_size]]))))

        for (a_or_b, idx), v in zip(bm_keys, vals):
            if a_or_b == 0:
                a_results[idx] = v
            else:
                b_results[idx] = v

        for i, x in enumerate(a_results):
            assert x is not None, i
        for i, x in enumerate(b_results):
            assert x is not None, i

        for a_res, b_res, comp_func, tb in zip(a_results, b_results, comparisons, tb):
            try:
                comp_func(a_res, b_res)
            except AssertionError as e:
                i = 0
                while i < len(tb):
                    if 'test/hail' in tb[i]:
                        break
                    i += 1
                raise AssertionError(
                    f'test failure:\n  left={a_res}\n right={b_res}\n f={comp_func.__name__}\n  failure at:\n{"".join(x for x in tb[i:])}') from e


class Tests(unittest.TestCase):
    @staticmethod
    def _np_matrix(a):
        if isinstance(a, BlockMatrix):
            return hl.eval(a.to_ndarray())
        else:
            return np.array(a)

    def _assert_eq(self, a, b):
        anp = self._np_matrix(a)
        bnp = self._np_matrix(b)
        np.testing.assert_equal(anp, bnp)

    def _assert_close(self, a, b):
        self.assertTrue(np.allclose(self._np_matrix(a), self._np_matrix(b)))

    def _assert_rectangles_eq(self, expected, rect_path, export_rects, binary=False):
        for (i, r) in enumerate(export_rects):
            piece_path = rect_path + '/rect-' + str(i) + '_' + '-'.join(map(str, r))
            with hl.current_backend().fs.open(piece_path, mode='rb' if binary else 'r') as file:
                expected_rect = expected[r[0]:r[1], r[2]:r[3]]
                if binary:
                    actual_rect = np.reshape(
                        np.frombuffer(file.read()),
                        (r[1] - r[0], r[3] - r[2]))
                else:
                    actual_rect = np.loadtxt(file, ndmin=2)
                self._assert_eq(expected_rect, actual_rect)

    def assert_sums_agree(self, bm, nd):
        self.assertAlmostEqual(bm.sum(), np.sum(nd))
        self._assert_close(bm.sum(axis=0), np.sum(nd, axis=0, keepdims=True))
        self._assert_close(bm.sum(axis=1), np.sum(nd, axis=1, keepdims=True))

    def test_from_entry_expr_simple(self):
        mt = get_dataset()
        mt = mt.annotate_entries(x=hl.or_else(mt.GT.n_alt_alleles(), 0)).cache()

        a1 = hl.eval(BlockMatrix.from_entry_expr(hl.or_else(mt.GT.n_alt_alleles(), 0), block_size=32).to_ndarray())
        a2 = hl.eval(BlockMatrix.from_entry_expr(mt.x, block_size=32).to_ndarray())
        a3 = hl.eval(BlockMatrix.from_entry_expr(hl.float64(mt.x), block_size=32).to_ndarray())

        self._assert_eq(a1, a2)
        self._assert_eq(a1, a3)

        with hl.TemporaryDirectory(ensure_exists=False) as path:
            BlockMatrix.write_from_entry_expr(mt.x, path, block_size=32)
            a4 = hl.eval(BlockMatrix.read(path).to_ndarray())
            self._assert_eq(a1, a4)

    def test_from_entry_expr_empty_parts(self):
        with hl.TemporaryDirectory(ensure_exists=False) as path:
            mt = hl.balding_nichols_model(n_populations=5, n_variants=2000, n_samples=20, n_partitions=200)
            mt = mt.filter_rows((mt.locus.position <= 500) | (mt.locus.position > 1500)).checkpoint(path)
            bm = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles())
            nd = (bm @ bm.T).to_numpy()
            assert nd.shape == (1000, 1000)

    def test_from_entry_expr_options(self):
        def build_mt(a):
            data = [{'v': 0, 's': 0, 'x': a[0]},
                    {'v': 0, 's': 1, 'x': a[1]},
                    {'v': 0, 's': 2, 'x': a[2]}]
            ht = hl.Table.parallelize(data, hl.dtype('struct{v: int32, s: int32, x: float64}'))
            mt = ht.to_matrix_table(['v'], ['s'])
            ids = mt.key_cols_by()['s'].collect()
            return mt.choose_cols([ids.index(0), ids.index(1), ids.index(2)])

        def check(expr, mean_impute, center, normalize, expected):
            actual = np.squeeze(hl.eval(BlockMatrix.from_entry_expr(expr,
                                                            mean_impute=mean_impute,
                                                            center=center,
                                                            normalize=normalize).to_ndarray()))
            assert np.allclose(actual, expected)

        a = np.array([0.0, 1.0, 2.0])

        mt = build_mt(a)
        check(mt.x, False, False, False, a)
        check(mt.x, False, True, False, a - 1.0)
        check(mt.x, False, False, True, a / np.sqrt(5))
        check(mt.x, False, True, True, (a - 1.0) / np.sqrt(2))
        check(mt.x + 1 - 1, False, False, False, a)

        mt = build_mt([0.0, hl.missing('float64'), 2.0])
        check(mt.x, True, False, False, a)
        check(mt.x, True, True, False, a - 1.0)
        check(mt.x, True, False, True, a / np.sqrt(5))
        check(mt.x, True, True, True, (a - 1.0) / np.sqrt(2))
        with self.assertRaises(Exception):
            BlockMatrix.from_entry_expr(mt.x)

    def test_write_from_entry_expr_overwrite(self):
        mt = hl.balding_nichols_model(1, 1, 1)
        mt = mt.select_entries(x=mt.GT.n_alt_alleles())
        bm = BlockMatrix.from_entry_expr(mt.x)

        with hl.TemporaryDirectory(ensure_exists=False) as path:
            BlockMatrix.write_from_entry_expr(mt.x, path)
            self.assertRaises(FatalError, lambda: BlockMatrix.write_from_entry_expr(mt.x, path))

            BlockMatrix.write_from_entry_expr(mt.x, path, overwrite=True)
            self._assert_eq(BlockMatrix.read(path), bm)

        with hl.TemporaryDirectory(ensure_exists=False) as path:
            # non-field expressions currently take a separate code path
            BlockMatrix.write_from_entry_expr(mt.x + 1, path)
            self.assertRaises(FatalError, lambda: BlockMatrix.write_from_entry_expr(mt.x + 1, path))

            BlockMatrix.write_from_entry_expr(mt.x + 2, path, overwrite=True)
            self._assert_eq(BlockMatrix.read(path), bm + 2)

    def test_random_uniform(self):
        uniform = BlockMatrix.random(10, 10, gaussian=False)

        nuniform = hl.eval(uniform.to_ndarray())
        for row in nuniform:
            for entry in row:
                assert entry > 0

    def test_bm_to_numpy(self):
        bm = BlockMatrix.from_ndarray(hl.nd.arange(20).map(lambda x: hl.float64(x)).reshape((4, 5)))
        np_bm = bm.to_numpy()
        self._assert_eq(np_bm, np.arange(20, dtype=np.float64).reshape((4, 5)))

    def test_numpy_round_trip(self):
        n_rows = 10
        n_cols = 11
        data = np.random.rand(n_rows * n_cols)

        bm = BlockMatrix._create(n_rows, n_cols, data.tolist(), block_size=4)
        a = data.reshape((n_rows, n_cols))

        with hl.TemporaryFilename() as bm_f, hl.TemporaryFilename() as a_f:
            bm.tofile(bm_f)
            hl.current_backend().fs.open(a_f, mode='wb').write(a.tobytes())

            a1 = bm.to_numpy()
            a2 = BlockMatrix.from_numpy(a, block_size=5).to_numpy()
            a3 = np.frombuffer(
                hl.current_backend().fs.open(bm_f, mode='rb').read()
            ).reshape((n_rows, n_cols))
            a4 = BlockMatrix.fromfile(a_f, n_rows, n_cols, block_size=3).to_numpy()
            a5 = BlockMatrix.fromfile(bm_f, n_rows, n_cols).to_numpy()

            with BatchedAsserts() as b:
                b.assert_eq(a1, a)
                b.assert_eq(a2, a)
                b.assert_eq(a3, a)
                b.assert_eq(a4, a)
                b.assert_eq(a5, a)

        bmt = bm.T
        at = a.T

        with hl.TemporaryFilename() as bmt_f, hl.TemporaryFilename() as at_f:
            bmt.tofile(bmt_f)
            hl.current_backend().fs.open(at_f, mode='wb').write(at.tobytes())

            at1 = bmt.to_numpy()
            at2 = BlockMatrix.from_numpy(at).to_numpy()
            at3 = np.frombuffer(
                hl.current_backend().fs.open(bmt_f, mode='rb').read()
            ).reshape((n_cols, n_rows))
            at4 = BlockMatrix.fromfile(at_f, n_cols, n_rows).to_numpy()
            at5 = BlockMatrix.fromfile(bmt_f, n_cols, n_rows).to_numpy()

            with BatchedAsserts() as b:
                b.assert_eq(at1, at)
                b.assert_eq(at2, at)
                b.assert_eq(at3, at)
                b.assert_eq(at4, at)
                b.assert_eq(at5, at)

    @fails_service_backend()
    @fails_local_backend()
    def test_numpy_round_trip_force_blocking(self):
        n_rows = 10
        n_cols = 11
        data = np.random.rand(n_rows * n_cols)
        a = data.reshape((n_rows, n_cols))

        bm = BlockMatrix._create(n_rows, n_cols, data.tolist(), block_size=4)
        self._assert_eq(bm.to_numpy(_force_blocking=True), a)

    @fails_service_backend()
    @fails_local_backend()
    def test_to_table(self):
        schema = hl.tstruct(row_idx=hl.tint64, entries=hl.tarray(hl.tfloat64))
        rows = [{'row_idx': 0, 'entries': [0.0, 1.0]},
                {'row_idx': 1, 'entries': [2.0, 3.0]},
                {'row_idx': 2, 'entries': [4.0, 5.0]},
                {'row_idx': 3, 'entries': [6.0, 7.0]},
                {'row_idx': 4, 'entries': [8.0, 9.0]}]

        for n_partitions in [1, 2, 3]:
            for block_size in [1, 2, 5]:
                expected = hl.Table.parallelize(rows, schema, 'row_idx', n_partitions)
                bm = BlockMatrix._create(5, 2, [float(i) for i in range(10)], block_size)
                actual = bm.to_table_row_major(n_partitions)
                self.assertTrue(expected._same(actual))

    @fails_service_backend()
    @fails_local_backend()
    def test_to_table_maximum_cache_memory_in_bytes_limits(self):
        bm = BlockMatrix._create(5, 2, [float(i) for i in range(10)], 2)
        try:
            bm.to_table_row_major(2, maximum_cache_memory_in_bytes=15)._force_count()
        except Exception as exc:
            assert 'BlockMatrixCachedPartFile must be able to hold at least one row of every block in memory' in exc.args[0]
        else:
            assert False

        bm = BlockMatrix._create(5, 2, [float(i) for i in range(10)], 2)
        bm.to_table_row_major(2, maximum_cache_memory_in_bytes=16)._force_count()

    @fails_service_backend()
    @fails_local_backend()
    def test_to_matrix_table(self):
        n_partitions = 2
        rows, cols = 2, 5
        bm = BlockMatrix._create(rows, cols, [float(i) for i in range(10)])
        actual = bm.to_matrix_table_row_major(n_partitions)

        expected = hl.utils.range_matrix_table(rows, cols)
        expected = expected.annotate_entries(element=hl.float64(expected.row_idx * cols + expected.col_idx))
        expected = expected.key_cols_by(col_idx=hl.int64(expected.col_idx))
        expected = expected.key_rows_by(row_idx=hl.int64(expected.row_idx))
        assert expected._same(actual)

        bm = BlockMatrix.random(50, 100, block_size=25, seed=0)
        mt = bm.to_matrix_table_row_major(n_partitions)
        mt_round_trip = BlockMatrix.from_entry_expr(mt.element).to_matrix_table_row_major()
        assert mt._same(mt_round_trip)

    def test_paired_elementwise_ops(self):
        nx = np.array([[2.0]])
        nc = np.array([[1.0], [2.0]])
        nr = np.array([[1.0, 2.0, 3.0]])
        nm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        e = 2.0
        # BlockMatrixMap requires very simple IRs on the SparkBackend. If I use
        # `from_ndarray` here, it generates an `NDArrayRef` expression that it can't handle.
        # Will be fixed by improving FoldConstants handling of ndarrays or fully lowering BlockMatrix.
        x = BlockMatrix._create(1, 1, [2.0], block_size=8)
        c = BlockMatrix.from_ndarray(hl.literal(nc), block_size=8)
        r = BlockMatrix.from_ndarray(hl.literal(nr), block_size=8)
        m = BlockMatrix.from_ndarray(hl.literal(nm), block_size=8)

        self.assertRaises(TypeError,
                          lambda: x + np.array(['one'], dtype=str))

        with BatchedAsserts() as b:

            b.assert_eq(+m, 0 + m)
            b.assert_eq(-m, 0 - m)

            # addition
            b.assert_eq(x + e, nx + e)
            b.assert_eq(c + e, nc + e)
            b.assert_eq(r + e, nr + e)
            b.assert_eq(m + e, nm + e)

            b.assert_eq(x + e, e + x)
            b.assert_eq(c + e, e + c)
            b.assert_eq(r + e, e + r)
            b.assert_eq(m + e, e + m)

            b.assert_eq(x + x, 2 * x)
            b.assert_eq(c + c, 2 * c)
            b.assert_eq(r + r, 2 * r)
            b.assert_eq(m + m, 2 * m)

            b.assert_eq(x + c, np.array([[3.0], [4.0]]))
            b.assert_eq(x + r, np.array([[3.0, 4.0, 5.0]]))
            b.assert_eq(x + m, np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]))
            b.assert_eq(c + m, np.array([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]]))
            b.assert_eq(r + m, np.array([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]]))
            b.assert_eq(x + c, c + x)
            b.assert_eq(x + r, r + x)
            b.assert_eq(x + m, m + x)
            b.assert_eq(c + m, m + c)
            b.assert_eq(r + m, m + r)

            b.assert_eq(x + nx, x + x)
            b.assert_eq(x + nc, x + c)
            b.assert_eq(x + nr, x + r)
            b.assert_eq(x + nm, x + m)
            b.assert_eq(c + nx, c + x)
            b.assert_eq(c + nc, c + c)
            b.assert_eq(c + nm, c + m)
            b.assert_eq(r + nx, r + x)
            b.assert_eq(r + nr, r + r)
            b.assert_eq(r + nm, r + m)
            b.assert_eq(m + nx, m + x)
            b.assert_eq(m + nc, m + c)
            b.assert_eq(m + nr, m + r)
            b.assert_eq(m + nm, m + m)

            # subtraction
            b.assert_eq(x - e, nx - e)
            b.assert_eq(c - e, nc - e)
            b.assert_eq(r - e, nr - e)
            b.assert_eq(m - e, nm - e)

            b.assert_eq(x - e, -(e - x))
            b.assert_eq(c - e, -(e - c))
            b.assert_eq(r - e, -(e - r))
            b.assert_eq(m - e, -(e - m))

            b.assert_eq(x - x, np.zeros((1, 1)))
            b.assert_eq(c - c, np.zeros((2, 1)))
            b.assert_eq(r - r, np.zeros((1, 3)))
            b.assert_eq(m - m, np.zeros((2, 3)))

            b.assert_eq(x - c, np.array([[1.0], [0.0]]))
            b.assert_eq(x - r, np.array([[1.0, 0.0, -1.0]]))
            b.assert_eq(x - m, np.array([[1.0, 0.0, -1.0], [-2.0, -3.0, -4.0]]))
            b.assert_eq(c - m, np.array([[0.0, -1.0, -2.0], [-2.0, -3.0, -4.0]]))
            b.assert_eq(r - m, np.array([[0.0, 0.0, 0.0], [-3.0, -3.0, -3.0]]))
            b.assert_eq(x - c, -(c - x))
            b.assert_eq(x - r, -(r - x))
            b.assert_eq(x - m, -(m - x))
            b.assert_eq(c - m, -(m - c))
            b.assert_eq(r - m, -(m - r))

            b.assert_eq(x - nx, x - x)
            b.assert_eq(x - nc, x - c)
            b.assert_eq(x - nr, x - r)
            b.assert_eq(x - nm, x - m)
            b.assert_eq(c - nx, c - x)
            b.assert_eq(c - nc, c - c)
            b.assert_eq(c - nm, c - m)
            b.assert_eq(r - nx, r - x)
            b.assert_eq(r - nr, r - r)
            b.assert_eq(r - nm, r - m)
            b.assert_eq(m - nx, m - x)
            b.assert_eq(m - nc, m - c)
            b.assert_eq(m - nr, m - r)
            b.assert_eq(m - nm, m - m)

            # multiplication
            b.assert_eq(x * e, nx * e)
            b.assert_eq(c * e, nc * e)
            b.assert_eq(r * e, nr * e)
            b.assert_eq(m * e, nm * e)

            b.assert_eq(x * e, e * x)
            b.assert_eq(c * e, e * c)
            b.assert_eq(r * e, e * r)
            b.assert_eq(m * e, e * m)

            b.assert_eq(x * x, x ** 2)
            b.assert_eq(c * c, c ** 2)
            b.assert_eq(r * r, r ** 2)
            b.assert_eq(m * m, m ** 2)

            b.assert_eq(x * c, np.array([[2.0], [4.0]]))
            b.assert_eq(x * r, np.array([[2.0, 4.0, 6.0]]))
            b.assert_eq(x * m, np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]))
            b.assert_eq(c * m, np.array([[1.0, 2.0, 3.0], [8.0, 10.0, 12.0]]))
            b.assert_eq(r * m, np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]))
            b.assert_eq(x * c, c * x)
            b.assert_eq(x * r, r * x)
            b.assert_eq(x * m, m * x)
            b.assert_eq(c * m, m * c)
            b.assert_eq(r * m, m * r)

            b.assert_eq(x * nx, x * x)
            b.assert_eq(x * nc, x * c)
            b.assert_eq(x * nr, x * r)
            b.assert_eq(x * nm, x * m)
            b.assert_eq(c * nx, c * x)
            b.assert_eq(c * nc, c * c)
            b.assert_eq(c * nm, c * m)
            b.assert_eq(r * nx, r * x)
            b.assert_eq(r * nr, r * r)
            b.assert_eq(r * nm, r * m)
            b.assert_eq(m * nx, m * x)
            b.assert_eq(m * nc, m * c)
            b.assert_eq(m * nr, m * r)
            b.assert_eq(m * nm, m * m)

            # division
            b.assert_close(x / e, nx / e)
            b.assert_close(c / e, nc / e)
            b.assert_close(r / e, nr / e)
            b.assert_close(m / e, nm / e)

            b.assert_close(x / e, 1 / (e / x))
            b.assert_close(c / e, 1 / (e / c))
            b.assert_close(r / e, 1 / (e / r))
            b.assert_close(m / e, 1 / (e / m))

            b.assert_close(x / x, np.ones((1, 1)))
            b.assert_close(c / c, np.ones((2, 1)))
            b.assert_close(r / r, np.ones((1, 3)))
            b.assert_close(m / m, np.ones((2, 3)))

            b.assert_close(x / c, np.array([[2 / 1.0], [2 / 2.0]]))
            b.assert_close(x / r, np.array([[2 / 1.0, 2 / 2.0, 2 / 3.0]]))
            b.assert_close(x / m, np.array([[2 / 1.0, 2 / 2.0, 2 / 3.0], [2 / 4.0, 2 / 5.0, 2 / 6.0]]))
            b.assert_close(c / m, np.array([[1 / 1.0, 1 / 2.0, 1 / 3.0], [2 / 4.0, 2 / 5.0, 2 / 6.0]]))
            b.assert_close(r / m, np.array([[1 / 1.0, 2 / 2.0, 3 / 3.0], [1 / 4.0, 2 / 5.0, 3 / 6.0]]))
            b.assert_close(x / c, 1 / (c / x))
            b.assert_close(x / r, 1 / (r / x))
            b.assert_close(x / m, 1 / (m / x))
            b.assert_close(c / m, 1 / (m / c))
            b.assert_close(r / m, 1 / (m / r))

            b.assert_close(x / nx, x / x)
            b.assert_close(x / nc, x / c)
            b.assert_close(x / nr, x / r)
            b.assert_close(x / nm, x / m)
            b.assert_close(c / nx, c / x)
            b.assert_close(c / nc, c / c)
            b.assert_close(c / nm, c / m)
            b.assert_close(r / nx, r / x)
            b.assert_close(r / nr, r / r)
            b.assert_close(r / nm, r / m)
            b.assert_close(m / nx, m / x)
            b.assert_close(m / nc, m / c)
            b.assert_close(m / nr, m / r)
            b.assert_close(m / nm, m / m)


    def test_special_elementwise_ops(self):
        nm = np.array([[1.0, 2.0, 3.0, 3.14], [4.0, 5.0, 6.0, 12.12]])
        m = BlockMatrix.from_ndarray(hl.nd.array(nm))

        self._assert_close(m ** 3, nm ** 3)
        self._assert_close(m.sqrt(), np.sqrt(nm))
        self._assert_close(m.ceil(), np.ceil(nm))
        self._assert_close(m.floor(), np.floor(nm))
        self._assert_close(m.log(), np.log(nm))
        self._assert_close((m - 4).abs(), np.abs(nm - 4))

    def test_matrix_ops(self):
        nm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = BlockMatrix.from_ndarray(hl.nd.array(nm), block_size=2)
        nsquare = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        square = BlockMatrix.from_ndarray(hl.nd.array(nsquare), block_size=2)

        nrow = np.array([[7.0, 8.0, 9.0]])
        row = BlockMatrix.from_ndarray(hl.nd.array(nrow), block_size=2)

        with BatchedAsserts() as b:
            b.assert_eq(m.T, nm.T)
            b.assert_eq(m.T, nm.T)
            b.assert_eq(row.T, nrow.T)

            b.assert_eq(m @ m.T, nm @ nm.T)
            b.assert_eq(m @ nm.T, nm @ nm.T)
            b.assert_eq(row @ row.T, nrow @ nrow.T)
            b.assert_eq(row @ nrow.T, nrow @ nrow.T)

            b.assert_eq(m.T @ m, nm.T @ nm)
            b.assert_eq(m.T @ nm, nm.T @ nm)
            b.assert_eq(row.T @ row, nrow.T @ nrow)
            b.assert_eq(row.T @ nrow, nrow.T @ nrow)

            self.assertRaises(ValueError, lambda: m @ m)
            self.assertRaises(ValueError, lambda: m @ nm)

            b.assert_eq(m.diagonal(), np.array([[1.0, 5.0]]))
            b.assert_eq(m.T.diagonal(), np.array([[1.0, 5.0]]))
            b.assert_eq((m @ m.T).diagonal(), np.array([[14.0, 77.0]]))

    def test_matrix_sums(self):
        nm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = BlockMatrix.from_ndarray(hl.nd.array(nm), block_size=2)
        nsquare = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        square = BlockMatrix.from_ndarray(hl.nd.array(nsquare), block_size=2)

        nrow = np.array([[7.0, 8.0, 9.0]])
        row = BlockMatrix.from_ndarray(hl.nd.array(nrow), block_size=2)

        with BatchedAsserts() as b:

            b.assert_eq(m.sum(axis=0).T, np.array([[5.0], [7.0], [9.0]]))
            b.assert_eq(m.sum(axis=1).T, np.array([[6.0, 15.0]]))
            b.assert_eq(m.sum(axis=0).T + row, np.array([[12.0, 13.0, 14.0],
                                                             [14.0, 15.0, 16.0],
                                                             [16.0, 17.0, 18.0]]))
            b.assert_eq(m.sum(axis=0) + row.T, np.array([[12.0, 14.0, 16.0],
                                                             [13.0, 15.0, 17.0],
                                                             [14.0, 16.0, 18.0]]))
            b.assert_eq(square.sum(axis=0).T + square.sum(axis=1), np.array([[18.0], [30.0], [42.0]]))

    @fails_service_backend()
    @fails_local_backend()
    def test_tree_matmul(self):
        nm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = BlockMatrix.from_numpy(nm, block_size=2)
        nrow = np.array([[7.0, 8.0, 9.0]])
        row = BlockMatrix.from_numpy(nrow, block_size=2)

        with BatchedAsserts() as b:

            b.assert_eq(m.tree_matmul(m.T, splits=2), nm @ nm.T)
            b.assert_eq(m.tree_matmul(nm.T, splits=2), nm @ nm.T)
            b.assert_eq(row.tree_matmul(row.T, splits=2), nrow @ nrow.T)
            b.assert_eq(row.tree_matmul(nrow.T, splits=2), nrow @ nrow.T)

            b.assert_eq(m.T.tree_matmul(m, splits=2), nm.T @ nm)
            b.assert_eq(m.T.tree_matmul(nm, splits=2), nm.T @ nm)
            b.assert_eq(row.T.tree_matmul(row, splits=2), nrow.T @ nrow)
            b.assert_eq(row.T.tree_matmul(nrow, splits=2), nrow.T @ nrow)

            # Variety of block sizes and splits
            fifty_by_sixty = np.arange(50 * 60).reshape((50, 60))
            sixty_by_twenty_five = np.arange(60 * 25).reshape((60, 25))
            block_sizes = [7, 10]
            split_sizes = [2, 9]
            for block_size in block_sizes:
                bm_fifty_by_sixty = BlockMatrix.from_numpy(fifty_by_sixty, block_size)
                bm_sixty_by_twenty_five = BlockMatrix.from_numpy(sixty_by_twenty_five, block_size)
                for split_size in split_sizes:
                    b.assert_eq(bm_fifty_by_sixty.tree_matmul(bm_fifty_by_sixty.T, splits=split_size), fifty_by_sixty @ fifty_by_sixty.T)
                    b.assert_eq(bm_fifty_by_sixty.tree_matmul(bm_sixty_by_twenty_five, splits=split_size), fifty_by_sixty @ sixty_by_twenty_five)

    def test_fill(self):
        nd = np.ones((3, 5))
        bm = BlockMatrix.fill(3, 5, 1.0)
        bm2 = BlockMatrix.fill(3, 5, 1.0, block_size=2)

        with BatchedAsserts() as b:

            self.assertTrue(bm.block_size == BlockMatrix.default_block_size())
            self.assertTrue(bm2.block_size == 2)
            b.assert_eq(bm, nd)
            b.assert_eq(bm2, nd)

    def test_sum(self):
        nd = np.arange(11 * 13, dtype=np.float64).reshape((11, 13))
        bm = BlockMatrix.from_ndarray(hl.literal(nd), block_size=3)

        self.assert_sums_agree(bm, nd)

    @fails_local_backend
    @fails_service_backend(reason='ExecuteContext.scoped requires SparkBackend')
    def test_sum_with_sparsify(self):
        nd = np.zeros(shape=(5, 7))
        nd[2, 4] = 1.0
        nd[2, 5] = 2.0
        nd[3, 4] = 3.0
        nd[3, 5] = 4.0

        hnd = hl.nd.array(nd)
        bm = BlockMatrix.from_ndarray(hnd, block_size=2).sparsify_rectangles([[2, 4, 4, 6]])

        bm2 = BlockMatrix.from_ndarray(hnd, block_size=2).sparsify_rectangles([[2, 4, 4, 6], [0, 5, 0, 1]])

        bm3 = BlockMatrix.from_ndarray(hnd, block_size=2).sparsify_rectangles([[2, 4, 4, 6], [0, 1, 0, 7]])

        nd4 = np.zeros(shape=(5, 7))
        bm4 = BlockMatrix.fill(5, 7, value=0.0, block_size=2).sparsify_rectangles([])

        self.assert_sums_agree(bm, nd)
        self.assert_sums_agree(bm2, nd)
        self.assert_sums_agree(bm3, nd)
        self.assert_sums_agree(bm4, nd4)

    def test_slicing(self):
        nd = np.array(np.arange(0, 80, dtype=float)).reshape(8, 10)
        bm = BlockMatrix.from_ndarray(hl.literal(nd), block_size=3)

        for indices in [(0, 0), (5, 7), (-3, 9), (-8, -10)]:
            self._assert_eq(bm[indices], nd[indices])

        with BatchedAsserts() as b:

            for indices in [(0, slice(3, 4)),
                            (1, slice(3, 4)),
                            (-8, slice(3, 4)),
                            (-1, slice(3, 4))]:
                b.assert_eq(bm[indices], np.expand_dims(nd[indices], 0))
                b.assert_eq(bm[indices] - bm, nd[indices] - nd)
                b.assert_eq(bm - bm[indices], nd - nd[indices])

            for indices in [(slice(3, 4), 0),
                            (slice(3, 4), 1),
                            (slice(3, 4), -8),
                            (slice(3, 4), -1)]:
                b.assert_eq(bm[indices], np.expand_dims(nd[indices], 1))
                b.assert_eq(bm[indices] - bm, nd[indices] - nd)
                b.assert_eq(bm - bm[indices], nd - nd[indices])

            for indices in [
                (slice(0, 8), slice(0, 10)),
                (slice(0, 8, 2), slice(0, 10, 2)),
                (slice(2, 4), slice(5, 7)),
                (slice(-8, -1), slice(-10, -1)),
                (slice(-8, -1, 2), slice(-10, -1, 2)),
                (slice(None, 4, 1), slice(None, 4, 1)),
                (slice(4, None), slice(4, None)),
                (slice(None, None), slice(None, None))
            ]:
                b.assert_eq(bm[indices], nd[indices])
                b.assert_eq(bm[indices][:, :2], nd[indices][:, :2])
                b.assert_eq(bm[indices][:2, :], nd[indices][:2, :])

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

    def test_diagonal_sparse(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0],
                       [17.0, 18.0, 19.0, 20.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)
        bm = bm.sparsify_row_intervals([0, 0, 0, 0, 0], [2, 2, 2, 2, 2])

        # FIXME doesn't work in service, if test_is_sparse works, uncomment below
        # self.assertTrue(bm.is_sparse)
        self._assert_eq(bm.diagonal(), np.array([[1.0, 6.0, 0.0, 0.0]]))

    @fails_service_backend()
    @fails_local_backend()
    def test_slices_with_sparsify(self):
        nd = np.array(np.arange(0, 80, dtype=float)).reshape(8, 10)
        bm = BlockMatrix.from_numpy(nd, block_size=3)
        bm2 = bm.sparsify_row_intervals([0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(bm2[0, 1], 1.0)
        self.assertEqual(bm2[0, 2], 0.0)
        self.assertEqual(bm2[0, 9], 0.0)

        nd2 = np.zeros(shape=(8, 10))
        nd2[0, 1] = 1.0

        with BatchedAsserts() as b:

            b.assert_eq(bm2[:, :], nd2)

            b.assert_eq(bm2[:, 1], nd2[:, 1:2])
            b.assert_eq(bm2[1, :], nd2[1:2, :])
            b.assert_eq(bm2[0:5, 0:5], nd2[0:5, 0:5])

    def test_sparsify_row_intervals(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)

        with BatchedAsserts() as b:

            b.assert_eq(
                bm.sparsify_row_intervals(
                    starts=[1, 0, 2, 2],
                    stops= [2, 0, 3, 4]),
                np.array([[ 0.,  2.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.],
                          [ 0.,  0., 11.,  0.],
                          [ 0.,  0., 15., 16.]]))

            b.assert_eq(
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
                b.assert_eq(actual, expected)

    def test_sparsify_band(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)

        with BatchedAsserts() as b:

            b.assert_eq(
                bm.sparsify_band(lower=-1, upper=2),
                np.array([[ 1.,  2.,  3.,  0.],
                          [ 5.,  6.,  7.,  8.],
                          [ 0., 10., 11., 12.],
                          [ 0.,  0., 15., 16.]]))

            b.assert_eq(
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
                b.assert_eq(actual, nd2 * mask)

    def test_sparsify_triangle(self):
        nd = np.array([[ 1.0,  2.0,  3.0,  4.0],
                       [ 5.0,  6.0,  7.0,  8.0],
                       [ 9.0, 10.0, 11.0, 12.0],
                       [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)

        # FIXME doesn't work in service, if test_is_sparse works, uncomment below
        # self.assertFalse(bm.is_sparse)
        # self.assertTrue(bm.sparsify_triangle().is_sparse)

        with BatchedAsserts() as b:

            b.assert_eq(
                bm.sparsify_triangle(),
                np.array([[ 1.,  2.,  3.,  4.],
                          [ 0.,  6.,  7.,  8.],
                          [ 0.,  0., 11., 12.],
                          [ 0.,  0.,  0., 16.]]))

            b.assert_eq(
                bm.sparsify_triangle(lower=True),
                np.array([[ 1.,  0.,  0.,  0.],
                          [ 5.,  6.,  0.,  0.],
                          [ 9., 10., 11.,  0.],
                          [13., 14., 15., 16.]]))

            b.assert_eq(
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

        with BatchedAsserts() as b:

            b.assert_eq(
                bm.sparsify_rectangles([[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4]]),
                np.array([[ 1.,  2.,  3.,  4.],
                          [ 5.,  6.,  7.,  8.],
                          [ 9., 10.,  0.,  0.],
                          [13., 14.,  0.,  0.]]))

            b.assert_eq(bm.sparsify_rectangles([]), np.zeros(shape=(4, 4)))

    @fails_service_backend()
    @fails_local_backend()
    def test_export_rectangles(self):
        nd = np.arange(0, 80, dtype=float).reshape(8, 10)

        rects1 = [[0, 1, 0, 1], [4, 5, 7, 8]]

        rects2 = [[4, 5, 0, 10], [0, 8, 4, 5]]

        rects3 = [[0, 1, 0, 1], [1, 2, 1, 2], [2, 3, 2, 3],
                  [3, 5, 3, 6], [3, 6, 3, 7], [3, 7, 3, 8],
                  [4, 5, 0, 10], [0, 8, 4, 5], [0, 8, 0, 10]]

        for rects in [rects1, rects2, rects3]:
            for block_size in [3, 4, 10]:
                with hl.TemporaryDirectory() as rect_uri, hl.TemporaryDirectory() as rect_uri_bytes:
                    bm = BlockMatrix.from_numpy(nd, block_size=block_size)

                    bm.export_rectangles(rect_uri, rects)
                    self._assert_rectangles_eq(nd, rect_uri, rects)

                    bm.export_rectangles(rect_uri_bytes, rects, binary=True)
                    self._assert_rectangles_eq(nd, rect_uri_bytes, rects, binary=True)

    @fails_service_backend()
    @fails_local_backend()
    def test_export_rectangles_sparse(self):
        with hl.TemporaryDirectory() as rect_uri:
            nd = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0],
                           [13.0, 14.0, 15.0, 16.0]])
            bm = BlockMatrix.from_numpy(nd, block_size=2)
            sparsify_rects = [[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4]]
            export_rects = [[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4], [2, 4, 2, 4]]
            bm.sparsify_rectangles(sparsify_rects).export_rectangles(rect_uri, export_rects)

            expected = np.array([[1.0, 2.0, 3.0, 4.0],
                                 [5.0, 6.0, 7.0, 8.0],
                                 [9.0, 10.0, 0.0, 0.0],
                                 [13.0, 14.0, 0.0, 0.0]])

            self._assert_rectangles_eq(expected, rect_uri, export_rects)

    @fails_service_backend()
    @fails_local_backend()
    def test_export_rectangles_filtered(self):
        with hl.TemporaryDirectory() as rect_uri:
            nd = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0],
                           [13.0, 14.0, 15.0, 16.0]])
            bm = BlockMatrix.from_numpy(nd)
            bm = bm[1:3, 1:3]
            export_rects = [[0, 1, 0, 2], [1, 2, 0, 2]]
            bm.export_rectangles(rect_uri, export_rects)

            expected = np.array([[6.0, 7.0],
                                 [10.0, 11.0]])

            self._assert_rectangles_eq(expected, rect_uri, export_rects)

    @fails_service_backend()
    @fails_local_backend()
    def test_export_blocks(self):
        nd = np.ones(shape=(8, 10))
        bm = BlockMatrix.from_numpy(nd, block_size=20)

        with hl.TemporaryDirectory() as bm_uri:
            bm.export_blocks(bm_uri, binary=True)
            actual = BlockMatrix.rectangles_to_numpy(bm_uri, binary=True)
            self._assert_eq(nd, actual)

    @fails_service_backend()
    @fails_local_backend()
    def test_rectangles_to_numpy(self):
        nd = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])

        rects = [[0, 3, 0, 1], [1, 2, 0, 2]]

        with hl.TemporaryDirectory() as rect_uri, hl.TemporaryDirectory() as rect_bytes_uri:
            BlockMatrix.from_numpy(nd).export_rectangles(rect_uri, rects)
            BlockMatrix.from_numpy(nd).export_rectangles(rect_bytes_uri, rects, binary=True)

            expected = np.array([[1.0, 0.0],
                                 [4.0, 5.0],
                                 [7.0, 0.0]])
            self._assert_eq(expected, BlockMatrix.rectangles_to_numpy(rect_uri))
            self._assert_eq(expected, BlockMatrix.rectangles_to_numpy(rect_bytes_uri, binary=True))

    def test_to_ndarray(self):
        np_mat = np.arange(12).reshape((4, 3)).astype(np.float64)
        mat = BlockMatrix.from_ndarray(hl.nd.array(np_mat)).to_ndarray()
        self.assertTrue(np.array_equal(np_mat, hl.eval(mat)))

        blocks_to_sparsify = [1, 4, 7, 12, 20, 42, 48]
        sparsed_numpy = sparsify_numpy(np.arange(25*25).reshape((25, 25)), 4,  blocks_to_sparsify)
        sparsed = BlockMatrix.from_ndarray(hl.nd.array(sparsed_numpy), block_size=4)._sparsify_blocks(blocks_to_sparsify).to_ndarray()
        self.assertTrue(np.array_equal(sparsed_numpy, hl.eval(sparsed)))

    def test_block_matrix_entries(self):
        n_rows, n_cols = 5, 3
        rows = [{'i': i, 'j': j, 'entry': float(i + j)} for i in range(n_rows) for j in range(n_cols)]
        schema = hl.tstruct(i=hl.tint32, j=hl.tint32, entry=hl.tfloat64)
        table = hl.Table.parallelize([hl.struct(i=row['i'], j=row['j'], entry=row['entry']) for row in rows], schema)
        table = table.annotate(i=hl.int64(table.i),
                               j=hl.int64(table.j)).key_by('i', 'j')

        ndarray = np.reshape(list(map(lambda row: row['entry'], rows)), (n_rows, n_cols))

        for block_size in [1, 2, 1024]:
            block_matrix = BlockMatrix.from_ndarray(hl.literal(ndarray), block_size)
            entries_table = block_matrix.entries()
            self.assertEqual(entries_table.count(), n_cols * n_rows)
            self.assertEqual(len(entries_table.row), 3)
            self.assertTrue(table._same(entries_table))

    def test_from_entry_expr_filtered(self):
        mt = hl.utils.range_matrix_table(1, 1).filter_entries(False)
        bm = hl.linalg.BlockMatrix.from_entry_expr(mt.row_idx + mt.col_idx, mean_impute=True) # should run without error
        assert np.isnan(bm.entries().entry.collect()[0])

    def test_array_windows(self):
        def assert_eq(a, b):
            self.assertTrue(np.array_equal(a, np.array(b)))

        starts, stops = hl.linalg.utils.array_windows(np.array([1, 2, 4, 4, 6, 8]), 2)
        assert_eq(starts, [0, 0, 1, 1, 2, 4])
        assert_eq(stops, [2, 4, 5, 5, 6, 6])

        starts, stops = hl.linalg.utils.array_windows(np.array([-10.0, -2.5, 0.0, 0.0, 1.2, 2.3, 3.0]), 2.5)
        assert_eq(starts, [0, 1, 1, 1, 2, 2, 4])
        assert_eq(stops, [1, 4, 6, 6, 7, 7, 7])

        starts, stops = hl.linalg.utils.array_windows(np.array([0, 0, 1]), 0)
        assert_eq(starts, [0, 0, 2])
        assert_eq(stops, [2, 2, 3])

        starts, stops = hl.linalg.utils.array_windows(np.array([]), 1)
        self.assertEqual(starts.size, 0)
        self.assertEqual(stops.size, 0)

        starts, stops = hl.linalg.utils.array_windows(np.array([-float('inf'), -1, 0, 1, float("inf")]), 1)
        assert_eq(starts, [0, 1, 1, 2, 4])
        assert_eq(stops, [1, 3, 4, 4, 5])

        self.assertRaises(ValueError, lambda: hl.linalg.utils.array_windows(np.array([1, 0]), -1))
        self.assertRaises(ValueError, lambda: hl.linalg.utils.array_windows(np.array([0, float('nan')]), 1))
        self.assertRaises(ValueError, lambda: hl.linalg.utils.array_windows(np.array([float('nan')]), 1))
        self.assertRaises(ValueError, lambda: hl.linalg.utils.array_windows(np.array([0.0, float('nan')]), 1))
        self.assertRaises(ValueError, lambda: hl.linalg.utils.array_windows(np.array([None]), 1))
        self.assertRaises(ValueError, lambda: hl.linalg.utils.array_windows(np.array([]), -1))
        self.assertRaises(ValueError, lambda: hl.linalg.utils.array_windows(np.array(['str']), 1))

    def test_locus_windows_per_contig(self):
        f = hl._locus_windows_per_contig([[1.0, 3.0, 4.0], [2.0, 2.0], [5.0]], 1.0)
        assert hl.eval(f) == ([0, 1, 1, 3, 3, 5], [1, 3, 3, 5, 5, 6])

    def assert_np_arrays_eq(self, a, b):
        assert np.array_equal(a, np.array(b)), f"a={a}, b={b}"

    def test_locus_windows_1(self):
        centimorgans = hl.literal([0.1, 1.0, 1.0, 1.5, 1.9])

        mt = hl.balding_nichols_model(1, 5, 5).add_row_index()
        mt = mt.annotate_rows(cm=centimorgans[hl.int32(mt.row_idx)]).cache()

        starts, stops = hl.linalg.utils.locus_windows(mt.locus, 2)
        self.assert_np_arrays_eq(starts, [0, 0, 0, 1, 2])
        self.assert_np_arrays_eq(stops, [3, 4, 5, 5, 5])

    def test_locus_windows_2(self):
        centimorgans = hl.literal([0.1, 1.0, 1.0, 1.5, 1.9])

        mt = hl.balding_nichols_model(1, 5, 5).add_row_index()
        mt = mt.annotate_rows(cm=centimorgans[hl.int32(mt.row_idx)]).cache()

        starts, stops = hl.linalg.utils.locus_windows(mt.locus, 0.5, coord_expr=mt.cm)
        self.assert_np_arrays_eq(starts, [0, 1, 1, 1, 3])
        self.assert_np_arrays_eq(stops, [1, 4, 4, 5, 5])

    def test_locus_windows_3(self):
        centimorgans = hl.literal([0.1, 1.0, 1.0, 1.5, 1.9])

        mt = hl.balding_nichols_model(1, 5, 5).add_row_index()
        mt = mt.annotate_rows(cm=centimorgans[hl.int32(mt.row_idx)]).cache()

        starts, stops = hl.linalg.utils.locus_windows(mt.locus, 1.0, coord_expr=2 * centimorgans[hl.int32(mt.row_idx)])
        self.assert_np_arrays_eq(starts, [0, 1, 1, 1, 3])
        self.assert_np_arrays_eq(stops, [1, 4, 4, 5, 5])

    def test_locus_windows_4(self):
        rows = [{'locus': hl.Locus('1', 1), 'cm': 1.0},
                {'locus': hl.Locus('1', 2), 'cm': 3.0},
                {'locus': hl.Locus('1', 4), 'cm': 4.0},
                {'locus': hl.Locus('2', 1), 'cm': 2.0},
                {'locus': hl.Locus('2', 1), 'cm': 2.0},
                {'locus': hl.Locus('3', 3), 'cm': 5.0}]

        ht = hl.Table.parallelize(rows,
                                  hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64),
                                  key=['locus'])

        starts, stops = hl.linalg.utils.locus_windows(ht.locus, 1)
        self.assert_np_arrays_eq(starts, [0, 0, 2, 3, 3, 5])
        self.assert_np_arrays_eq(stops, [2, 2, 3, 5, 5, 6])

    def dummy_table_with_loci_and_cms():
        rows = [{'locus': hl.Locus('1', 1), 'cm': 1.0},
                {'locus': hl.Locus('1', 2), 'cm': 3.0},
                {'locus': hl.Locus('1', 4), 'cm': 4.0},
                {'locus': hl.Locus('2', 1), 'cm': 2.0},
                {'locus': hl.Locus('2', 1), 'cm': 2.0},
                {'locus': hl.Locus('3', 3), 'cm': 5.0}]

        return hl.Table.parallelize(rows,
                                    hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64),
                                    key=['locus'])

    def test_locus_windows_5(self):
        ht = self.dummy_table_with_loci_and_cms()
        starts, stops = hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=ht.cm)
        self.assert_np_arrays_eq(starts, [0, 1, 1, 3, 3, 5])
        self.assert_np_arrays_eq(stops, [1, 3, 3, 5, 5, 6])

    def test_locus_windows_6(self):
        ht = self.dummy_table_with_loci_and_cms()
        with self.assertRaises(HailUserError) as cm:
            hl.linalg.utils.locus_windows(ht.order_by(ht.cm).locus, 1.0)
        assert 'ascending order' in str(cm.exception)

    def test_locus_windows_7(self):
        ht = self.dummy_table_with_loci_and_cms()
        with self.assertRaises(ExpressionException) as cm:
            hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=hl.utils.range_table(1).idx)
        assert 'different source' in str(cm.exception)

    def test_locus_windows_8(self):
        ht = self.dummy_table_with_loci_and_cms()
        with self.assertRaises(ExpressionException) as cm:
            hl.linalg.utils.locus_windows(hl.locus('1', 1), 1.0)
        assert "no source" in str(cm.exception)

    def test_locus_windows_9(self):
        ht = self.dummy_table_with_loci_and_cms()
        with self.assertRaises(ExpressionException) as cm:
            hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=0.0)
        assert "no source" in str(cm.exception)

    def test_locus_windows_10(self):
        ht = self.dummy_table_with_loci_and_cms()
        ht = ht.annotate_globals(x = hl.locus('1', 1), y = 1.0)
        with self.assertRaises(ExpressionException) as cm:
            hl.linalg.utils.locus_windows(ht.x, 1.0)
        assert "row-indexed" in str(cm.exception)

        with self.assertRaises(ExpressionException) as cm:
            hl.linalg.utils.locus_windows(ht.locus, 1.0, ht.y)
        assert "row-indexed" in str(cm.exception)

    def test_locus_windows_11(self):
        ht = hl.Table.parallelize([{'locus': hl.missing(hl.tlocus()), 'cm': 1.0}],
                                  hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64), key=['locus'])
        with self.assertRaises(HailUserError) as cm:
            hl.linalg.utils.locus_windows(ht.locus, 1.0)
        assert "missing value for 'locus_expr'" in str(cm.exception)

        with self.assertRaises(HailUserError) as cm:
            hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=ht.cm)
        assert "missing value for 'locus_expr'" in str(cm.exception)

    def test_locus_windows_12(self):
        ht = hl.Table.parallelize([{'locus': hl.Locus('1', 1), 'cm': hl.missing(hl.tfloat64)}],
                                  hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64), key=['locus'])
        with self.assertRaises(FatalError) as cm:
            hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=ht.cm)
        assert "missing value for 'coord_expr'" in str(cm.exception)

    def test_write_overwrite(self):
        with hl.TemporaryDirectory(ensure_exists=False) as path:
            bm = BlockMatrix.from_numpy(np.array([[0]]))
            bm.write(path)
            self.assertRaises(FatalError, lambda: bm.write(path))

            bm2 = BlockMatrix.from_numpy(np.array([[1]]))
            bm2.write(path, overwrite=True)
            self._assert_eq(BlockMatrix.read(path), bm2)

    def test_stage_locally(self):
        nd = np.arange(0, 80, dtype=float).reshape(8, 10)
        with hl.TemporaryDirectory(ensure_exists=False) as bm_uri:
            BlockMatrix.from_numpy(nd, block_size=3).write(bm_uri, stage_locally=True)

            bm = BlockMatrix.read(bm_uri)
            self._assert_eq(nd, bm)

    def test_svd(self):
        def assert_same_columns_up_to_sign(a, b):
            for j in range(a.shape[1]):
                assert np.allclose(a[:, j], b[:, j]) or np.allclose(-a[:, j], b[:, j])

        x0 = np.array([[-2.0, 0.0, 3.0],
                       [-1.0, 2.0, 4.0]])
        u0, s0, vt0 = np.linalg.svd(x0, full_matrices=False)

        x = BlockMatrix.from_numpy(x0)

        # _svd
        u, s, vt = x.svd()
        assert_same_columns_up_to_sign(u, u0)
        assert np.allclose(s, s0)
        assert_same_columns_up_to_sign(vt.T, vt0.T)

        s = x.svd(compute_uv=False)
        assert np.allclose(s, s0)

        # left _svd_gramian
        u, s, vt = x.svd(complexity_bound=0)
        assert_same_columns_up_to_sign(u, u0)
        assert np.allclose(s, s0)
        assert_same_columns_up_to_sign(vt.to_numpy().T, vt0.T)

        s = x.svd(compute_uv=False, complexity_bound=0)
        assert np.allclose(s, s0)

        # right _svd_gramian
        x = BlockMatrix.from_numpy(x0.T)
        u, s, vt = x.svd(complexity_bound=0)
        assert_same_columns_up_to_sign(u.to_numpy(), vt0.T)
        assert np.allclose(s, s0)
        assert_same_columns_up_to_sign(vt.T, u0)

        s = x.svd(compute_uv=False, complexity_bound=0)
        assert np.allclose(s, s0)

        # left _svd_gramian when dimensions agree
        x = BlockMatrix.from_numpy(x0[:, :2])
        u, s, vt = x.svd(complexity_bound=0)
        assert isinstance(u, np.ndarray)
        assert isinstance(vt, BlockMatrix)

        # rank-deficient X sets negative eigenvalues to 0.0
        a = np.array([[0.0, 1.0, np.e, np.pi, 10.0, 25.0]])
        x0 = a.T @ a  # rank 1
        e, _ = np.linalg.eigh(x0 @ x0.T)

        x = BlockMatrix.from_numpy(x0)
        s = x.svd(complexity_bound=0, compute_uv=False)
        assert np.all(s >= 0.0)

        s = x.svd(compute_uv=False, complexity_bound=0)
        assert np.all(s >= 0)

    def test_filtering(self):
        np_square = np.arange(16, dtype=np.float64).reshape((4, 4))
        bm = BlockMatrix.from_numpy(np_square)
        assert np.array_equal(bm.filter([3], [3]).to_numpy(), np.array([[15]]))
        assert np.array_equal(bm.filter_rows([3]).filter_cols([3]).to_numpy(), np.array([[15]]))
        assert np.array_equal(bm.filter_cols([3]).filter_rows([3]).to_numpy(), np.array([[15]]))
        assert np.array_equal(bm.filter_rows([2]).filter_rows([0]).to_numpy(), np_square[2:3, :])
        assert np.array_equal(bm.filter_cols([2]).filter_cols([0]).to_numpy(), np_square[:, 2:3])

        with pytest.raises(ValueError) as exc:
            bm.filter_cols([0]).filter_cols([3]).to_numpy()
        assert "index" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            bm.filter_rows([0]).filter_rows([3]).to_numpy()
        assert "index" in str(exc.value)

    @fails_service_backend()
    def test_is_sparse(self):
        block_list = [1, 2]
        np_square = np.arange(16, dtype=np.float64).reshape((4, 4))
        bm = BlockMatrix.from_numpy(np_square, block_size=2)
        bm = bm._sparsify_blocks(block_list)
        assert bm.is_sparse
        assert np.array_equal(
            bm.to_numpy(),
            np.array([[0,  0,  2, 3],
                      [0,  0,  6, 7],
                      [8,  9,  0, 0],
                      [12, 13, 0, 0]]))

    def test_sparsify_blocks(self):
        block_list = [1, 2]
        np_square = np.arange(16, dtype=np.float64).reshape((4, 4))
        block_size = 2
        bm = BlockMatrix.from_numpy(np_square, block_size=block_size)
        bm = bm._sparsify_blocks(block_list)
        sparse_numpy = sparsify_numpy(np_square, block_size, block_list)
        assert np.array_equal(bm.to_numpy(), sparse_numpy)
        assert np.array_equal(
            sparse_numpy,
            np.array([[0,  0,  2, 3],
                      [0,  0,  6, 7],
                      [8,  9,  0, 0],
                      [12, 13, 0, 0]]))

        block_list = [4, 8, 10, 12, 13, 14]
        np_square = np.arange(225, dtype=np.float64).reshape((15, 15))
        block_size = 4
        bm = BlockMatrix.from_numpy(np_square, block_size=block_size)
        bm = bm._sparsify_blocks(block_list)
        sparse_numpy = sparsify_numpy(np_square, block_size, block_list)
        assert np.array_equal(bm.to_numpy(), sparse_numpy)

    def test_sparse_transposition(self):
        block_list = [1, 2]
        np_square = np.arange(16, dtype=np.float64).reshape((4, 4))
        block_size = 2
        bm = BlockMatrix.from_numpy(np_square, block_size=block_size)
        sparse_bm = bm._sparsify_blocks(block_list).T
        sparse_np = sparsify_numpy(np_square, block_size, block_list).T
        assert np.array_equal(sparse_bm.to_numpy(), sparse_np)

        block_list = [4, 8, 10, 12, 13, 14]
        np_square = np.arange(225, dtype=np.float64).reshape((15, 15))
        block_size = 4
        bm = BlockMatrix.from_numpy(np_square, block_size=block_size)
        sparse_bm = bm._sparsify_blocks(block_list).T
        sparse_np = sparsify_numpy(np_square, block_size, block_list).T
        assert np.array_equal(sparse_bm.to_numpy(), sparse_np)

        block_list = [2, 5, 8, 10, 11]
        np_square = np.arange(150, dtype=np.float64).reshape((10, 15))
        block_size = 4
        bm = BlockMatrix.from_numpy(np_square, block_size=block_size)
        sparse_bm = bm._sparsify_blocks(block_list).T
        sparse_np = sparsify_numpy(np_square, block_size, block_list).T
        assert np.array_equal(sparse_bm.to_numpy(), sparse_np)

        block_list = [2, 5, 8, 10, 11]
        np_square = np.arange(165, dtype=np.float64).reshape((15, 11))
        block_size = 4
        bm = BlockMatrix.from_numpy(np_square, block_size=block_size)
        sparse_bm = bm._sparsify_blocks(block_list).T
        sparse_np = sparsify_numpy(np_square, block_size, block_list).T
        assert np.array_equal(sparse_bm.to_numpy(), sparse_np)

    def test_row_blockmatrix_sum(self):

        row = BlockMatrix.from_numpy(np.arange(10))
        col = row.T

        # Summing vertically along a column vector to get a single value
        b = col.sum(axis=0)
        assert b.to_numpy().shape == (1,1)

        # Summing horizontally along a row vector to create a single value
        d = row.sum(axis=1)
        assert d.to_numpy().shape == (1,1)

        # Summing vertically along a row vector to make sure nothing changes
        e = row.sum(axis=0)
        assert e.to_numpy().shape == (1, 10)

        # Summing horizontally along a column vector to make sure nothing changes
        f = col.sum(axis=1)
        assert f.to_numpy().shape == (10, 1)

    @fails_spark_backend()
    def test_map(self):
        np_mat = np.arange(20, dtype=np.float64).reshape((4, 5))
        bm = BlockMatrix.from_ndarray(hl.nd.array(np_mat))
        bm_mapped_arith = bm._map_dense(lambda x: (x * x) + 5)
        self._assert_eq(bm_mapped_arith, np_mat * np_mat + 5)

        bm_mapped_if = bm._map_dense(lambda x: hl.if_else(x >= 1, x, -8.0))
        np_if = np_mat.copy()
        np_if[0, 0] = -8.0
        self._assert_eq(bm_mapped_if, np_if)
