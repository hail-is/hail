import math
from contextlib import contextmanager

import numpy as np
import pytest

import hail as hl
from hail.expr.expressions import ExpressionException
from hail.linalg import BlockMatrix
from hail.utils import FatalError, HailUserError

from ..helpers import fails_local_backend, fails_service_backend, fails_spark_backend, test_timeout


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


def _np_matrix(a):
    return hl.eval(a.to_ndarray()) if isinstance(a, BlockMatrix) else np.array(a)


def _assert_eq(a, b):
    anp = _np_matrix(a)
    bnp = _np_matrix(b)
    np.testing.assert_equal(anp, bnp)


def _assert_close(a, b):
    assert np.allclose(_np_matrix(a), _np_matrix(b))


def _assert_rectangles_eq(expected, rect_path, export_rects, binary=False):
    for (i, r) in enumerate(export_rects):
        piece_path = rect_path + '/rect-' + str(i) + '_' + '-'.join(map(str, r))

        with hl.current_backend().fs.open(piece_path, mode='rb' if binary else 'r') as file:
            expected_rect = expected[r[0] : r[1], r[2] : r[3]]
            actual_rect = (
                np.loadtxt(file, ndmin=2)
                if not binary
                else np.reshape(np.frombuffer(file.read()), (r[1] - r[0], r[3] - r[2]))
            )
            _assert_eq(expected_rect, actual_rect)


def assert_sums_agree(bm, nd):
    assert bm.sum() == pytest.approx(np.sum(nd))
    _assert_close(bm.sum(axis=0), np.sum(nd, axis=0, keepdims=True))
    _assert_close(bm.sum(axis=1), np.sum(nd, axis=1, keepdims=True))


def assert_np_arrays_eq(a, b):
    assert np.array_equal(a, np.array(b)), f"a={a}, b={b}"


def test_from_entry_expr_empty_parts():
    with hl.TemporaryDirectory(ensure_exists=False) as path:
        mt = hl.balding_nichols_model(n_populations=5, n_variants=2000, n_samples=20, n_partitions=200)
        mt = mt.filter_rows((mt.locus.position <= 500) | (mt.locus.position > 1500)).checkpoint(path)
        bm = BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles())
        nd = (bm @ bm.T).to_numpy()
        assert nd.shape == (1000, 1000)


@pytest.mark.parametrize(
    'mean_impute, center, normalize, mk_expected',
    [
        (
            False,
            False,
            False,
            lambda a: a,
        ),
        (
            False,
            False,
            True,
            lambda a: a / np.sqrt(5),
        ),
        (
            False,
            True,
            False,
            lambda a: a - 1.0,
        ),
        (False, True, True, lambda a: (a - 1.0) / np.sqrt(2)),
        (
            True,
            False,
            False,
            lambda a: a,
        ),
        (
            True,
            False,
            True,
            lambda a: a / np.sqrt(5),
        ),
        (
            True,
            True,
            False,
            lambda a: a - 1.0,
        ),
        (True, True, True, lambda a: (a - 1.0) / np.sqrt(2)),
    ],
)
def test_from_entry_expr_options(mean_impute, center, normalize, mk_expected):
    a = np.array([0.0, 1.0, 2.0])

    mt = hl.utils.range_matrix_table(1, 3)
    mt = mt.rename({'row_idx': 'v', 'col_idx': 's'})

    xs = hl.array([0.0, hl.missing(hl.tfloat), 2.0]) if mean_impute else hl.literal(a)
    mt = mt.annotate_entries(x=xs[mt.s])

    expected = mk_expected(a)

    bm = BlockMatrix.from_entry_expr(mt.x, mean_impute, center, normalize)
    actual = np.squeeze(hl.eval(bm.to_ndarray()))
    assert np.allclose(actual, expected)


def test_from_entry_expr_raises_when_values_missing():
    mt = hl.utils.range_matrix_table(1, 3)
    mt = mt.rename({'row_idx': 'v', 'col_idx': 's'})
    actual = hl.array([0.0, hl.missing(hl.tfloat), 2.0])
    mt = mt.annotate_entries(x=actual[mt.s])
    with pytest.raises(Exception, match='Cannot construct an ndarray with missing values'):
        BlockMatrix.from_entry_expr(mt.x)


def test_write_from_entry_expr_overwrite():
    mt = hl.balding_nichols_model(1, 1, 1)
    mt = mt.select_entries(x=mt.GT.n_alt_alleles())
    bm = BlockMatrix.from_entry_expr(mt.x)

    with hl.TemporaryDirectory(ensure_exists=False) as path:
        BlockMatrix.write_from_entry_expr(mt.x, path)
        with pytest.raises(FatalError):
            BlockMatrix.write_from_entry_expr(mt.x, path)

        BlockMatrix.write_from_entry_expr(mt.x, path, overwrite=True)
        _assert_eq(BlockMatrix.read(path), bm)


# non-field expressions currently take a separate code path
def test_write_from_entry_expr_overwrite_non_field_expressions():
    mt = hl.balding_nichols_model(1, 1, 1)
    mt = mt.select_entries(x=mt.GT.n_alt_alleles())
    bm = BlockMatrix.from_entry_expr(mt.x)

    with hl.TemporaryDirectory(ensure_exists=False) as path:
        BlockMatrix.write_from_entry_expr(mt.x + 1, path)
        with pytest.raises(FatalError):
            BlockMatrix.write_from_entry_expr(mt.x + 1, path)

        BlockMatrix.write_from_entry_expr(mt.x + 2, path, overwrite=True)
        _assert_eq(BlockMatrix.read(path), bm + 2)


def test_random_uniform():
    uniform = BlockMatrix.random(10, 10, gaussian=False)

    nuniform = hl.eval(uniform.to_ndarray())
    for row in nuniform:
        for entry in row:
            assert entry > 0


def test_bm_to_numpy():
    bm = BlockMatrix.from_ndarray(hl.nd.arange(20).map(hl.float64).reshape((4, 5)))
    np_bm = bm.to_numpy()
    _assert_eq(np_bm, np.arange(20, dtype=np.float64).reshape((4, 5)))


def test_bm_transpose_to_numpy():
    bm = BlockMatrix.from_ndarray(hl.nd.arange(20).map(hl.float64).reshape((4, 5)))
    np_bm = bm.T.to_numpy()
    _assert_eq(np_bm, np.arange(20, dtype=np.float64).reshape((4, 5)).T)


@contextmanager
def block_matrix_to_tmp_file(data: 'np.ndarray', transpose=False) -> 'str':
    with hl.TemporaryFilename() as f:
        (n_rows, n_cols) = data.shape
        bm = BlockMatrix._create(n_rows, n_cols, data.flatten().tolist(), block_size=4)

        if transpose:
            bm = bm.T

        bm.tofile(f)
        yield f


def test_block_matrix_from_numpy():
    data = np.random.rand(10, 11)
    bm = BlockMatrix.from_numpy(data, block_size=5)
    _assert_eq(data, bm.to_numpy())


def test_block_matrix_from_numpy_transpose():
    data = np.random.rand(10, 11)
    bm = BlockMatrix.from_numpy(data.T, block_size=5)
    _assert_eq(data, bm.T.to_numpy())


def test_block_matrix_to_file_transpose():
    data = np.random.rand(10, 11)
    with block_matrix_to_tmp_file(data, transpose=True) as f:
        _assert_eq(data.T, np.frombuffer(hl.current_backend().fs.open(f, mode='rb').read()).reshape((11, 10)))


def test_numpy_read_block_matrix_to_file():
    data = np.random.rand(10, 11)
    with block_matrix_to_tmp_file(data) as f:
        _assert_eq(data, np.frombuffer(hl.current_backend().fs.open(f, mode='rb').read()).reshape((10, 11)))


def test_block_matrix_from_numpy_bytes():
    data = np.random.rand(10, 11)
    with hl.TemporaryFilename() as f:
        hl.current_backend().fs.open(f, mode='wb').write(data.tobytes())
        array = BlockMatrix.fromfile(f, 10, 11, block_size=3).to_numpy()
        _assert_eq(array, data)


def test_block_matrix_from_file():
    data = np.random.rand(10, 11)
    with block_matrix_to_tmp_file(data) as f:
        array = BlockMatrix.fromfile(f, 10, 11).to_numpy()
        _assert_eq(array, data)


@fails_service_backend()
@fails_local_backend()
def test_numpy_round_trip_force_blocking():
    n_rows = 10
    n_cols = 11
    data = np.random.rand(n_rows * n_cols)
    a = data.reshape((n_rows, n_cols))

    bm = BlockMatrix._create(n_rows, n_cols, data.tolist(), block_size=4)
    _assert_eq(bm.to_numpy(_force_blocking=True), a)


@fails_service_backend()
@fails_local_backend()
@pytest.mark.parametrize(
    'n_partitions,block_size', [(n_partitions, block_size) for n_partitions in [1, 2, 3] for block_size in [1, 2, 5]]
)
def test_to_table(n_partitions, block_size):
    schema = hl.tstruct(row_idx=hl.tint64, entries=hl.tarray(hl.tfloat64))
    rows = [
        {'row_idx': 0, 'entries': [0.0, 1.0]},
        {'row_idx': 1, 'entries': [2.0, 3.0]},
        {'row_idx': 2, 'entries': [4.0, 5.0]},
        {'row_idx': 3, 'entries': [6.0, 7.0]},
        {'row_idx': 4, 'entries': [8.0, 9.0]},
    ]

    expected = hl.Table.parallelize(rows, schema, 'row_idx', n_partitions)
    bm = BlockMatrix._create(5, 2, [float(i) for i in range(10)], block_size)
    actual = bm.to_table_row_major(n_partitions)
    assert expected._same(actual)


@fails_service_backend()
@fails_local_backend()
def test_to_table_maximum_cache_memory_in_bytes_limits():
    bm = BlockMatrix._create(5, 2, [float(i) for i in range(10)], 2)

    with pytest.raises(Exception) as exc_info:
        bm.to_table_row_major(2, maximum_cache_memory_in_bytes=15)._force_count()

    assert (
        'BlockMatrixCachedPartFile must be able to hold at least one row of every block in memory'
        in exc_info.value.args[0]
    )

    bm = BlockMatrix._create(5, 2, [float(i) for i in range(10)], 2)
    bm.to_table_row_major(2, maximum_cache_memory_in_bytes=16)._force_count()


@fails_service_backend()
@fails_local_backend()
def test_to_matrix_table_0():
    n_partitions = 2
    rows, cols = 2, 5
    bm = BlockMatrix._create(rows, cols, [float(i) for i in range(10)])
    actual = bm.to_matrix_table_row_major(n_partitions)

    expected = hl.utils.range_matrix_table(rows, cols)
    expected = expected.annotate_entries(element=hl.float64(expected.row_idx * cols + expected.col_idx))
    expected = expected.key_cols_by(col_idx=hl.int64(expected.col_idx))
    expected = expected.key_rows_by(row_idx=hl.int64(expected.row_idx))
    assert expected._same(actual)


@fails_service_backend()
@fails_local_backend()
def test_to_matrix_table_1():
    n_partitions = 2
    bm = BlockMatrix.random(50, 100, block_size=25, seed=0)
    mt = bm.to_matrix_table_row_major(n_partitions)
    mt_round_trip = BlockMatrix.from_entry_expr(mt.element).to_matrix_table_row_major()
    assert mt._same(mt_round_trip)


@pytest.fixture(scope='module')
def block_matrix_bindings():
    nc = np.array([[1.0], [2.0]])
    nr = np.array([[1.0, 2.0, 3.0]])
    nm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nrow = np.array([[7.0, 8.0, 9.0]])
    nsquare = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    yield {
        'nx': np.array([[2.0]]),
        'nc': nc,
        'nr': nr,
        'nm': nm,
        'nrow': nrow,
        'nsquare': nsquare,
        'e': 2.0,
        # BlockMatrixMap requires very simple IRs on the SparkBackend. If I use
        # `from_ndarray` here, it generates an `NDArrayRef` expression that it can't handle.
        # Will be fixed by improving FoldConstants handling of ndarrays or fully lowering BlockMatrix.
        'x': BlockMatrix._create(1, 1, [2.0], block_size=8),
        'c': BlockMatrix.from_ndarray(hl.literal(nc), block_size=8),
        'r': BlockMatrix.from_ndarray(hl.literal(nr), block_size=8),
        'm': BlockMatrix.from_ndarray(hl.literal(nm), block_size=8),
        'row': BlockMatrix.from_ndarray(hl.nd.array(nrow), block_size=8),
        'square': BlockMatrix.from_ndarray(hl.nd.array(nsquare), block_size=8),
    }


@pytest.mark.parametrize(
    'x, y',
    [  # addition
        ('+m', '0 + m'),
        ('x + e', 'nx + e'),
        ('c + e', 'nc + e'),
        ('r + e', 'nr + e'),
        ('m + e', 'nm + e'),
        ('x + e', 'e + x'),
        ('c + e', 'e + c'),
        ('r + e', 'e + r'),
        ('m + e', 'e + m'),
        ('x + x', '2 * x'),
        ('c + c', '2 * c'),
        ('r + r', '2 * r'),
        ('m + m', '2 * m'),
        ('x + c', 'np.array([[3.0], [4.0]])'),
        ('x + r', 'np.array([[3.0, 4.0, 5.0]])'),
        ('x + m', 'np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])'),
        ('c + m', 'np.array([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]])'),
        ('r + m', 'np.array([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]])'),
        ('x + c', 'c + x'),
        ('x + r', 'r + x'),
        ('x + m', 'm + x'),
        ('c + m', 'm + c'),
        ('r + m', 'm + r'),
        ('x + nx', 'x + x'),
        ('x + nc', 'x + c'),
        ('x + nr', 'x + r'),
        ('x + nm', 'x + m'),
        ('c + nx', 'c + x'),
        ('c + nc', 'c + c'),
        ('c + nm', 'c + m'),
        ('r + nx', 'r + x'),
        ('r + nr', 'r + r'),
        ('r + nm', 'r + m'),
        ('m + nx', 'm + x'),
        ('m + nc', 'm + c'),
        ('m + nr', 'm + r'),
        ('m + nm', 'm + m')
        # subtraction
        ,
        ('-m', '0 - m'),
        ('x - e', 'nx - e'),
        ('c - e', 'nc - e'),
        ('r - e', 'nr - e'),
        ('m - e', 'nm - e'),
        ('x - e', '-(e - x)'),
        ('c - e', '-(e - c)'),
        ('r - e', '-(e - r)'),
        ('m - e', '-(e - m)'),
        ('x - x', 'np.zeros((1, 1))'),
        ('c - c', 'np.zeros((2, 1))'),
        ('r - r', 'np.zeros((1, 3))'),
        ('m - m', 'np.zeros((2, 3))'),
        ('x - c', 'np.array([[1.0], [0.0]])'),
        ('x - r', 'np.array([[1.0, 0.0, -1.0]])'),
        ('x - m', 'np.array([[1.0, 0.0, -1.0], [-2.0, -3.0, -4.0]])'),
        ('c - m', 'np.array([[0.0, -1.0, -2.0], [-2.0, -3.0, -4.0]])'),
        ('r - m', 'np.array([[0.0, 0.0, 0.0], [-3.0, -3.0, -3.0]])'),
        ('x - c', '-(c - x)'),
        ('x - r', '-(r - x)'),
        ('x - m', '-(m - x)'),
        ('c - m', '-(m - c)'),
        ('r - m', '-(m - r)'),
        ('x - nx', 'x - x'),
        ('x - nc', 'x - c'),
        ('x - nr', 'x - r'),
        ('x - nm', 'x - m'),
        ('c - nx', 'c - x'),
        ('c - nc', 'c - c'),
        ('c - nm', 'c - m'),
        ('r - nx', 'r - x'),
        ('r - nr', 'r - r'),
        ('r - nm', 'r - m'),
        ('m - nx', 'm - x'),
        ('m - nc', 'm - c'),
        ('m - nr', 'm - r'),
        ('m - nm', 'm - m')
        # multiplication
        ,
        ('x * e', 'nx * e'),
        ('c * e', 'nc * e'),
        ('r * e', 'nr * e'),
        ('m * e', 'nm * e'),
        ('x * e', 'e * x'),
        ('c * e', 'e * c'),
        ('r * e', 'e * r'),
        ('m * e', 'e * m'),
        ('x * x', 'x ** 2'),
        ('c * c', 'c ** 2'),
        ('r * r', 'r ** 2'),
        ('m * m', 'm ** 2'),
        ('x * c', 'np.array([[2.0], [4.0]])'),
        ('x * r', 'np.array([[2.0, 4.0, 6.0]])'),
        ('x * m', 'np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])'),
        ('c * m', 'np.array([[1.0, 2.0, 3.0], [8.0, 10.0, 12.0]])'),
        ('r * m', 'np.array([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]])'),
        ('x * c', 'c * x'),
        ('x * r', 'r * x'),
        ('x * m', 'm * x'),
        ('c * m', 'm * c'),
        ('r * m', 'm * r'),
        ('x * nx', 'x * x'),
        ('x * nc', 'x * c'),
        ('x * nr', 'x * r'),
        ('x * nm', 'x * m'),
        ('c * nx', 'c * x'),
        ('c * nc', 'c * c'),
        ('c * nm', 'c * m'),
        ('r * nx', 'r * x'),
        ('r * nr', 'r * r'),
        ('r * nm', 'r * m'),
        ('m * nx', 'm * x'),
        ('m * nc', 'm * c'),
        ('m * nr', 'm * r'),
        ('m * nm', 'm * m'),
        ('m.T', 'nm.T'),
        ('m.T', 'nm.T'),
        ('row.T', 'nrow.T'),
        ('m @ m.T', 'nm @ nm.T'),
        ('m @ nm.T', 'nm @ nm.T'),
        ('row @ row.T', 'nrow @ nrow.T'),
        ('row @ nrow.T', 'nrow @ nrow.T'),
        ('m.T @ m', 'nm.T @ nm'),
        ('m.T @ nm', 'nm.T @ nm'),
        ('row.T @ row', 'nrow.T @ nrow'),
        ('row.T @ nrow', 'nrow.T @ nrow'),
    ],
)
def test_block_matrix_elementwise_arithmetic(block_matrix_bindings, x, y):
    lhs = eval(x, block_matrix_bindings)
    rhs = eval(y, {'np': np}, block_matrix_bindings)
    _assert_eq(lhs, rhs)


@pytest.mark.parametrize(
    'x, y',
    [  # division
        ('x / e', 'nx / e'),
        ('c / e', 'nc / e'),
        ('r / e', 'nr / e'),
        ('m / e', 'nm / e'),
        ('x / e', '1 / (e / x)'),
        ('c / e', '1 / (e / c)'),
        ('r / e', '1 / (e / r)'),
        ('m / e', '1 / (e / m)'),
        ('x / x', 'np.ones((1, 1))'),
        ('c / c', 'np.ones((2, 1))'),
        ('r / r', 'np.ones((1, 3))'),
        ('m / m', 'np.ones((2, 3))'),
        ('x / c', 'np.array([[2 / 1.0], [2 / 2.0]])'),
        ('x / r', 'np.array([[2 / 1.0, 2 / 2.0, 2 / 3.0]])'),
        ('x / m', 'np.array([[2 / 1.0, 2 / 2.0, 2 / 3.0], [2 / 4.0, 2 / 5.0, 2 / 6.0]])'),
        ('c / m', 'np.array([[1 / 1.0, 1 / 2.0, 1 / 3.0], [2 / 4.0, 2 / 5.0, 2 / 6.0]])'),
        ('r / m', 'np.array([[1 / 1.0, 2 / 2.0, 3 / 3.0], [1 / 4.0, 2 / 5.0, 3 / 6.0]])'),
        ('x / c', '1 / (c / x)'),
        ('x / r', '1 / (r / x)'),
        ('x / m', '1 / (m / x)'),
        ('c / m', '1 / (m / c)'),
        ('r / m', '1 / (m / r)'),
        ('x / nx', 'x / x'),
        ('x / nc', 'x / c'),
        ('x / nr', 'x / r'),
        ('x / nm', 'x / m'),
        ('c / nx', 'c / x'),
        ('c / nc', 'c / c'),
        ('c / nm', 'c / m'),
        ('r / nx', 'r / x'),
        ('r / nr', 'r / r'),
        ('r / nm', 'r / m'),
        ('m / nx', 'm / x'),
        ('m / nc', 'm / c'),
        ('m / nr', 'm / r'),
        ('m / nm', 'm / m')
        # other ops
        ,
        ('m ** 3', 'nm ** 3'),
        ('m.sqrt()', 'np.sqrt(nm)'),
        ('m.ceil()', 'np.ceil(nm)'),
        ('m.floor()', 'np.floor(nm)'),
        ('m.log()', 'np.log(nm)'),
        ('(m - 4).abs()', 'np.abs(nm - 4)'),
    ],
)
def test_block_matrix_elementwise_close_arithmetic(block_matrix_bindings, x, y):
    lhs = eval(x, block_matrix_bindings)
    rhs = eval(y, {'np': np}, block_matrix_bindings)
    _assert_close(lhs, rhs)


@pytest.mark.parametrize(
    'expr, expectation',
    [
        ('x + np.array([\'one\'], dtype=str)', pytest.raises(TypeError)),
        ('m @ m ', pytest.raises(ValueError)),
        ('m @ nm', pytest.raises(ValueError)),
    ],
)
def test_block_matrix_raises(block_matrix_bindings, expr, expectation):
    with expectation:
        eval(expr, {'np': np}, block_matrix_bindings)


@pytest.mark.parametrize(
    'x, y',
    [
        ('m.sum(axis=0).T', 'np.array([[5.0], [7.0], [9.0]])'),
        ('m.sum(axis=1).T', 'np.array([[6.0, 15.0]])'),
        ('m.sum(axis=0).T + row', 'np.array([[12.0, 13.0, 14.0],[14.0, 15.0, 16.0],[16.0, 17.0, 18.0]])'),
        ('m.sum(axis=0) + row.T', 'np.array([[12.0, 14.0, 16.0],[13.0, 15.0, 17.0],[14.0, 16.0, 18.0]])'),
        ('square.sum(axis=0).T + square.sum(axis=1)', 'np.array([[18.0], [30.0], [42.0]])'),
    ],
)
def test_matrix_sums(block_matrix_bindings, x, y):
    lhs = eval(x, block_matrix_bindings)
    rhs = eval(y, {'np': np}, block_matrix_bindings)
    _assert_eq(lhs, rhs)


@fails_service_backend()
@fails_local_backend()
@pytest.mark.parametrize(
    'x, y',
    [
        ('m.tree_matmul(m.T, splits=2)', 'nm @ nm.T'),
        ('m.tree_matmul(nm.T, splits=2)', 'nm @ nm.T'),
        ('row.tree_matmul(row.T, splits=2)', 'nrow @ nrow.T'),
        ('row.tree_matmul(nrow.T, splits=2)', 'nrow @ nrow.T'),
        ('m.T.tree_matmul(m, splits=2)', 'nm.T @ nm'),
        ('m.T.tree_matmul(nm, splits=2)', 'nm.T @ nm'),
        ('row.T.tree_matmul(row, splits=2)', 'nrow.T @ nrow'),
        ('row.T.tree_matmul(nrow, splits=2)', 'nrow.T @ nrow'),
    ],
)
def test_tree_matmul(block_matrix_bindings, x, y):
    lhs = eval(x, block_matrix_bindings)
    rhs = eval(y, block_matrix_bindings)
    _assert_eq(lhs, rhs)


@fails_service_backend()
@fails_local_backend()
@pytest.mark.parametrize(
    'nrows,ncols,block_size,split_size',
    [
        (nrows, ncols, block_size, split_size)
        for (nrows, ncols) in [(50, 60), (60, 25)]
        for block_size in [7, 10]
        for split_size in [2, 9]
    ],
)
def test_tree_matmul_splits(block_size, split_size, nrows, ncols):
    # Variety of block sizes and splits
    ndarray = np.arange(nrows * ncols).reshape((nrows, ncols))
    bm = BlockMatrix.from_numpy(ndarray, block_size)
    _assert_eq(bm.tree_matmul(bm.T, splits=split_size), ndarray @ ndarray.T)


def test_fill():
    nd = np.ones((3, 5))
    bm = BlockMatrix.fill(3, 5, 1.0)
    bm2 = BlockMatrix.fill(3, 5, 1.0, block_size=2)

    assert bm.block_size == BlockMatrix.default_block_size()
    assert bm2.block_size == 2
    _assert_eq(bm, nd)
    _assert_eq(bm2, nd)


def test_sum():
    nd = np.arange(11 * 13, dtype=np.float64).reshape((11, 13))
    bm = BlockMatrix.from_ndarray(hl.literal(nd), block_size=3)
    assert_sums_agree(bm, nd)


@fails_local_backend
@fails_service_backend(reason='ExecuteContext.scoped requires SparkBackend')
def test_sum_with_sparsify():
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

    assert_sums_agree(bm, nd)
    assert_sums_agree(bm2, nd)
    assert_sums_agree(bm3, nd)
    assert_sums_agree(bm4, nd4)


@pytest.mark.parametrize('indices', [(0, 0), (5, 7), (-3, 9), (-8, -10)])
def test_slicing_0(indices):
    nd = np.array(np.arange(0, 80, dtype=float)).reshape(8, 10)
    bm = BlockMatrix.from_ndarray(hl.literal(nd), block_size=3)
    _assert_eq(bm[indices], nd[indices])


@pytest.mark.parametrize(
    'indices',
    [
        (slice(0, 8), slice(0, 10)),
        (slice(0, 8, 2), slice(0, 10, 2)),
        (slice(2, 4), slice(5, 7)),
        (slice(-8, -1), slice(-10, -1)),
        (slice(-8, -1, 2), slice(-10, -1, 2)),
        (slice(None, 4, 1), slice(None, 4, 1)),
        (slice(4, None), slice(4, None)),
        (slice(None, None), slice(None, None)),
    ],
)
def test_slicing_1(indices):
    nd = np.array(np.arange(0, 80, dtype=float)).reshape(8, 10)
    bm = BlockMatrix.from_ndarray(hl.literal(nd), block_size=3)
    _assert_eq(bm[indices], nd[indices])
    _assert_eq(bm[indices][:, :2], nd[indices][:, :2])
    _assert_eq(bm[indices][:2, :], nd[indices][:2, :])


@pytest.mark.parametrize(
    'indices, axis',
    [
        ((0, slice(3, 4)), 0),
        ((1, slice(3, 4)), 0),
        ((-8, slice(3, 4)), 0),
        ((-1, slice(3, 4)), 0),
        ((slice(3, 4), 0), 1),
        ((slice(3, 4), 1), 1),
        ((slice(3, 4), -8), 1),
        ((slice(3, 4), -1), 1),
    ],
)
def test_slicing_2(indices, axis):
    nd = np.array(np.arange(0, 80, dtype=float)).reshape(8, 10)
    bm = BlockMatrix.from_ndarray(hl.literal(nd), block_size=3)
    _assert_eq(bm[indices], np.expand_dims(nd[indices], axis))
    _assert_eq(bm[indices] - bm, nd[indices] - nd)
    _assert_eq(bm - bm[indices], nd - nd[indices])


@pytest.mark.parametrize(
    'expr',
    [
        'square[0, ]',
        'square[9, 0]',
        'square[-9, 0]',
        'square[0, 11]',
        'square[0, -11]',
        'square[::-1, 0]',
        'square[0, ::-1]',
        'square[:0, 0]',
        'square[0, :0]',
        'square[0:9, 0]',
        'square[-9:, 0]',
        'square[:-9, 0]',
        'square[0, :11]',
        'square[0, -11:]',
        'square[0, :-11] ',
    ],
)
def test_block_matrix_illegal_indexing(block_matrix_bindings, expr):
    with pytest.raises(ValueError):
        eval(expr, block_matrix_bindings)


def test_diagonal_sparse():
    nd = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0],
        ]
    )
    bm = BlockMatrix.from_numpy(nd, block_size=2)
    bm = bm.sparsify_row_intervals([0, 0, 0, 0, 0], [2, 2, 2, 2, 2])

    # FIXME doesn't work in service, if test_is_sparse works, uncomment below
    # .assertTrue(bm.is_sparse)
    _assert_eq(bm.diagonal(), np.array([[1.0, 6.0, 0.0, 0.0]]))


@fails_service_backend()
@fails_local_backend()
def test_slices_with_sparsify():
    nd = np.array(np.arange(0, 80, dtype=float)).reshape(8, 10)
    bm = BlockMatrix.from_numpy(nd, block_size=3)
    bm2 = bm.sparsify_row_intervals([0, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0])
    assert bm2[0, 1] == 1.0
    assert bm2[0, 2] == 0.0
    assert bm2[0, 9] == 0.0

    nd2 = np.zeros(shape=(8, 10))
    nd2[0, 1] = 1.0

    _assert_eq(bm2[:, :], nd2)
    _assert_eq(bm2[:, 1], nd2[:, 1:2])
    _assert_eq(bm2[1, :], nd2[1:2, :])
    _assert_eq(bm2[0:5, 0:5], nd2[0:5, 0:5])


def test_sparsify_row_intervals_0():
    nd = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
    bm = BlockMatrix.from_numpy(nd, block_size=2)

    _assert_eq(
        bm.sparsify_row_intervals(starts=[1, 0, 2, 2], stops=[2, 0, 3, 4]),
        np.array([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 11.0, 0.0], [0.0, 0.0, 15.0, 16.0]]),
    )

    _assert_eq(
        bm.sparsify_row_intervals(starts=[1, 0, 2, 2], stops=[2, 0, 3, 4], blocks_only=True),
        np.array([[1.0, 2.0, 0.0, 0.0], [5.0, 6.0, 0.0, 0.0], [0.0, 0.0, 11.0, 12.0], [0.0, 0.0, 15.0, 16.0]]),
    )


@pytest.mark.parametrize(
    'starts, stops',
    [
        ([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]),
        ([0, 0, 5, 3, 4, 5, 8, 2], [9, 0, 5, 3, 4, 5, 9, 5]),
        ([0, 5, 10, 8, 7, 6, 5, 4], [0, 5, 10, 9, 8, 7, 6, 5]),
    ],
)
def test_row_intervals_1(starts, stops):
    nd2 = np.random.normal(size=(8, 10))
    bm2 = BlockMatrix.from_numpy(nd2, block_size=3)
    actual = bm2.sparsify_row_intervals(starts, stops, blocks_only=False).to_numpy()
    expected = nd2.copy()
    for i in range(0, 8):
        for j in range(0, starts[i]):
            expected[i, j] = 0.0
        for j in range(stops[i], 10):
            expected[i, j] = 0.0
    _assert_eq(actual, expected)


def test_sparsify_band_0():
    nd = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
    bm = BlockMatrix.from_numpy(nd, block_size=2)

    _assert_eq(
        bm.sparsify_band(lower=-1, upper=2),
        np.array([[1.0, 2.0, 3.0, 0.0], [5.0, 6.0, 7.0, 8.0], [0.0, 10.0, 11.0, 12.0], [0.0, 0.0, 15.0, 16.0]]),
    )

    _assert_eq(
        bm.sparsify_band(lower=0, upper=0, blocks_only=True),
        np.array([[1.0, 2.0, 0.0, 0.0], [5.0, 6.0, 0.0, 0.0], [0.0, 0.0, 11.0, 12.0], [0.0, 0.0, 15.0, 16.0]]),
    )


@pytest.mark.parametrize('lower, upper', [(0, 0), (1, 1), (2, 2), (-5, 5), (-7, 0), (0, 9), (-100, 100)])
def test_sparsify_band_1(lower, upper):
    nd2 = np.arange(0, 80, dtype=float).reshape(8, 10)
    bm2 = BlockMatrix.from_numpy(nd2, block_size=3)
    actual = bm2.sparsify_band(lower, upper, blocks_only=False).to_numpy()
    mask = np.fromfunction(lambda i, j: (lower <= j - i) * (j - i <= upper), (8, 10))
    _assert_eq(actual, nd2 * mask)


def test_sparsify_triangle():
    nd = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
    bm = BlockMatrix.from_numpy(nd, block_size=2)

    # FIXME doesn't work in service, if test_is_sparse works, uncomment below
    # assert not bm.is_sparse
    # assert bm.sparsify_triangle().is_sparse

    _assert_eq(
        bm.sparsify_triangle(),
        np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 6.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0], [0.0, 0.0, 0.0, 16.0]]),
    )

    _assert_eq(
        bm.sparsify_triangle(lower=True),
        np.array([[1.0, 0.0, 0.0, 0.0], [5.0, 6.0, 0.0, 0.0], [9.0, 10.0, 11.0, 0.0], [13.0, 14.0, 15.0, 16.0]]),
    )

    _assert_eq(
        bm.sparsify_triangle(blocks_only=True),
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0], [0.0, 0.0, 15.0, 16.0]]),
    )


def test_sparsify_rectangles():
    nd = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
    bm = BlockMatrix.from_numpy(nd, block_size=2)

    _assert_eq(
        bm.sparsify_rectangles([[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4]]),
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 0.0, 0.0], [13.0, 14.0, 0.0, 0.0]]),
    )

    _assert_eq(bm.sparsify_rectangles([]), np.zeros(shape=(4, 4)))


@fails_service_backend()
@fails_local_backend()
@pytest.mark.parametrize(
    'rects,block_size,binary',
    [
        (rects, block_size, binary)
        for binary in [False, True]
        for block_size in [3, 4, 10]
        for rects in [
            [[0, 1, 0, 1], [4, 5, 7, 8]],
            [[4, 5, 0, 10], [0, 8, 4, 5]],
            [
                [0, 1, 0, 1],
                [1, 2, 1, 2],
                [2, 3, 2, 3],
                [3, 5, 3, 6],
                [3, 6, 3, 7],
                [3, 7, 3, 8],
                [4, 5, 0, 10],
                [0, 8, 4, 5],
                [0, 8, 0, 10],
            ],
        ]
    ],
)
def test_export_rectangles(rects, block_size, binary):
    nd = np.arange(0, 80, dtype=float).reshape(8, 10)
    bm = BlockMatrix.from_numpy(nd, block_size=block_size)
    with hl.TemporaryDirectory() as rect_uri:
        bm.export_rectangles(rect_uri, rects, binary=binary)
        _assert_rectangles_eq(nd, rect_uri, rects, binary=binary)


@fails_service_backend()
@fails_local_backend()
def test_export_rectangles_sparse():
    with hl.TemporaryDirectory() as rect_uri:
        nd = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd, block_size=2)
        sparsify_rects = [[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4]]
        export_rects = [[0, 1, 0, 1], [0, 3, 0, 2], [1, 2, 0, 4], [2, 4, 2, 4]]
        bm.sparsify_rectangles(sparsify_rects).export_rectangles(rect_uri, export_rects)

        expected = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 0.0, 0.0], [13.0, 14.0, 0.0, 0.0]])

        _assert_rectangles_eq(expected, rect_uri, export_rects)


@fails_service_backend()
@fails_local_backend()
def test_export_rectangles_filtered():
    with hl.TemporaryDirectory() as rect_uri:
        nd = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
        bm = BlockMatrix.from_numpy(nd)
        bm = bm[1:3, 1:3]
        export_rects = [[0, 1, 0, 2], [1, 2, 0, 2]]
        bm.export_rectangles(rect_uri, export_rects)

        expected = np.array([[6.0, 7.0], [10.0, 11.0]])

        _assert_rectangles_eq(expected, rect_uri, export_rects)


@fails_service_backend()
@fails_local_backend()
def test_export_blocks():
    nd = np.ones(shape=(8, 10))
    bm = BlockMatrix.from_numpy(nd, block_size=20)

    with hl.TemporaryDirectory() as bm_uri:
        bm.export_blocks(bm_uri, binary=True)
        actual = BlockMatrix.rectangles_to_numpy(bm_uri, binary=True)
        _assert_eq(nd, actual)


@fails_service_backend()
@fails_local_backend()
@pytest.mark.parametrize('binary', [True, False])
def test_rectangles_to_numpy(binary):
    nd = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    rects = [[0, 3, 0, 1], [1, 2, 0, 2]]
    expected = np.array([[1.0, 0.0], [4.0, 5.0], [7.0, 0.0]])
    with hl.TemporaryDirectory() as rect_uri:
        BlockMatrix.from_numpy(nd).export_rectangles(rect_uri, rects, binary=binary)
        _assert_eq(expected, BlockMatrix.rectangles_to_numpy(rect_uri, binary=binary))


def test_to_ndarray():
    np_mat = np.arange(12).reshape((4, 3)).astype(np.float64)
    mat = BlockMatrix.from_ndarray(hl.nd.array(np_mat)).to_ndarray()
    assert np.array_equal(np_mat, hl.eval(mat))

    blocks_to_sparsify = [1, 4, 7, 12, 20, 42, 48]
    sparsed_numpy = sparsify_numpy(np.arange(25 * 25).reshape((25, 25)), 4, blocks_to_sparsify)
    sparsed = (
        BlockMatrix.from_ndarray(hl.nd.array(sparsed_numpy), block_size=4)
        ._sparsify_blocks(blocks_to_sparsify)
        .to_ndarray()
    )
    assert np.array_equal(sparsed_numpy, hl.eval(sparsed))


@test_timeout(batch=5 * 60)
@pytest.mark.parametrize('block_size', [1, 2, 1024])
def test_block_matrix_entries(block_size):
    n_rows, n_cols = 5, 3
    rows = [{'i': i, 'j': j, 'entry': float(i + j)} for i in range(n_rows) for j in range(n_cols)]
    schema = hl.tstruct(i=hl.tint32, j=hl.tint32, entry=hl.tfloat64)
    table = hl.Table.parallelize([hl.struct(i=row['i'], j=row['j'], entry=row['entry']) for row in rows], schema)
    table = table.annotate(i=hl.int64(table.i), j=hl.int64(table.j)).key_by('i', 'j')

    ndarray = np.reshape(list(map(lambda row: row['entry'], rows)), (n_rows, n_cols))

    block_matrix = BlockMatrix.from_ndarray(hl.literal(ndarray), block_size)
    entries_table = block_matrix.entries()
    assert entries_table.count() == n_cols * n_rows
    assert len(entries_table.row) == 3
    assert table._same(entries_table)


def test_from_entry_expr_filtered():
    mt = hl.utils.range_matrix_table(1, 1).filter_entries(False)
    bm = hl.linalg.BlockMatrix.from_entry_expr(mt.row_idx + mt.col_idx, mean_impute=True)  # should run without error
    assert np.isnan(bm.entries().entry.collect()[0])


def test_array_windows():
    def assert_eq(a, b):
        assert np.array_equal(a, np.array(b))

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
    assert starts.size == 0
    assert stops.size == 0

    starts, stops = hl.linalg.utils.array_windows(np.array([-float('inf'), -1, 0, 1, float("inf")]), 1)
    assert_eq(starts, [0, 1, 1, 2, 4])
    assert_eq(stops, [1, 3, 4, 4, 5])


@pytest.mark.parametrize(
    'array,radius',
    [
        ([1, 0], -1),
        ([0, float('nan')], 1),
        ([float('nan')], 1),
        ([0.0, float('nan')], 1),
        ([None], 1),
        ([], -1),
        (['str'], 1),
    ],
)
def test_array_windows_illegal_arguments(array, radius):
    with pytest.raises(ValueError):
        hl.linalg.utils.array_windows(np.array(array), radius)


def test_locus_windows_per_contig():
    f = hl._locus_windows_per_contig([[1.0, 3.0, 4.0], [2.0, 2.0], [5.0]], 1.0)
    assert hl.eval(f) == ([0, 1, 1, 3, 3, 5], [1, 3, 3, 5, 5, 6])


def test_locus_windows_1():
    centimorgans = hl.literal([0.1, 1.0, 1.0, 1.5, 1.9])

    mt = hl.balding_nichols_model(1, 5, 5).add_row_index()
    mt = mt.annotate_rows(cm=centimorgans[hl.int32(mt.row_idx)]).cache()

    starts, stops = hl.linalg.utils.locus_windows(mt.locus, 2)
    assert_np_arrays_eq(starts, [0, 0, 0, 1, 2])
    assert_np_arrays_eq(stops, [3, 4, 5, 5, 5])


def test_locus_windows_2():
    centimorgans = hl.literal([0.1, 1.0, 1.0, 1.5, 1.9])

    mt = hl.balding_nichols_model(1, 5, 5).add_row_index()
    mt = mt.annotate_rows(cm=centimorgans[hl.int32(mt.row_idx)]).cache()

    starts, stops = hl.linalg.utils.locus_windows(mt.locus, 0.5, coord_expr=mt.cm)
    assert_np_arrays_eq(starts, [0, 1, 1, 1, 3])
    assert_np_arrays_eq(stops, [1, 4, 4, 5, 5])


def test_locus_windows_3():
    centimorgans = hl.literal([0.1, 1.0, 1.0, 1.5, 1.9])

    mt = hl.balding_nichols_model(1, 5, 5).add_row_index()
    mt = mt.annotate_rows(cm=centimorgans[hl.int32(mt.row_idx)]).cache()

    starts, stops = hl.linalg.utils.locus_windows(mt.locus, 1.0, coord_expr=2 * centimorgans[hl.int32(mt.row_idx)])
    assert_np_arrays_eq(starts, [0, 1, 1, 1, 3])
    assert_np_arrays_eq(stops, [1, 4, 4, 5, 5])


def test_locus_windows_4():
    rows = [
        {'locus': hl.Locus('1', 1), 'cm': 1.0},
        {'locus': hl.Locus('1', 2), 'cm': 3.0},
        {'locus': hl.Locus('1', 4), 'cm': 4.0},
        {'locus': hl.Locus('2', 1), 'cm': 2.0},
        {'locus': hl.Locus('2', 1), 'cm': 2.0},
        {'locus': hl.Locus('3', 3), 'cm': 5.0},
    ]

    ht = hl.Table.parallelize(rows, hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64), key=['locus'])

    starts, stops = hl.linalg.utils.locus_windows(ht.locus, 1)
    assert_np_arrays_eq(starts, [0, 0, 2, 3, 3, 5])
    assert_np_arrays_eq(stops, [2, 2, 3, 5, 5, 6])


def dummy_table_with_loci_and_cms():
    rows = [
        {'locus': hl.Locus('1', 1), 'cm': 1.0},
        {'locus': hl.Locus('1', 2), 'cm': 3.0},
        {'locus': hl.Locus('1', 4), 'cm': 4.0},
        {'locus': hl.Locus('2', 1), 'cm': 2.0},
        {'locus': hl.Locus('2', 1), 'cm': 2.0},
        {'locus': hl.Locus('3', 3), 'cm': 5.0},
    ]

    return hl.Table.parallelize(rows, hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64), key=['locus'])


def test_locus_windows_5():
    ht = dummy_table_with_loci_and_cms()
    starts, stops = hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=ht.cm)
    assert_np_arrays_eq(starts, [0, 1, 1, 3, 3, 5])
    assert_np_arrays_eq(stops, [1, 3, 3, 5, 5, 6])


def test_locus_windows_6():
    ht = dummy_table_with_loci_and_cms()
    with pytest.raises(HailUserError, match='ascending order'):
        hl.linalg.utils.locus_windows(ht.order_by(ht.cm).locus, 1.0)


def test_locus_windows_7():
    ht = dummy_table_with_loci_and_cms()
    with pytest.raises(ExpressionException, match='different source'):
        hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=hl.utils.range_table(1).idx)


def test_locus_windows_8():
    with pytest.raises(ExpressionException, match='no source'):
        hl.linalg.utils.locus_windows(hl.locus('1', 1), 1.0)


def test_locus_windows_9():
    ht = dummy_table_with_loci_and_cms()
    with pytest.raises(ExpressionException, match='no source'):
        hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=0.0)


def test_locus_windows_10():
    ht = dummy_table_with_loci_and_cms()
    ht = ht.annotate_globals(x=hl.locus('1', 1), y=1.0)
    with pytest.raises(ExpressionException, match='row-indexed'):
        hl.linalg.utils.locus_windows(ht.x, 1.0)

    with pytest.raises(ExpressionException, match='row-indexed'):
        hl.linalg.utils.locus_windows(ht.locus, 1.0, ht.y)


def test_locus_windows_11():
    ht = hl.Table.parallelize(
        [{'locus': hl.missing(hl.tlocus()), 'cm': 1.0}],
        hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64),
        key=['locus'],
    )
    with pytest.raises(HailUserError, match='missing value for \'locus_expr\''):
        hl.linalg.utils.locus_windows(ht.locus, 1.0)

    with pytest.raises(HailUserError, match='missing value for \'locus_expr\''):
        hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=ht.cm)


def test_locus_windows_12():
    ht = hl.Table.parallelize(
        [{'locus': hl.Locus('1', 1), 'cm': hl.missing(hl.tfloat64)}],
        hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64),
        key=['locus'],
    )
    with pytest.raises(FatalError, match='missing value for \'coord_expr\''):
        hl.linalg.utils.locus_windows(ht.locus, 1.0, coord_expr=ht.cm)


def test_write_overwrite():
    with hl.TemporaryDirectory(ensure_exists=False) as path:
        bm = BlockMatrix.from_numpy(np.array([[0]]))
        bm.write(path)
        with pytest.raises(FatalError):
            bm.write(path)

        bm2 = BlockMatrix.from_numpy(np.array([[1]]))
        bm2.write(path, overwrite=True)
        _assert_eq(BlockMatrix.read(path), bm2)


def test_stage_locally():
    nd = np.arange(0, 80, dtype=float).reshape(8, 10)
    with hl.TemporaryDirectory(ensure_exists=False) as bm_uri:
        BlockMatrix.from_numpy(nd, block_size=3).write(bm_uri, stage_locally=True)
        bm = BlockMatrix.read(bm_uri)
        _assert_eq(nd, bm)


def test_svd():
    def assert_same_columns_up_to_sign(a, b):
        for j in range(a.shape[1]):
            assert np.allclose(a[:, j], b[:, j]) or np.allclose(-a[:, j], b[:, j])

    x0 = np.array([[-2.0, 0.0, 3.0], [-1.0, 2.0, 4.0]])
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
    _, _ = np.linalg.eigh(x0 @ x0.T)

    x = BlockMatrix.from_numpy(x0)
    s = x.svd(complexity_bound=0, compute_uv=False)
    assert np.all(s >= 0.0)

    s = x.svd(compute_uv=False, complexity_bound=0)
    assert np.all(s >= 0)


def test_filtering():
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
def test_is_sparse():
    block_list = [1, 2]
    np_square = np.arange(16, dtype=np.float64).reshape((4, 4))
    bm = BlockMatrix.from_numpy(np_square, block_size=2)
    bm = bm._sparsify_blocks(block_list)
    assert bm.is_sparse
    assert np.array_equal(bm.to_numpy(), np.array([[0, 0, 2, 3], [0, 0, 6, 7], [8, 9, 0, 0], [12, 13, 0, 0]]))


@pytest.mark.parametrize('block_list,nrows,ncols,block_size', [([1, 2], 4, 4, 2), ([4, 8, 10, 12, 13, 14], 15, 15, 4)])
def test_sparsify_blocks(block_list, nrows, ncols, block_size):
    np_square = np.arange(nrows * ncols, dtype=np.float64).reshape((nrows, ncols))
    bm = BlockMatrix.from_numpy(np_square, block_size=block_size)
    bm = bm._sparsify_blocks(block_list)
    sparse_numpy = sparsify_numpy(np_square, block_size, block_list)
    assert np.array_equal(bm.to_numpy(), sparse_numpy)


@pytest.mark.parametrize(
    'block_list,nrows,ncols,block_size',
    [
        ([1, 2], 4, 4, 2),
        ([4, 8, 10, 12, 13, 14], 15, 15, 4),
        ([2, 5, 8, 10, 11], 10, 15, 4),
        ([2, 5, 8, 10, 11], 15, 11, 4),
    ],
)
def test_sparse_transposition(block_list, nrows, ncols, block_size):
    np_square = np.arange(nrows * ncols, dtype=np.float64).reshape((nrows, ncols))
    bm = BlockMatrix.from_numpy(np_square, block_size=block_size)
    sparse_bm = bm._sparsify_blocks(block_list).T
    sparse_np = sparsify_numpy(np_square, block_size, block_list).T
    assert np.array_equal(sparse_bm.to_numpy(), sparse_np)


def test_row_blockmatrix_sum():
    row = BlockMatrix.from_numpy(np.arange(10))
    col = row.T

    # Summing vertically along a column vector to get a single value
    b = col.sum(axis=0)
    assert b.to_numpy().shape == (1, 1)

    # Summing horizontally along a row vector to create a single value
    d = row.sum(axis=1)
    assert d.to_numpy().shape == (1, 1)

    # Summing vertically along a row vector to make sure nothing changes
    e = row.sum(axis=0)
    assert e.to_numpy().shape == (1, 10)

    # Summing horizontally along a column vector to make sure nothing changes
    f = col.sum(axis=1)
    assert f.to_numpy().shape == (10, 1)


@fails_spark_backend()
def test_map():
    np_mat = np.arange(20, dtype=np.float64).reshape((4, 5))
    bm = BlockMatrix.from_ndarray(hl.nd.array(np_mat))
    bm_mapped_arith = bm._map_dense(lambda x: (x * x) + 5)
    _assert_eq(bm_mapped_arith, np_mat * np_mat + 5)

    bm_mapped_if = bm._map_dense(lambda x: hl.if_else(x >= 1, x, -8.0))
    np_if = np_mat.copy()
    np_if[0, 0] = -8.0
    _assert_eq(bm_mapped_if, np_if)


def test_from_entry_expr_simple_gt_n_alt_alleles():
    mt = hl.utils.range_matrix_table(346, 100)
    mt = mt.annotate_entries(x=(mt.row_idx * mt.col_idx) % 3)
    mt = mt.annotate_entries(GT=hl.unphased_diploid_gt_index_call(mt.x))
    expected = np.array([(x * y) % 3 for x in range(346) for y in range(100)]).reshape((346, 100))
    actual = hl.eval(BlockMatrix.from_entry_expr(hl.or_else(mt.GT.n_alt_alleles(), 0), block_size=32).to_ndarray())
    assert np.array_equal(expected, actual)


def test_from_entry_expr_simple_direct_from_field():
    mt = hl.utils.range_matrix_table(346, 100)
    mt = mt.annotate_entries(x=(mt.row_idx * mt.col_idx) % 3)

    expected = np.array([(x * y) % 3 for x in range(346) for y in range(100)]).reshape((346, 100))
    actual = hl.eval(BlockMatrix.from_entry_expr(mt.x, block_size=32).to_ndarray())
    assert np.array_equal(expected, actual)


def test_from_entry_expr_simple_with_float_conversion():
    mt = hl.utils.range_matrix_table(346, 100)
    mt = mt.annotate_entries(x=(mt.row_idx * mt.col_idx) % 3)

    expected = np.array([(x * y) % 3 for x in range(346) for y in range(100)]).reshape((346, 100))
    actual = hl.eval(BlockMatrix.from_entry_expr(hl.float64(mt.x), block_size=32).to_ndarray())
    assert np.array_equal(expected, actual)


def test_write_from_entry_expr_simple():
    mt = hl.utils.range_matrix_table(346, 100)
    mt = mt.annotate_entries(x=(mt.row_idx * mt.col_idx) % 3)

    expected = np.array([(x * y) % 3 for x in range(346) for y in range(100)]).reshape((346, 100))
    with hl.TemporaryDirectory(ensure_exists=False) as path:
        BlockMatrix.write_from_entry_expr(mt.x, path, block_size=32)
        actual = hl.eval(BlockMatrix.read(path).to_ndarray())
        assert np.array_equal(expected, actual)
