import hail as hl
from hail.utils.java import FatalError
import numpy as np
from ..helpers import *
import tempfile
import pytest

from hail.utils.java import FatalError


def assert_ndarrays(asserter, exprs_and_expecteds):
    exprs, expecteds = zip(*exprs_and_expecteds)

    expr_tuple = hl.tuple(exprs)
    evaled_exprs = hl.eval(expr_tuple)

    for (evaled, expected) in zip(evaled_exprs, expecteds):
        assert asserter(evaled, expected)


def assert_ndarrays_eq(*expr_and_expected):
    assert_ndarrays(np.array_equal, expr_and_expected)


def assert_ndarrays_almost_eq(*expr_and_expected):
    assert_ndarrays(np.allclose, expr_and_expected)

@skip_unless_spark_backend()
def test_ndarray_ref():

    scalar = 5.0
    np_scalar = np.array(scalar)
    h_scalar = hl._ndarray(scalar)
    h_np_scalar = hl._ndarray(np_scalar)

    assert_evals_to(h_scalar[()], 5.0)
    assert_evals_to(h_np_scalar[()], 5.0)

    cube = [[[0, 1],
             [2, 3]],
            [[4, 5],
             [6, 7]]]
    h_cube = hl._ndarray(cube)
    h_np_cube = hl._ndarray(np.array(cube))
    missing = hl._ndarray(hl.null(hl.tarray(hl.tint32)))

    assert_all_eval_to(
        (h_cube[0, 0, 1], 1),
        (h_cube[1, 1, 0], 6),
        (h_np_cube[0, 0, 1], 1),
        (h_np_cube[1, 1, 0], 6),
        (hl._ndarray([[[[1]]]])[0, 0, 0, 0], 1),
        (hl._ndarray([[[1, 2]], [[3, 4]]])[1, 0, 0], 3),
        (missing[1], None),
        (hl._ndarray([1, 2, 3])[hl.null(hl.tint32)], None),
        (h_cube[0, 0, hl.null(hl.tint32)], None)
    )

    with pytest.raises(FatalError) as exc:
        hl.eval(hl._ndarray([1, 2, 3])[4])
    assert "Index out of bounds" in str(exc)

@skip_unless_spark_backend()
@run_with_cxx_compile()
def test_ndarray_slice():
    np_arr = np.array([[[0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 10, 11]],
                       [[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]]])
    arr = hl._ndarray(np_arr)
    np_mat = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8]])
    mat = hl._ndarray(np_mat)

    assert_ndarrays_eq(
        (arr[:, :, :], np_arr[:, :, :]),
        (arr[:, :, 1], np_arr[:, :, 1]),
        (arr[:, :, 1:4:2], np_arr[:, :, 1:4:2]),
        (arr[:, 2, 1:4:2], np_arr[:, 2, 1:4:2]),
        (arr[0, 2, 1:4:2], np_arr[0, 2, 1:4:2]),
        (arr[0, :, 1:4:2] + arr[:, :1, 1:4:2], np_arr[0, :, 1:4:2] + np_arr[:, :1, 1:4:2]),
        (arr[0:, :, 1:4:2] + arr[:, :1, 1:4:2], np_arr[0:, :, 1:4:2] + np_arr[:, :1, 1:4:2]),
        (mat[0, 1:4:2] + mat[:, 1:4:2], np_mat[0, 1:4:2] + np_mat[:, 1:4:2]))

@skip_unless_spark_backend()
def test_ndarray_eval():
    data_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    nd_expr = hl._ndarray(data_list)
    evaled = hl.eval(nd_expr)
    np_equiv = np.array(data_list, dtype=np.int32)
    assert(np.array_equal(evaled, np_equiv))
    assert(evaled.strides == np_equiv.strides)

    assert hl.eval(hl._ndarray([[], []])).strides == (8, 8)
    assert np.array_equal(hl.eval(hl._ndarray([])), np.array([]))

    zero_array = np.zeros((10, 10), dtype=np.int64)
    evaled_zero_array = hl.eval(hl.literal(zero_array))

    assert np.array_equal(evaled_zero_array, zero_array)
    assert zero_array.dtype == evaled_zero_array.dtype

    # Testing from hail arrays
    assert np.array_equal(hl.eval(hl._ndarray(hl.range(6))), np.arange(6))
    assert np.array_equal(hl.eval(hl._ndarray(hl.int64(4))), np.array(4))

    # Testing missing data
    assert hl.eval(hl._ndarray(hl.null(hl.tarray(hl.tint32)))) is None

    with pytest.raises(ValueError) as exc:
        hl._ndarray([[4], [1, 2, 3], 5])
    assert "inner dimensions do not match" in str(exc.value)


@skip_unless_spark_backend()
def test_ndarray_shape():
    np_e = np.array(3)
    np_row = np.array([1, 2, 3])
    np_col = np.array([[1], [2], [3]])
    np_m = np.array([[1, 2], [3, 4]])
    np_nd = np.arange(30).reshape((2, 5, 3))

    e = hl._ndarray(np_e)
    row = hl._ndarray(np_row)
    col = hl._ndarray(np_col)
    m = hl._ndarray(np_m)
    nd = hl._ndarray(np_nd)
    missing = hl._ndarray(hl.null(hl.tarray(hl.tint32)))

    assert_all_eval_to(
        (e.shape, np_e.shape),
        (row.shape, np_row.shape),
        (col.shape, np_col.shape),
        (m.shape, np_m.shape),
        (nd.shape, np_nd.shape),
        ((row + nd).shape, (np_row + np_nd).shape),
        ((row + col).shape, (np_row + np_col).shape),
        (m.transpose().shape, np_m.transpose().shape),
        (missing.shape, None)
    )

@skip_unless_spark_backend()
def test_ndarray_reshape():
    np_single = np.array([8])
    single = hl._ndarray([8])

    np_zero_dim = np.array(4)
    zero_dim = hl._ndarray(4)

    np_a = np.array([1, 2, 3, 4, 5, 6])
    a = hl._ndarray(np_a)

    np_cube = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape((2, 2, 2))
    cube = hl._ndarray([0, 1, 2, 3, 4, 5, 6, 7]).reshape((2, 2, 2))
    cube_to_rect = cube.reshape((2, 4))
    np_cube_to_rect = np_cube.reshape((2, 4))
    cube_t_to_rect = cube.transpose((1, 0, 2)).reshape((2, 4))
    np_cube_t_to_rect = np_cube.transpose((1, 0, 2)).reshape((2, 4))

    np_hypercube = np.arange(3 * 5 * 7 * 9).reshape((3, 5, 7, 9))
    hypercube = hl._ndarray(np_hypercube)

    assert_ndarrays_eq(
        (single.reshape(()), np_single.reshape(())),
        (zero_dim.reshape(()), np_zero_dim.reshape(())),
        (zero_dim.reshape((1,)), np_zero_dim.reshape((1,))),
        (a.reshape((6,)), np_a.reshape((6,))),
        (a.reshape((2, 3)), np_a.reshape((2, 3))),
        (a.reshape((3, 2)), np_a.reshape((3, 2))),
        (a.reshape((3, -1)), np_a.reshape((3, -1))),
        (a.reshape((-1, 2)), np_a.reshape((-1, 2))),
        (cube_to_rect, np_cube_to_rect),
        (cube_t_to_rect, np_cube_t_to_rect),
        (hypercube.reshape((5, 7, 9, 3)).reshape((7, 9, 3, 5)), np_hypercube.reshape((7, 9, 3, 5))),
        (hypercube.reshape(hl.tuple([5, 7, 9, 3])), np_hypercube.reshape((5, 7, 9, 3)))
    )

    assert hl.eval(hl.null(hl.tndarray(hl.tfloat, 2)).reshape((4, 5))) is None
    assert hl.eval(hl._ndarray(hl.range(20)).reshape(hl.null(hl.ttuple(hl.tint64, hl.tint64)))) is None

    with pytest.raises(FatalError) as exc:
        hl.eval(hl.literal(np_cube).reshape((-1, -1)))
    assert "more than one -1" in str(exc)

    with pytest.raises(FatalError) as exc:
        hl.eval(hl.literal(np_cube).reshape((20,)))
    assert "requested shape is incompatible with number of elements" in str(exc)

    with pytest.raises(FatalError) as exc:
        hl.eval(a.reshape((3,)))
    assert "requested shape is incompatible with number of elements" in str(exc)

    with pytest.raises(FatalError) as exc:
        hl.eval(a.reshape(()))
    assert "requested shape is incompatible with number of elements" in str(exc)

    with pytest.raises(FatalError) as exc:
        hl.eval(hl.literal(np_cube).reshape((0, 2, 2)))
    assert "must contain only positive numbers or -1" in str(exc)

    with pytest.raises(FatalError) as exc:
        hl.eval(hl.literal(np_cube).reshape((2, 2, -2)))
    assert "must contain only positive numbers or -1" in str(exc)


@skip_unless_spark_backend()
def test_ndarray_map():
    a = hl._ndarray([[2, 3, 4], [5, 6, 7]])
    b = hl.map(lambda x: -x, a)
    c = hl.map(lambda x: True, a)

    assert_ndarrays_eq(
        (b, [[-2, -3, -4], [-5, -6, -7]]),
        (c, [[True, True, True],
             [True, True, True]]))

    assert hl.eval(hl.null(hl.tndarray(hl.tfloat, 1)).map(lambda x: x * 2)) is None

@skip_unless_spark_backend()
def test_ndarray_map2():

    a = 2.0
    b = 3.0
    x = np.array([a, b])
    y = np.array([b, a])
    row_vec = np.array([[1, 2]])
    cube1 = np.array([[[1, 2],
                       [3, 4]],
                      [[5, 6],
                       [7, 8]]])
    cube2 = np.array([[[9, 10],
                       [11, 12]],
                      [[13, 14],
                       [15, 16]]])

    na = hl._ndarray(a)
    nx = hl._ndarray(x)
    ny = hl._ndarray(y)
    nrow_vec = hl._ndarray(row_vec)
    ncube1 = hl._ndarray(cube1)
    ncube2 = hl._ndarray(cube2)

    assert_ndarrays_eq(
        # with lists/numerics
        (na + b, np.array(a + b)),
        (b + na, np.array(a + b)),
        (nx + y, x + y),
        (ncube1 + cube2, cube1 + cube2),

        # Addition
        (na + na, np.array(a + a)),
        (nx + ny, x + y),
        (ncube1 + ncube2, cube1 + cube2),
        # Broadcasting
        (ncube1 + na, cube1 + a),
        (na + ncube1, a + cube1),
        (ncube1 + ny, cube1 + y),
        (ny + ncube1, y + cube1),
        (nrow_vec + ncube1, row_vec + cube1),
        (ncube1 + nrow_vec, cube1 + row_vec),

        # Subtraction
        (na - na, np.array(a - a)),
        (nx - nx, x - x),
        (ncube1 - ncube2, cube1 - cube2),
        # Broadcasting
        (ncube1 - na, cube1 - a),
        (na - ncube1, a - cube1),
        (ncube1 - ny, cube1 - y),
        (ny - ncube1, y - cube1),
        (ncube1 - nrow_vec, cube1 - row_vec),
        (nrow_vec - ncube1, row_vec - cube1),

        # Multiplication
        (na * na, np.array(a * a)),
        (nx * nx, x * x),
        (nx * na, x * a),
        (na * nx, a * x),
        (ncube1 * ncube2, cube1 * cube2),
        # Broadcasting
        (ncube1 * na, cube1 * a),
        (na * ncube1, a * cube1),
        (ncube1 * ny, cube1 * y),
        (ny * ncube1, y * cube1),
        (ncube1 * nrow_vec, cube1 * row_vec),
        (nrow_vec * ncube1, row_vec * cube1),

        # Floor div
        (na // na, np.array(a // a)),
        (nx // nx, x // x),
        (nx // na, x // a),
        (na // nx, a // x),
        (ncube1 // ncube2, cube1 // cube2),
        # Broadcasting
        (ncube1 // na, cube1 // a),
        (na // ncube1, a // cube1),
        (ncube1 // ny, cube1 // y),
        (ny // ncube1, y // cube1),
        (ncube1 // nrow_vec, cube1 // row_vec),
        (nrow_vec // ncube1, row_vec // cube1))

    # Division
    assert_ndarrays_almost_eq(
        (na / na, np.array(a / a)),
        (nx / nx, x / x),
        (nx / na, x / a),
        (na / nx, a / x),
        (ncube1 / ncube2, cube1 / cube2),
        # Broadcasting
        (ncube1 / na, cube1 / a),
        (na / ncube1, a / cube1),
        (ncube1 / ny, cube1 / y),
        (ny / ncube1, y / cube1),
        (ncube1 / nrow_vec, cube1 / row_vec),
        (nrow_vec / ncube1, row_vec / cube1))

    # Missingness tests
    missing = hl.null(hl.tndarray(hl.tfloat64, 2))
    present = hl._ndarray(np.arange(10).reshape(5, 2))

    assert hl.eval(missing + missing) is None
    assert hl.eval(missing + present) is None
    assert hl.eval(present + missing) is None

@skip_unless_spark_backend()
@run_with_cxx_compile()
def test_ndarray_to_numpy():
    nd = np.array([[1, 2, 3], [4, 5, 6]])
    np.array_equal(hl._ndarray(nd).to_numpy(), nd)

@skip_unless_spark_backend()
@run_with_cxx_compile()
def test_ndarray_save():
    arrs = [
        np.array([[[1, 2, 3], [4, 5, 6]],
                  [[7, 8, 9], [10, 11, 12]]], dtype=np.int32),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        np.array(3.0, dtype=np.float32),
        np.array([3.0], dtype=np.float64),
        np.array([True, False, True, True])
    ]

    for expected in arrs:
        with tempfile.NamedTemporaryFile(suffix='.npy') as f:
            hl._ndarray(expected).save(f.name)
            actual = np.load(f.name)

            assert(expected.dtype == actual.dtype, f'expected: {expected.dtype}, actual: {actual.dtype}')
            assert(np.array_equal(expected, actual))

@skip_unless_spark_backend()
@run_with_cxx_compile()
def test_ndarray_sum():
    np_m = np.array([[1, 2], [3, 4]])
    m = hl._ndarray(np_m)

    assert_all_eval_to(
        (m.sum(axis=0), np_m.sum(axis=0)),
        (m.sum(axis=1), np_m.sum(axis=1)),
        (m.sum(), np_m.sum()))

@skip_unless_spark_backend()
def test_ndarray_transpose():
    np_v = np.array([1, 2, 3])
    np_m = np.array([[1, 2, 3], [4, 5, 6]])
    np_cube = np.array([[[1, 2],
                         [3, 4]],
                        [[5, 6],
                         [7, 8]]])
    v = hl._ndarray(np_v)
    m = hl._ndarray(np_m)
    cube = hl._ndarray(np_cube)

    assert_ndarrays_eq(
        (v.T, np_v.T),
        (v.T, np_v),
        (m.T, np_m.T),
        (cube.transpose((0, 2, 1)), np_cube.transpose((0, 2, 1))),
        (cube.T, np_cube.T))

    assert hl.eval(hl.null(hl.tndarray(hl.tfloat, 1)).T) is None

    with pytest.raises(ValueError) as exc:
        v.transpose((1,))
    assert "Invalid axis: 1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        cube.transpose((1, 1))
    assert "Expected 3 axes, got 2" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        cube.transpose((1, 1, 1))
    assert "Axes cannot contain duplicates" in str(exc.value)

@skip_unless_spark_backend()
def test_ndarray_matmul():
    np_v = np.array([1, 2])
    np_m = np.array([[1, 2], [3, 4]])
    np_r = np.array([[1, 2, 3], [4, 5, 6]])
    np_cube = np.arange(8).reshape((2, 2, 2))
    np_rect_prism = np.arange(12).reshape((3, 2, 2))
    np_broadcasted_mat = np.arange(4).reshape((1, 2, 2))
    np_six_dim_tensor = np.arange(3 * 7 * 1 * 9 * 4 * 5).reshape((3, 7, 1, 9, 4, 5))
    np_five_dim_tensor = np.arange(7 * 5 * 1 * 5 * 3).reshape((7, 5, 1, 5, 3))

    v = hl._ndarray(np_v)
    m = hl._ndarray(np_m)
    r = hl._ndarray(np_r)
    cube = hl._ndarray(np_cube)
    rect_prism = hl._ndarray(np_rect_prism)
    broadcasted_mat = hl._ndarray(np_broadcasted_mat)
    six_dim_tensor = hl._ndarray(np_six_dim_tensor)
    five_dim_tensor = hl._ndarray(np_five_dim_tensor)

    assert_ndarrays_eq(
        (v @ v, np_v @ np_v),
        (m @ m, np_m @ np_m),
        (m @ m.T, np_m @ np_m.T),
        (r @ r.T, np_r @ np_r.T),
        (v @ m, np_v @ np_m),
        (m @ v, np_m @ np_v),
        (cube @ cube, np_cube @ np_cube),
        (cube @ v, np_cube @ np_v),
        (v @ cube, np_v @ np_cube),
        (cube @ m, np_cube @ np_m),
        (m @ cube, np_m @ np_cube),
        (rect_prism @ m, np_rect_prism @ np_m),
        (m @ rect_prism, np_m @ np_rect_prism),
        (m @ rect_prism.T, np_m @ np_rect_prism.T),
        (broadcasted_mat @ rect_prism, np_broadcasted_mat @ np_rect_prism),
        (six_dim_tensor @ five_dim_tensor, np_six_dim_tensor @ np_five_dim_tensor)
    )

    assert hl.eval(hl.null(hl.tndarray(hl.tfloat64, 2)) @ hl.null(hl.tndarray(hl.tfloat64, 2))) is None
    assert hl.eval(hl.null(hl.tndarray(hl.tint64, 2)) @ hl._ndarray(np.arange(10).reshape(5, 2))) is None
    assert hl.eval(hl._ndarray(np.arange(10).reshape(5, 2)) @ hl.null(hl.tndarray(hl.tint64, 2))) is None

    with pytest.raises(ValueError):
        m @ 5

    with pytest.raises(ValueError):
        m @ hl._ndarray(5)

    with pytest.raises(ValueError):
        cube @ hl._ndarray(5)

    with pytest.raises(FatalError) as exc:
        hl.eval(r @ r)
    assert "Matrix dimensions incompatible: 3 2" in str(exc)

    with pytest.raises(FatalError) as exc:
        hl.eval(hl._ndarray([1, 2]) @ hl._ndarray([1, 2, 3]))
    assert "Matrix dimensions incompatible" in str(exc)

@skip_unless_spark_backend()
def test_ndarray_big():
    assert hl.eval(hl._ndarray(hl.range(100_000))).size == 100_000

@skip_unless_spark_backend()
def test_ndarray_full():
    assert_ndarrays_eq(
        (hl._nd.zeros(4), np.zeros(4)),
        (hl._nd.zeros((3, 4, 5)), np.zeros((3, 4, 5))),
        (hl._nd.ones(6), np.ones(6)),
        (hl._nd.ones((6, 6, 6)), np.ones((6, 6, 6))),
        (hl._nd.full(7, 9), np.full(7, 9)),
        (hl._nd.full((3, 4, 5), 9), np.full((3, 4, 5), 9))
    )

@skip_unless_spark_backend()
def test_ndarray_mixed():
    assert hl.eval(hl.null(hl.tndarray(hl.tint64, 2)).map(lambda x: x * x).reshape((4, 5)).T) is None
    assert hl.eval(
        (hl._nd.zeros((5, 10)).map(lambda x: x - 2) +
         hl._nd.ones((5, 10)).map(lambda x: x + 5)).reshape(hl.null(hl.ttuple(hl.tint64, hl.tint64))).T.reshape((10, 5))) is None
    assert hl.eval(hl.or_missing(False, hl._ndarray(np.arange(10)).reshape((5,2)).map(lambda x: x * 2)).map(lambda y: y * 2)) is None
