import math
import numpy as np
import re
from ..helpers import *
import pytest

from hail.utils.java import FatalError, HailUserError

def assert_ndarrays(asserter, exprs_and_expecteds):
    exprs, expecteds = zip(*exprs_and_expecteds)

    expr_tuple = hl.tuple(exprs)
    evaled_exprs = hl.eval(expr_tuple)

    evaled_and_expected = zip(evaled_exprs, expecteds)
    for (idx, (evaled, expected)) in enumerate(evaled_and_expected):
        assert asserter(evaled, expected), f"NDArray comparison {idx} failed, got: {evaled}, expected: {expected}"


def assert_ndarrays_eq(*expr_and_expected):
    assert_ndarrays(np.array_equal, expr_and_expected)


def assert_ndarrays_almost_eq(*expr_and_expected):
    assert_ndarrays(np.allclose, expr_and_expected)


def test_ndarray_ref():

    scalar = 5.0
    np_scalar = np.array(scalar)
    h_scalar = hl.nd.array(scalar)
    h_np_scalar = hl.nd.array(np_scalar)

    assert_evals_to(h_scalar[()], 5.0)
    assert_evals_to(h_np_scalar[()], 5.0)

    cube = [[[0, 1],
             [2, 3]],
            [[4, 5],
             [6, 7]]]
    h_cube = hl.nd.array(cube)
    h_np_cube = hl.nd.array(np.array(cube))
    missing = hl.nd.array(hl.missing(hl.tarray(hl.tint32)))

    assert_all_eval_to(
        (h_cube[0, 0, 1], 1),
        (h_cube[1, 1, 0], 6),
        (h_np_cube[0, 0, 1], 1),
        (h_np_cube[1, 1, 0], 6),
        (hl.nd.array([[[[1]]]])[0, 0, 0, 0], 1),
        (hl.nd.array([[[1, 2]], [[3, 4]]])[1, 0, 0], 3),
        (missing[1], None),
        (hl.nd.array([1, 2, 3])[hl.missing(hl.tint32)], None),
        (h_cube[0, 0, hl.missing(hl.tint32)], None)
    )


def test_ndarray_ref_bounds_check():
    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([1, 2, 3])[4])
    assert "Index 4 is out of bounds for axis 0 with size 3" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([1, 2, 3])[-1])
    assert "Index -1 is out of bounds for axis 0 with size 3" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([1, 2, 3])[-4])
    assert "Index -4 is out of bounds for axis 0 with size 3" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([[1], [2], [3]])[4, :])
    assert "Index 4 is out of bounds for axis 0 with size 3" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([[1], [2], [3]])[-4, :])
    assert "Index -4 is out of bounds for axis 0 with size 3" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([[1], [2], [3]])[:, 4])
    assert "Index 4 is out of bounds for axis 1 with size 1" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([[1], [2], [3]])[:, -4])
    assert "Index -4 is out of bounds for axis 1 with size 1" in str(exc.value)


def test_ndarray_slice():
    np_rect_prism = np.arange(24).reshape((2, 3, 4))
    rect_prism = hl.nd.array(np_rect_prism)
    np_mat = np.arange(8).reshape((2, 4))
    mat = hl.nd.array(np_mat)
    np_flat = np.arange(20)
    flat = hl.nd.array(np_flat)
    a = [0, 1]
    an = np.array(a)
    ah = hl.nd.array(a)
    ae_np = np.arange(4*4*5*6*5*4).reshape((4, 4, 5, 6, 5, 4))
    ae = hl.nd.array(ae_np)
    assert_ndarrays_eq(
        (rect_prism[:, :, :], np_rect_prism[:, :, :]),
        (rect_prism[:, :, 1], np_rect_prism[:, :, 1]),
        (rect_prism[0:1, 1:3, 0:2], np_rect_prism[0:1, 1:3, 0:2]),
        (rect_prism[:, :, 1:4:2], np_rect_prism[:, :, 1:4:2]),
        (rect_prism[:, 2, 1:4:2], np_rect_prism[:, 2, 1:4:2]),
        (rect_prism[0, 2, 1:4:2], np_rect_prism[0, 2, 1:4:2]),
        (rect_prism[0, :, 1:4:2] + rect_prism[:, :1, 1:4:2],
         np_rect_prism[0, :, 1:4:2] + np_rect_prism[:, :1, 1:4:2]),
        (rect_prism[0:, :, 1:4:2] + rect_prism[:, :1, 1:4:2],
         np_rect_prism[0:, :, 1:4:2] + np_rect_prism[:, :1, 1:4:2]),
        (rect_prism[0, 0, -3:-1], np_rect_prism[0, 0, -3:-1]),
        (rect_prism[-1, 0:1, 3:0:-1], np_rect_prism[-1, 0:1, 3:0:-1]),
        # partial indexing
        (rect_prism[1], np_rect_prism[1]),
        (rect_prism[1:2], np_rect_prism[1:2]),
        (rect_prism[1:2:2], np_rect_prism[1:2:2]),
        (rect_prism[1, 2], np_rect_prism[1, 2]),
        (rect_prism[-1, 1:2:2], np_rect_prism[-1, 1:2:2]),
        # ellipses inclusion
        (rect_prism[...], np_rect_prism[...]),
        (rect_prism[1, ...], np_rect_prism[1, ...]),
        (rect_prism[..., 1], np_rect_prism[..., 1]),
        # np.newaxis inclusion
        (rect_prism[hl.nd.newaxis, :, :], np_rect_prism[np.newaxis, :, :]),
        (rect_prism[hl.nd.newaxis], np_rect_prism[np.newaxis]),
        (rect_prism[hl.nd.newaxis, np.newaxis, np.newaxis], np_rect_prism[np.newaxis, np.newaxis, np.newaxis]),
        (rect_prism[hl.nd.newaxis, np.newaxis, 1:4:2], np_rect_prism[np.newaxis, np.newaxis, 1:4:2]),
        (rect_prism[1, :, hl.nd.newaxis], np_rect_prism[1, :, np.newaxis]),
        (rect_prism[1, hl.nd.newaxis, 1], np_rect_prism[1, np.newaxis, 1]),
        (rect_prism[..., hl.nd.newaxis, 1], np_rect_prism[..., np.newaxis, 1]),
    )
    assert_ndarrays_eq(
        (flat[15:5:-1], np_flat[15:5:-1]),
        (flat[::-1], np_flat[::-1]),
        (flat[::22], np_flat[::22]),
        (flat[::-22], np_flat[::-22]),
        (flat[15:5], np_flat[15:5]),
        (flat[3:12:-1], np_flat[3:12:-1]),
        (flat[12:3:1], np_flat[12:3:1]),
        (flat[4:1:-2], np_flat[4:1:-2]),
        (flat[0:0:1], np_flat[0:0:1]),
        (flat[-4:-1:2], np_flat[-4:-1:2]),
        # ellipses inclusion
        (flat[...], np_flat[...]),


        (mat[::-1, :], np_mat[::-1, :]),
        (mat[0, 1:4:2] + mat[:, 1:4:2], np_mat[0, 1:4:2] + np_mat[:, 1:4:2]),
        (mat[-1:4:1, 0], np_mat[-1:4:1, 0]),
        (mat[-1:4:-1, 0], np_mat[-1:4:-1, 0]),
        # out of bounds on start
        (mat[9:2:-1, 1:4], np_mat[9:2:-1, 1:4]),
        (mat[9:-1:-1, 1:4], np_mat[9:-1:-1, 1:4]),
        (mat[-5::, 0], np_mat[-5::, 0]),
        (mat[-5::-1, 0], np_mat[-5::-1, 0]),
        (mat[-5:-1:-1, 0], np_mat[-5:-1:-1, 0]),
        (mat[-5:-5:-1, 0], np_mat[-5:-5:-1, 0]),
        (mat[4::, 0], np_mat[4::, 0]),
        (mat[4:-1:, 0], np_mat[4:-1:, 0]),
        (mat[4:-1:-1, 0], np_mat[4:-1:-1, 0]),
        (mat[5::, 0], np_mat[5::, 0]),
        (mat[5::-1, 0], np_mat[5::-1, 0]),
        (mat[-5::-1, 0], np_mat[-5::-1, 0]),
        (mat[-5::1, 0], np_mat[-5::1, 0]),
        (mat[5:-1:-1, 0], np_mat[5:-1:-1, 0]),
        (mat[5:-5:-1, 0], np_mat[5:-5:-1, 0]),
        # out of bounds on stop
        (mat[0:20, 0:17], np_mat[0:20, 0:17]),
        (mat[0:20, 2:17], np_mat[0:20, 2:17]),
        (mat[:4, 0], np_mat[:4, 0]),
        (mat[:4:-1, 0], np_mat[:4:-1, 0]),
        (mat[:-5, 0], np_mat[:-5, 0]),
        (mat[:-5:-1, 0], np_mat[:-5:-1, 0]),
        (mat[0:-5, 0], np_mat[0:-5, 0]),
        (mat[0:-5:-1, 0], np_mat[0:-5:-1, 0]),
        # partial indexing
        (mat[1], np_mat[1]),
        (mat[0:1], np_mat[0:1]),
        # ellipses inclusion
        (mat[...], np_mat[...]),

        (ah[:-3:1], an[:-3:1]),
        (ah[:-3:-1], an[:-3:-1]),
        (ah[-3::-1], an[-3::-1]),
        (ah[-3::1], an[-3::1]),

        # ellipses inclusion
        (ae[..., 3], ae_np[..., 3]),
        (ae[3, ...], ae_np[3, ...]),
        (ae[2, 3, 1:2:2, ...], ae_np[2, 3, 1:2:2, ...]),
        (ae[3, 2, 3, ..., 2], ae_np[3, 2, 3, ..., 2]),
        (ae[3, 2, 2, ..., 2, 1:2:2], ae_np[3, 2, 2, ..., 2, 1:2:2]),
        (ae[3, :, hl.nd.newaxis, ..., :, hl.nd.newaxis, 2], ae_np[3, :, np.newaxis, ..., :, np.newaxis, 2])
    )

    assert hl.eval(flat[hl.missing(hl.tint32):4:1]) is None
    assert hl.eval(flat[4:hl.missing(hl.tint32)]) is None
    assert hl.eval(flat[4:10:hl.missing(hl.tint32)]) is None
    assert hl.eval(rect_prism[:, :, 0:hl.missing(hl.tint32):1]) is None
    assert hl.eval(rect_prism[hl.missing(hl.tint32), :, :]) is None

    with pytest.raises(HailUserError, match="Slice step cannot be zero"):
        hl.eval(flat[::0])

    with pytest.raises(HailUserError, match="Index 3 is out of bounds for axis 0 with size 2"):
        hl.eval(mat[3, 1:3])

    with pytest.raises(HailUserError, match="Index -4 is out of bounds for axis 0 with size 2"):
        hl.eval(mat[-4, 0:3])

    with pytest.raises(IndexError, match="an index can only have a single ellipsis"):
        hl.eval(rect_prism[..., ...])

    with pytest.raises(IndexError, match="too many indices for array: array is 3-dimensional, but 4 were indexed"):
        hl.eval(rect_prism[1, 1, 1, 1])


def test_ndarray_transposed_slice():
    a = hl.nd.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    np_a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    aT = a.T
    np_aT = np_a.T
    assert_ndarrays_eq(
        (a, np_a),
        (aT[0:aT.shape[0], 0:5], np_aT[0:np_aT.shape[0], 0:5])
    )


def test_ndarray_eval():
    data_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mishapen_data_list1 = [[4], [1, 2, 3]]
    mishapen_data_list2 = [[[1], [2, 3]]]
    mishapen_data_list3 = [[4], [1, 2, 3], 5]

    nd_expr = hl.nd.array(data_list)
    evaled = hl.eval(nd_expr)
    np_equiv = np.array(data_list, dtype=np.int32)
    np_equiv_fortran_style = np.asfortranarray(np_equiv)
    np_equiv_extra_dimension = np_equiv.reshape((3, 1, 3))
    assert(np.array_equal(evaled, np_equiv))

    assert np.array_equal(hl.eval(hl.nd.array([])), np.array([]))

    zero_array = np.zeros((10, 10), dtype=np.int64)
    evaled_zero_array = hl.eval(hl.literal(zero_array))

    assert np.array_equal(evaled_zero_array, zero_array)
    assert zero_array.dtype == evaled_zero_array.dtype

    # Testing correct interpretation of numpy strides
    assert np.array_equal(hl.eval(hl.literal(np_equiv_fortran_style)), np_equiv_fortran_style)
    assert np.array_equal(hl.eval(hl.literal(np_equiv_extra_dimension)), np_equiv_extra_dimension)

    # Testing from hail arrays
    assert np.array_equal(hl.eval(hl.nd.array(hl.range(6))), np.arange(6))
    assert np.array_equal(hl.eval(hl.nd.array(hl.int64(4))), np.array(4))

    # Testing from nested hail arrays
    assert np.array_equal(
        hl.eval(hl.nd.array(hl.array([hl.array(x) for x in data_list]))), np.arange(9).reshape((3, 3)) + 1)

    # Testing missing data
    assert hl.eval(hl.nd.array(hl.missing(hl.tarray(hl.tint32)))) is None

    with pytest.raises(ValueError) as exc:
        hl.nd.array(mishapen_data_list1)
    assert "inner dimensions do not match" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array(hl.array(mishapen_data_list1)))
    assert "inner dimensions do not match" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array(hl.array(mishapen_data_list2)))
    assert "inner dimensions do not match" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        hl.nd.array(mishapen_data_list3)
    assert "inner dimensions do not match" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([1, hl.missing(hl.tint32), 3]))


def test_ndarray_shape():
    np_e = np.array(3)
    np_row = np.array([1, 2, 3])
    np_col = np.array([[1], [2], [3]])
    np_m = np.array([[1, 2], [3, 4]])
    np_nd = np.arange(30).reshape((2, 5, 3))

    e = hl.nd.array(np_e)
    row = hl.nd.array(np_row)
    col = hl.nd.array(np_col)
    m = hl.nd.array(np_m)
    nd = hl.nd.array(np_nd)
    missing = hl.nd.array(hl.missing(hl.tarray(hl.tint32)))

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


def test_ndarray_reshape():
    np_single = np.array([8])
    single = hl.nd.array([8])

    np_zero_dim = np.array(4)
    zero_dim = hl.nd.array(4)

    np_a = np.array([1, 2, 3, 4, 5, 6])
    a = hl.nd.array(np_a)

    np_cube = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape((2, 2, 2))
    cube = hl.nd.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape((2, 2, 2))
    cube_to_rect = cube.reshape((2, 4))
    np_cube_to_rect = np_cube.reshape((2, 4))
    cube_t_to_rect = cube.transpose((1, 0, 2)).reshape((2, 4))
    np_cube_t_to_rect = np_cube.transpose((1, 0, 2)).reshape((2, 4))

    np_hypercube = np.arange(3 * 5 * 7 * 9).reshape((3, 5, 7, 9))
    hypercube = hl.nd.array(np_hypercube)

    np_shape_zero = np.array([])
    shape_zero = hl.nd.array(np_shape_zero)

    assert_ndarrays_eq(
        (single.reshape(()), np_single.reshape(())),
        (zero_dim.reshape(()), np_zero_dim.reshape(())),
        (zero_dim.reshape((1,)), np_zero_dim.reshape((1,))),
        (a.reshape((6,)), np_a.reshape((6,))),
        (a.reshape((2, 3)), np_a.reshape((2, 3))),
        (a.reshape(2, 3), np_a.reshape(2, 3)),
        (a.reshape((3, 2)), np_a.reshape((3, 2))),
        (a.reshape((3, -1)), np_a.reshape((3, -1))),
        (a.reshape((-1, 2)), np_a.reshape((-1, 2))),
        (cube_to_rect, np_cube_to_rect),
        (cube_t_to_rect, np_cube_t_to_rect),
        (hypercube.reshape((5, 7, 9, 3)).reshape((7, 9, 3, 5)), np_hypercube.reshape((7, 9, 3, 5))),
        (hypercube.reshape(hl.tuple([5, 7, 9, 3])), np_hypercube.reshape((5, 7, 9, 3))),
        (shape_zero.reshape((0, 5)), np_shape_zero.reshape((0, 5))),
        (shape_zero.reshape((-1, 5)), np_shape_zero.reshape((-1, 5)))
    )

    assert hl.eval(hl.missing(hl.tndarray(hl.tfloat, 2)).reshape((4, 5))) is None
    assert hl.eval(hl.nd.array(hl.range(20)).reshape(
        hl.missing(hl.ttuple(hl.tint64, hl.tint64)))) is None

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.literal(np_cube).reshape((-1, -1)))
    assert "more than one -1" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.literal(np_cube).reshape((20,)))
    assert "requested shape is incompatible with number of elements" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(a.reshape((3,)))
    assert "requested shape is incompatible with number of elements" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(a.reshape(()))
    assert "requested shape is incompatible with number of elements" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.literal(np_cube).reshape((0, 2, 2)))
    assert "requested shape is incompatible with number of elements" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.literal(np_cube).reshape((2, 2, -2)))
    assert "must contain only nonnegative numbers or -1" in str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(shape_zero.reshape((0, -1)))
    assert "Can't reshape" in str(exc.value)

    with pytest.raises(TypeError):
        a.reshape(hl.tuple(['4', '5']))


def test_ndarray_map1():
    a = hl.nd.array([[2, 3, 4], [5, 6, 7]])
    b = hl.map(lambda x: -x, a)
    b2 = b.map(lambda x: x * x)
    c = hl.map(lambda x: True, a)

    assert_ndarrays_eq(
        (b, [[-2, -3, -4], [-5, -6, -7]]),
        (b2, [[4, 9, 16], [25, 36, 49]]),
        (c, [[True, True, True],
             [True, True, True]]))

    assert hl.eval(hl.missing(hl.tndarray(hl.tfloat, 1)).map(lambda x: x * 2)) is None

    # NDArrays don't correctly support elements that contain pointers at the moment.
    # s = hl.nd.array(["hail", "is", "great"])
    # s_lens = s.map(lambda e: hl.len(e))
    # assert np.array_equal(hl.eval(s_lens), np.array([4, 2, 5]))

    structs = hl.nd.array([hl.struct(x=5, y=True), hl.struct(x=9, y=False)])
    assert np.array_equal(hl.eval(structs.map(lambda e: e.y)), np.array([True, False]))


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

    na = hl.nd.array(a)
    nx = hl.nd.array(x)
    ny = hl.nd.array(y)
    nrow_vec = hl.nd.array(row_vec)
    ncube1 = hl.nd.array(cube1)
    ncube2 = hl.nd.array(cube2)

    assert_ndarrays_eq(
        # with lists/numerics
        (na + b, np.array(a + b)),
        (b + na, np.array(a + b)),
        (nx + y, x + y),
        (ncube1 + cube2, cube1 + cube2),
        (na + na, np.array(a + a)),
        (nx + ny, x + y),
        (ncube1 + ncube2, cube1 + cube2),
        (nx.map2(y, lambda c, d: c+d), x + y),
        (ncube1.map2(cube2, lambda c, d: c+d), cube1 + cube2),
        # Broadcasting
        (ncube1 + na, cube1 + a),
        (na + ncube1, a + cube1),
        (ncube1 + ny, cube1 + y),
        (ny + ncube1, y + cube1),
        (nrow_vec + ncube1, row_vec + cube1),
        (ncube1 + nrow_vec, cube1 + row_vec),
        (ncube1.map2(na, lambda c, d: c+d), cube1 + a),
        (nrow_vec.map2(ncube1, lambda c, d: c+d), row_vec + cube1),


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
        (nrow_vec // ncube1, row_vec // cube1)
    )

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
    missing = hl.missing(hl.tndarray(hl.tfloat64, 2))
    present = hl.nd.array(np.arange(10).reshape(5, 2))

    assert hl.eval(missing + missing) is None
    assert hl.eval(missing + present) is None
    assert hl.eval(present + missing) is None

def test_ndarray_sum():
    np_m = np.array([[1, 2], [3, 4]])
    m = hl.nd.array(np_m)

    assert_ndarrays_eq(
        (m.sum(axis=0), np_m.sum(axis=0)),
        (m.sum(axis=1), np_m.sum(axis=1)),
        (m.sum(tuple([])), np_m.sum(tuple([]))))

    assert hl.eval(m.sum()) == 10
    assert hl.eval(m.sum((0, 1))) == 10

    bool_nd = hl.nd.array([[True, False, True], [False, True, True]])
    assert hl.eval(bool_nd.sum()) == 4

    with pytest.raises(ValueError) as exc:
        m.sum(3)
    assert "out of bounds for ndarray of dimension 2" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        m.sum((1, 1))
    assert "duplicate" in str(exc.value)


def test_ndarray_transpose():
    np_v = np.array([1, 2, 3])
    np_m = np.array([[1, 2, 3], [4, 5, 6]])
    np_cube = np.array([[[1, 2],
                         [3, 4]],
                        [[5, 6],
                         [7, 8]]])
    v = hl.nd.array(np_v)
    m = hl.nd.array(np_m)
    cube = hl.nd.array(np_cube)

    assert_ndarrays_eq(
        (v.T, np_v.T),
        (v.T, np_v),
        (m.T, np_m.T),
        (cube.transpose((0, 2, 1)), np_cube.transpose((0, 2, 1))),
        (cube.T, np_cube.T))

    assert hl.eval(hl.missing(hl.tndarray(hl.tfloat, 1)).T) is None

    with pytest.raises(ValueError) as exc:
        v.transpose((1,))
    assert "Invalid axis: 1" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        cube.transpose((1, 1))
    assert "Expected 3 axes, got 2" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        cube.transpose((1, 1, 1))
    assert "Axes cannot contain duplicates" in str(exc.value)

def test_ndarray_matmul():
    np_v = np.array([1, 2])
    np_y = np.array([1, 1, 1])
    np_m = np.array([[1, 2], [3, 4]])
    np_m_f32 = np_m.astype(np.float32)
    np_m_f64 = np_m.astype(np.float64)
    np_r = np.array([[1, 2, 3], [4, 5, 6]])
    np_r_f32 = np_r.astype(np.float32)
    np_r_f64 = np_r.astype(np.float64)
    np_cube = np.arange(8).reshape((2, 2, 2))
    np_rect_prism = np.arange(12).reshape((3, 2, 2))
    np_broadcasted_mat = np.arange(4).reshape((1, 2, 2))
    np_six_dim_tensor = np.arange(3 * 7 * 1 * 9 * 4 * 5).reshape((3, 7, 1, 9, 4, 5))
    np_five_dim_tensor = np.arange(7 * 5 * 1 * 5 * 3).reshape((7, 5, 1, 5, 3))
    np_ones_int32 = np.ones((4, 4), dtype=np.int32)
    np_ones_float64 = np.ones((4, 4), dtype=np.float64)
    np_zero_by_four = np.array([], dtype=np.float64).reshape((0, 4))

    v = hl.nd.array(np_v)
    y = hl.nd.array(np_y)
    m = hl.nd.array(np_m)
    m_f32 = hl.nd.array(np_m_f32)
    m_f64 = hl.nd.array(np_m_f64)
    r = hl.nd.array(np_r)
    r_f32 = hl.nd.array(np_r_f32)
    r_f64 = hl.nd.array(np_r_f64)
    cube = hl.nd.array(np_cube)
    rect_prism = hl.nd.array(np_rect_prism)
    broadcasted_mat = hl.nd.array(np_broadcasted_mat)
    six_dim_tensor = hl.nd.array(np_six_dim_tensor)
    five_dim_tensor = hl.nd.array(np_five_dim_tensor)
    ones_int32 = hl.nd.array(np_ones_int32)
    ones_float64 = hl.nd.array(np_ones_float64)
    zero_by_four = hl.nd.array(np_zero_by_four)

    assert_ndarrays_eq(
        (v @ v, np_v @ np_v),
        (m @ m, np_m @ np_m),
        (m_f32 @ m_f32, np_m_f32 @ np_m_f32),
        (m_f64 @ m_f64, np_m_f64 @ np_m_f64),
        (m @ m.T, np_m @ np_m.T),
        (m_f64 @ m_f64.T, np_m_f64 @ np_m_f64.T),
        (r @ r.T, np_r @ np_r.T),
        (r_f32 @ r_f32.T, np_r_f32 @ np_r_f32.T),
        (r_f64 @ r_f64.T, np_r_f64 @ np_r_f64.T),
        (v @ m, np_v @ np_m),
        (m @ v, np_m @ np_v),
        (v @ r, np_v @ np_r),
        (r @ y, np_r @ np_y),
        (cube @ cube, np_cube @ np_cube),
        (cube @ v, np_cube @ np_v),
        (v @ cube, np_v @ np_cube),
        (cube @ m, np_cube @ np_m),
        (m @ cube, np_m @ np_cube),
        (rect_prism @ m, np_rect_prism @ np_m),
        (m @ rect_prism, np_m @ np_rect_prism),
        (m @ rect_prism.T, np_m @ np_rect_prism.T),
        (broadcasted_mat @ rect_prism, np_broadcasted_mat @ np_rect_prism),
        (six_dim_tensor @ five_dim_tensor, np_six_dim_tensor @ np_five_dim_tensor),
        (zero_by_four @ ones_float64, np_zero_by_four, np_ones_float64),
        (zero_by_four.transpose() @ zero_by_four, np_zero_by_four.transpose() @ np_zero_by_four)
    )

    assert hl.eval(hl.missing(hl.tndarray(hl.tfloat64, 2)) @
                   hl.missing(hl.tndarray(hl.tfloat64, 2))) is None
    assert hl.eval(hl.missing(hl.tndarray(hl.tint64, 2)) @
                   hl.nd.array(np.arange(10).reshape(5, 2))) is None
    assert hl.eval(hl.nd.array(np.arange(10).reshape(5, 2)) @
                   hl.missing(hl.tndarray(hl.tint64, 2))) is None

    assert np.array_equal(hl.eval(ones_int32 @ ones_float64), np_ones_int32 @ np_ones_float64)

    with pytest.raises(ValueError):
        m @ 5

    with pytest.raises(ValueError):
        m @ hl.nd.array(5)

    with pytest.raises(ValueError):
        cube @ hl.nd.array(5)

    with pytest.raises(HailUserError) as exc:
        hl.eval(r @ r)
    assert "Matrix dimensions incompatible: (2, 3) can't be multiplied by matrix with dimensions (2, 3)" in str(exc.value), str(exc.value)

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.array([1, 2]) @ hl.nd.array([1, 2, 3]))
    assert "Matrix dimensions incompatible" in str(exc.value)

def test_ndarray_matmul_dgemv():
    np_mat_3_4 = np.arange(12, dtype=np.float64).reshape((3, 4))
    np_mat_4_3 = np.arange(12, dtype=np.float64).reshape((4, 3))
    np_vec_3 = np.array([4, 2, 7], dtype=np.float64)
    np_vec_4 = np.array([9, 17, 3, 1], dtype=np.float64)

    mat_3_4 = hl.nd.array(np_mat_3_4)
    mat_4_3 = hl.nd.array(np_mat_4_3)
    vec_3 = hl.nd.array(np_vec_3)
    vec_4 = hl.nd.array(np_vec_4)

    assert_ndarrays_eq(
        (mat_3_4 @ vec_4, np_mat_3_4 @ np_vec_4),
        (mat_4_3 @ vec_3, np_mat_4_3 @ np_vec_3),
        (mat_3_4.T @ vec_3, np_mat_3_4.T @ np_vec_3)
    )

def test_ndarray_big():
    assert hl.eval(hl.nd.array(hl.range(100_000))).size == 100_000


def test_ndarray_full():
    assert_ndarrays_eq(
        (hl.nd.zeros(4), np.zeros(4)),
        (hl.nd.zeros((3, 4, 5)), np.zeros((3, 4, 5))),
        (hl.nd.ones(6), np.ones(6)),
        (hl.nd.ones((6, 6, 6)), np.ones((6, 6, 6))),
        (hl.nd.full(7, 9), np.full(7, 9)),
        (hl.nd.full((3, 4, 5), 9), np.full((3, 4, 5), 9))
    )

    assert hl.eval(hl.nd.zeros((5, 5), dtype=hl.tfloat32)).dtype, np.float32
    assert hl.eval(hl.nd.ones(3, dtype=hl.tint64)).dtype, np.int64
    assert hl.eval(hl.nd.full((5, 6, 7), hl.int32(3), dtype=hl.tfloat64)).dtype, np.float64


def test_ndarray_arange():
    assert_ndarrays_eq(
        (hl.nd.arange(40), np.arange(40)),
        (hl.nd.arange(5, 50), np.arange(5, 50)),
        (hl.nd.arange(2, 47, 13), np.arange(2, 47, 13))
    )

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.arange(5, 20, 0))
    assert "Array range cannot have step size 0" in str(exc.value)


def test_ndarray_mixed():
    assert hl.eval(hl.missing(hl.tndarray(hl.tint64, 2)).map(
        lambda x: x * x).reshape((4, 5)).T) is None
    assert hl.eval(
        (hl.nd.zeros((5, 10)).map(lambda x: x - 2) +
         hl.nd.ones((5, 10)).map(lambda x: x + 5)).reshape(hl.missing(hl.ttuple(hl.tint64, hl.tint64))).T.reshape((10, 5))) is None
    assert hl.eval(hl.or_missing(False, hl.nd.array(np.arange(10)).reshape(
        (5, 2)).map(lambda x: x * 2)).map(lambda y: y * 2)) is None


def test_ndarray_show():
    hl.nd.array(3).show()
    hl.nd.arange(6).show()
    hl.nd.arange(6).reshape((2, 3)).show()
    hl.nd.arange(8).reshape((2, 2, 2)).show()


def test_ndarray_diagonal():
    assert np.array_equal(hl.eval(hl.nd.diagonal(hl.nd.array([[1, 2], [3, 4]]))), np.array([1, 4]))
    assert np.array_equal(hl.eval(hl.nd.diagonal(
        hl.nd.array([[1, 2, 3], [4, 5, 6]]))), np.array([1, 5]))
    assert np.array_equal(hl.eval(hl.nd.diagonal(
        hl.nd.array([[1, 2], [3, 4], [5, 6]]))), np.array([1, 4]))

    with pytest.raises(AssertionError) as exc:
        hl.nd.diagonal(hl.nd.array([1, 2]))
    assert "2 dimensional" in str(exc.value)


def test_ndarray_solve_triangular():
    a = hl.nd.array([[1, 1], [0, 1]])
    b = hl.nd.array([2, 1])
    b2 = hl.nd.array([[11, 5], [6, 3]])

    a_low = hl.nd.array([[4, 0], [2, 1]])
    b_low = hl.nd.array([4, 5])

    a_sing = hl.nd.array([[0, 1], [0, 1]])
    b_sing = hl.nd.array([2, 2])

    assert np.allclose(hl.eval(hl.nd.solve_triangular(a, b)), np.array([1., 1.]))
    assert np.allclose(hl.eval(hl.nd.solve_triangular(a, b2)), np.array([[5., 2.], [6., 3.]]))
    assert np.allclose(hl.eval(hl.nd.solve_triangular(a_low, b_low, True)), np.array([[1., 3.]]))
    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.solve_triangular(a_sing, b_sing))
    assert "singular" in str(exc.value), str(exc.value)

def test_ndarray_solve():
    a = hl.nd.array([[1, 2], [3, 5]])
    b = hl.nd.array([1, 2])
    b2 = hl.nd.array([[1, 8], [2, 12]])

    assert np.allclose(hl.eval(hl.nd.solve(a, b)), np.array([-1., 1.]))
    assert np.allclose(hl.eval(hl.nd.solve(a, b2)), np.array([[-1., -16.], [1, 12]]))
    assert np.allclose(hl.eval(hl.nd.solve(a.T, b2.T)), np.array([[19., 26.], [-6, -8]]))

    with pytest.raises(HailUserError) as exc:
        hl.eval(hl.nd.solve(hl.nd.array([[1, 2], [1, 2]]), hl.nd.array([8, 10])))
    assert "singular" in str(exc.value), str(exc.value)


def test_ndarray_qr():
    def assert_raw_equivalence(hl_ndarray, np_ndarray):
        ndarray_h, ndarray_tau = hl.eval(hl.nd.qr(hl_ndarray, mode="raw"))
        np_ndarray_h, np_ndarray_tau = np.linalg.qr(np_ndarray, mode="raw")

        # Can't ask for the rank of something that has a 0 in its shape.
        if 0 in np_ndarray.shape:
            assert ndarray_h.shape == np_ndarray_h.shape
            assert ndarray_tau.shape == np_ndarray_tau.shape
        else:
            rank = np.linalg.matrix_rank(np_ndarray)

            assert np.allclose(ndarray_h[:, :rank], np_ndarray_h[:, :rank])
            assert np.allclose(ndarray_tau[:rank], np_ndarray_tau[:rank])

    def assert_r_equivalence(hl_ndarray, np_ndarray):
        assert np.allclose(hl.eval(hl.nd.qr(hl_ndarray, mode="r")),
                           np.linalg.qr(np_ndarray, mode="r"))

    def assert_reduced_equivalence(hl_ndarray, np_ndarray):
        q, r = hl.eval(hl.nd.qr(hl_ndarray, mode="reduced"))
        nq, nr = np.linalg.qr(np_ndarray, mode="reduced")

        # Can't ask for the rank of something that has a 0 in its shape.
        if 0 in np_ndarray.shape:
            assert q.shape == nq.shape
            assert r.shape == nr.shape
        else:
            rank = np.linalg.matrix_rank(np_ndarray)

            assert np.allclose(q[:, :rank], nq[:, :rank])
            assert np.allclose(r, nr)
            assert np.allclose(q @ r, np_ndarray)

    def assert_complete_equivalence(hl_ndarray, np_ndarray):
        q, r = hl.eval(hl.nd.qr(hl_ndarray, mode="complete"))
        nq, nr = np.linalg.qr(np_ndarray, mode="complete")

        # Can't ask for the rank of something that has a 0 in its shape.
        if 0 in np_ndarray.shape:
            assert q.shape == nq.shape
            assert r.shape == nr.shape
        else:
            rank = np.linalg.matrix_rank(np_ndarray)

            assert np.allclose(q[:, :rank], nq[:, :rank])
            assert np.allclose(r, nr)
            assert np.allclose(q @ r, np_ndarray)

    def assert_same_qr(hl_ndarray, np_ndarray):
        assert_raw_equivalence(hl_ndarray, np_ndarray)
        assert_r_equivalence(hl_ndarray, np_ndarray)
        assert_reduced_equivalence(hl_ndarray, np_ndarray)
        assert_complete_equivalence(hl_ndarray, np_ndarray)

    np_identity4 = np.identity(4)
    identity4 = hl.nd.array(np_identity4)

    assert_same_qr(identity4, np_identity4)

    np_size_zero_n = np.zeros((10, 0))
    size_zero_n = hl.nd.zeros((10, 0))

    assert_same_qr(size_zero_n, np_size_zero_n)

    np_size_zero_m = np.zeros((0, 10))
    size_zero_m = hl.nd.zeros((0, 10))

    assert_same_qr(size_zero_m, np_size_zero_m)

    np_all3 = np.full((3, 3), 3)
    all3 = hl.nd.full((3, 3), 3)

    assert_same_qr(all3, np_all3)

    np_nine_square = np.arange(9).reshape((3, 3))
    nine_square = hl.nd.arange(9).reshape((3, 3))

    assert_same_qr(nine_square, np_nine_square)

    np_wiki_example = np.array([[12, -51, 4],
                                [6, 167, -68],
                                [-4, 24, -41]])
    wiki_example = hl.nd.array(np_wiki_example)

    assert_same_qr(wiki_example, np_wiki_example)

    np_wide_rect = np.arange(12).reshape((3, 4))
    wide_rect = hl.nd.arange(12).reshape((3, 4))

    assert_same_qr(wide_rect, np_wide_rect)

    np_tall_rect = np.arange(12).reshape((4, 3))
    tall_rect = hl.nd.arange(12).reshape((4, 3))

    assert_same_qr(tall_rect, np_tall_rect)

    np_single_element = np.array([1]).reshape((1, 1))
    single_element = hl.nd.array([1]).reshape((1, 1))

    assert_same_qr(single_element, np_single_element)

    np_no_elements = np.array([]).reshape((0, 10))
    no_elements = hl.nd.array(np_no_elements)

    assert_same_qr(no_elements, np_no_elements)

    with pytest.raises(ValueError) as exc:
        hl.nd.qr(wiki_example, mode="invalid")
    assert "Unrecognized mode" in str(exc.value)

    with pytest.raises(AssertionError) as exc:
        hl.nd.qr(hl.nd.arange(6))
    assert "requires 2 dimensional" in str(exc.value)


def test_svd():
    def assert_evals_to_same_svd(nd_expr, np_array, full_matrices=True, compute_uv=True):
        evaled = hl.eval(hl.nd.svd(nd_expr, full_matrices, compute_uv))
        np_svd = np.linalg.svd(np_array, full_matrices, compute_uv)

        # check shapes
        for h, n in zip(evaled, np_svd):
            assert h.shape == n.shape

        k = min(np_array.shape)
        rank = np.linalg.matrix_rank(np_array)

        if compute_uv:
            hu, hs, hv = evaled
            nu, ns, nv = np_svd

            # Singular values match
            np.testing.assert_array_almost_equal(hs, ns)

            # U is orthonormal
            uut = hu.T @ hu
            np.testing.assert_array_almost_equal(uut, np.identity(uut.shape[0]))

            # V is orthonormal
            vvt = hv @ hv.T
            np.testing.assert_array_almost_equal(vvt, np.identity(vvt.shape[0]))

            # Multiplying together gets back to original
            smat = np.zeros(np_array.shape) if full_matrices else np.zeros((k, k))
            smat[:k, :k] = np.diag(hs)
            np.testing.assert_array_almost_equal(hu @ smat @ hv, np_array)

        else:
            np.testing.assert_array_almost_equal(evaled, np_svd)

    np_small_square = np.arange(4).reshape((2, 2))
    small_square = hl.nd.array(np_small_square)
    np_rank_2_wide_rectangle = np.arange(12).reshape((4, 3))
    rank_2_wide_rectangle = hl.nd.array(np_rank_2_wide_rectangle)
    np_rank_2_tall_rectangle = np_rank_2_wide_rectangle.T
    rank_2_tall_rectangle = hl.nd.array(np_rank_2_tall_rectangle)

    assert_evals_to_same_svd(small_square, np_small_square)
    assert_evals_to_same_svd(small_square, np_small_square, compute_uv=False)

    assert_evals_to_same_svd(rank_2_wide_rectangle, np_rank_2_wide_rectangle)
    assert_evals_to_same_svd(rank_2_wide_rectangle, np_rank_2_wide_rectangle, full_matrices=False)

    assert_evals_to_same_svd(rank_2_tall_rectangle, np_rank_2_tall_rectangle)
    assert_evals_to_same_svd(rank_2_tall_rectangle, np_rank_2_tall_rectangle, full_matrices=False)


def test_eigh():
    def assert_evals_to_same_eig(nd_expr, np_array, eigvals_only=True):
        evaled = hl.eval(hl.nd.eigh(nd_expr, eigvals_only))
        np_eig = np.linalg.eigvalsh(np_array)

        # check shapes
        for h, n in zip(evaled, np_eig):
            assert h.shape == n.shape

        if eigvals_only:
            np.testing.assert_array_almost_equal(evaled, np_eig)
        else:
            he, hv = evaled

            # eigvals match
            np.testing.assert_array_almost_equal(he, np_eig)

            # V is orthonormal
            vvt = hv @ hv.T
            np.testing.assert_array_almost_equal(vvt, np.identity(vvt.shape[0]))

            # V is eigenvectors
            np.testing.assert_array_almost_equal(np_array @ hv, hv * he)

    A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
    hA = hl.nd.array(A)

    assert_evals_to_same_eig(hA, A)
    assert_evals_to_same_eig(hA, A, eigvals_only=True)


def test_numpy_interop():
    v = [2, 3]
    w = [3, 5]
    a = [[2, 3]]
    b = [[3], [5]]

    assert np.array_equal(hl.eval(np.array(v) * hl.literal(3)), np.array([6, 9]))
    assert np.array_equal(hl.eval(hl.literal(3) * np.array(v)), np.array([6, 9]))

    assert np.array_equal(hl.eval(np.array(v) * hl.nd.array(w)), np.array([6, 15]))
    assert np.array_equal(hl.eval(hl.nd.array(w) * np.array(v)), np.array([6, 15]))

    assert np.array_equal(hl.eval(np.array(v) + hl.literal(3)), np.array([5, 6]))
    assert np.array_equal(hl.eval(hl.literal(3) + np.array(v)), np.array([5, 6]))

    assert np.array_equal(hl.eval(np.array(v) + hl.nd.array(w)), np.array([5, 8]))
    assert np.array_equal(hl.eval(hl.nd.array(w) + np.array(v)), np.array([5, 8]))

    assert np.array_equal(hl.eval(np.array(v) @ hl.nd.array(w)), 21)
    assert np.array_equal(hl.eval(hl.nd.array(v) @ np.array(w)), 21)

    assert np.array_equal(hl.eval(np.array(a) @ hl.nd.array(b)), np.array([[21]]))
    assert np.array_equal(hl.eval(hl.nd.array(a) @ np.array(b)), np.array([[21]]))

    assert np.array_equal(hl.eval(hl.nd.array(b) @ np.array(a)),
                          np.array([[6, 9], [10, 15]]))
    assert np.array_equal(hl.eval(np.array(b) @ hl.nd.array(a)),
                          np.array([[6, 9], [10, 15]]))


def test_ndarray_emitter_extract():
    np_mat = np.array([0, 1, 2, 1, 0])
    mat = hl.nd.array(np_mat)
    mapped_mat = mat.map(lambda x: hl.array([3, 4, 5])[hl.int(x)])
    assert hl.eval(hl.range(5).map(lambda i: mapped_mat[i])) == [3, 4, 5, 4, 3]


def test_ndarrays_transmute_ops():
    u = hl.utils.range_table(10, n_partitions=10)
    u = u.annotate(x=hl.nd.array([u.idx]), y=hl.nd.array([u.idx]))
    u = u.transmute(xxx=u.x @ u.y)
    assert u.xxx.collect() == [x * x for x in range(10)]


def test_ndarray():
    a1 = hl.eval(hl.nd.array((1, 2, 3)))
    a2 = hl.eval(hl.nd.array([1, 2, 3]))
    an1 = np.array((1, 2, 3))
    an2 = np.array([1, 2, 3])

    assert(np.array_equal(a1, a2) and np.array_equal(a2, an2))

    a1 = hl.eval(hl.nd.array(((1), (2), (3))))
    a2 = hl.eval(hl.nd.array(([1], [2], [3])))
    a3 = hl.eval(hl.nd.array([[1], [2], [3]]))

    an1 = np.array(((1), (2), (3)))
    an2 = np.array(([1], [2], [3]))
    an3 = np.array([[1], [2], [3]])

    assert(np.array_equal(a1, an1) and np.array_equal(a2, an2) and np.array_equal(a3, an3))

    a1 = hl.eval(hl.nd.array(((1, 2), (2, 5), (3, 8))))
    a2 = hl.eval(hl.nd.array([[1, 2], [2, 5], [3, 8]]))

    an1 = np.array(((1, 2), (2, 5), (3, 8)))
    an2 = np.array([[1, 2], [2, 5], [3, 8]])

    assert(np.array_equal(a1, an1) and np.array_equal(a2, an2))


def test_cast():
    def testequal(a, hdtype, ndtype):
        ah = hl.eval(hl.nd.array(a, dtype=hdtype))
        an = np.array(a, dtype=ndtype)

        assert(ah.dtype == an.dtype)

    def test(a):
        testequal(a, hl.tfloat64, np.float64)
        testequal(a, hl.tfloat32, np.float32)
        testequal(a, hl.tint32, np.int32)
        testequal(a, hl.tint64, np.int64)

    test([1, 2, 3])
    test([1, 2, 3.])
    test([1., 2., 3.])
    test([[1, 2], [3, 4]])


def test_inv():
    c = np.random.randn(5, 5)
    d = np.linalg.inv(c)
    dhail = hl.eval(hl.nd.inv(c))
    assert np.allclose(dhail, d)


def test_concatenate():
    x = np.array([[1., 2.], [3., 4.]])
    y = np.array([[5.], [6.]])
    np_res = np.concatenate([x, y], axis=1)

    res = hl.eval(hl.nd.concatenate([x, y], axis=1))
    assert np.array_equal(np_res, res)

    res = hl.eval(hl.nd.concatenate(hl.array([x, y]), axis=1))
    assert np.array_equal(np_res, res)

    x = np.array([[1], [3]])
    y = np.array([[5], [6]])

    seq = [x, y]
    seq2 = hl.array(seq)
    np_res = np.concatenate(seq)
    res = hl.eval(hl.nd.concatenate(seq))
    assert np.array_equal(np_res, res)

    res = hl.eval(hl.nd.concatenate(seq2))
    assert np.array_equal(np_res, res)

    seq = (x, y)
    seq2 = hl.array([x, y])
    np_res = np.concatenate(seq)
    res = hl.eval(hl.nd.concatenate(seq))
    assert np.array_equal(np_res, res)

    res = hl.eval(hl.nd.concatenate(seq2))
    assert np.array_equal(np_res, res)


def test_concatenate_differing_shapes():
    with pytest.raises(ValueError, match='hl.nd.concatenate: ndarrays must have same number of dimensions, found: 1, 2'):
        hl.nd.concatenate([
            hl.nd.array([1]),
            hl.nd.array([[1]])
        ])

    with pytest.raises(ValueError, match=re.escape('hl.nd.concatenate: ndarrays must have same element types, found these element types: (int32, float64)')):
        hl.nd.concatenate([
            hl.nd.array([1]),
            hl.nd.array([1.0])
        ])

    with pytest.raises(ValueError, match=re.escape('hl.nd.concatenate: ndarrays must have same element types, found these element types: (int32, float64)')):
        hl.nd.concatenate([
            hl.nd.array([1]),
            hl.nd.array([[1.0]])
        ])


def test_vstack_1():
    ht = hl.utils.range_table(10)

    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])

    seq = (a, b)
    seq2 = hl.array([a, b])
    assert(np.array_equal(hl.eval(hl.nd.vstack(seq)), np.vstack(seq)))
    assert(np.array_equal(hl.eval(hl.nd.vstack(seq2)), np.vstack(seq)))
    ht2 = ht.annotate(x=hl.nd.array(a), y=hl.nd.array(b))
    ht2 = ht2.annotate(stacked=hl.nd.vstack([ht2.x, ht2.y]))
    assert np.array_equal(ht2.collect()[0].stacked, np.vstack([a, b]))

def test_vstack_2():
    ht = hl.utils.range_table(10)

    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    empty = np.array([], np.int64).reshape((0, 1))

    assert(np.array_equal(hl.eval(hl.nd.vstack((a, b))), np.vstack((a, b))))
    assert(np.array_equal(hl.eval(hl.nd.vstack(hl.array([a, b]))), np.vstack((a, b))))
    assert(np.array_equal(hl.eval(hl.nd.vstack((a, empty, b))), np.vstack((a, empty, b))))
    assert(np.array_equal(hl.eval(hl.nd.vstack(hl.array([a, empty, b]))), np.vstack((a, empty, b))))
    assert(np.array_equal(hl.eval(hl.nd.vstack((empty, a, b))), np.vstack((empty, a, b))))
    assert(np.array_equal(hl.eval(hl.nd.vstack(hl.array([empty, a, b]))), np.vstack((empty, a, b))))

    ht2 = ht.annotate(x=hl.nd.array(a), y=hl.nd.array(b))
    ht2 = ht2.annotate(stacked=hl.nd.vstack([ht2.x, ht2.y]))
    assert np.array_equal(ht2.collect()[0].stacked, np.vstack([a, b]))


def test_hstack():
    ht = hl.utils.range_table(10)

    def assert_table(a, b):
        ht2 = ht.annotate(x=hl.nd.array(a), y=hl.nd.array(b))
        ht2 = ht2.annotate(stacked=hl.nd.hstack([ht2.x, ht2.y]))
        assert np.array_equal(ht2.collect()[0].stacked, np.hstack([a, b]))

    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    assert(np.array_equal(hl.eval(hl.nd.hstack((a, b))), np.hstack((a, b))))
    assert(np.array_equal(hl.eval(hl.nd.hstack(hl.array([a, b]))), np.hstack((a, b))))
    assert_table(a, b)

    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    assert(np.array_equal(hl.eval(hl.nd.hstack((a, b))), np.hstack((a, b))))
    assert(np.array_equal(hl.eval(hl.nd.hstack(hl.array([a, b]))), np.hstack((a, b))))
    assert_table(a, b)

    empty = np.array([], np.int64).reshape((3, 0))
    assert(np.array_equal(hl.eval(hl.nd.hstack((a, empty))), np.hstack((a, empty))))
    assert(np.array_equal(hl.eval(hl.nd.hstack(hl.array([a, empty]))), np.hstack((a, empty))))
    assert(np.array_equal(hl.eval(hl.nd.hstack((empty, a))), np.hstack((empty, a))))
    assert(np.array_equal(hl.eval(hl.nd.hstack(hl.array([empty, a]))), np.hstack((empty, a))))
    assert_table(a, empty)
    assert_table(empty, a)


def test_eye():
    for i in range(13):
        assert_ndarrays_eq(*[(hl.nd.eye(i, y), np.eye(i, y)) for y in range(13)])


def test_identity():
    assert_ndarrays_eq(*[(hl.nd.identity(i), np.identity(i)) for i in range(13)])


def test_agg_ndarray_sum_empty():
    no_values = hl.utils.range_table(0).annotate(x=hl.nd.arange(5))
    assert no_values.aggregate(hl.agg.ndarray_sum(no_values.x)) is None


def test_agg_ndarray_sum_0_to_10():
    increasing_0d = hl.utils.range_table(10)
    increasing_0d = increasing_0d.annotate(x=hl.nd.array(increasing_0d.idx))
    assert np.array_equal(increasing_0d.aggregate(hl.agg.ndarray_sum(increasing_0d.x)), np.array(45))


def test_agg_ndarray_sum_ones_1d():
    just_ones_1d = hl.utils.range_table(20).annotate(x=hl.nd.ones((7,)))
    assert np.array_equal(just_ones_1d.aggregate(hl.agg.ndarray_sum(just_ones_1d.x)), np.full((7,), 20))


def test_agg_ndarray_sum_ones_2d():
    just_ones_2d = hl.utils.range_table(100).annotate(x=hl.nd.ones((2, 3)))
    assert np.array_equal(just_ones_2d.aggregate(hl.agg.ndarray_sum(just_ones_2d.x)), np.full((2, 3), 100))


def test_agg_ndarray_sum_with_transposes():
    transposes = hl.utils.range_table(4).annotate(x=hl.nd.arange(16).reshape((4, 4)))
    transposes = transposes.annotate(x = hl.if_else((transposes.idx % 2) == 0, transposes.x, transposes.x.T))
    np_arange_4_by_4 = np.arange(16).reshape((4, 4))
    transposes_result = (np_arange_4_by_4 * 2) + (np_arange_4_by_4.T * 2)
    assert np.array_equal(transposes.aggregate(hl.agg.ndarray_sum(transposes.x)), transposes_result)


def test_agg_ndarray_mismatched_dims_raises_fatal_error():
    with pytest.raises(FatalError) as exc:
        mismatched = hl.utils.range_table(5)
        mismatched = mismatched.annotate(x=hl.nd.ones((mismatched.idx,)))
        mismatched.aggregate(hl.agg.ndarray_sum(mismatched.x))
    assert "Can't sum" in str(exc.value)


def test_maximum_minimuim():
    x = np.arange(4)
    y = np.array([7, 0, 2, 4])
    z = [5, 2, 3, 1]
    nan_elem = np.array([1.0, float("nan"), 3.0, 6.0])
    f = np.array([1.0, 3.0, 6.0, 4.0])
    nx = hl.nd.array(x)
    ny = hl.nd.array(y)
    nf = hl.nd.array(f)
    ndnan_elem = hl.nd.array([1.0, hl.float64(float("NaN")), 3.0, 6.0])

    assert_ndarrays_eq(
        (hl.nd.maximum(nx, ny), np.maximum(x, y)),
        (hl.nd.maximum(ny, z), np.maximum(y, z)),
        (hl.nd.minimum(nx, ny), np.minimum(x, y)),
         (hl.nd.minimum(ny, z), np.minimum(y, z)),
    )

    np_nan_max = np.maximum(nan_elem, f)
    nan_max = hl.eval(hl.nd.maximum(ndnan_elem, nf))
    np_nan_min = np.minimum(nan_elem, f)
    nan_min = hl.eval(hl.nd.minimum(ndnan_elem, nf))
    max_matches = 0
    min_matches = 0
    for a, b in zip(np_nan_max, nan_max):
        if a == b:
            max_matches += 1
        elif np.isnan(a) and np.isnan(b):
            max_matches += 1
    for a, b in zip(np_nan_min, nan_min):
        if a == b:
            min_matches += 1
        elif np.isnan(a) and np.isnan(b):
            min_matches += 1

    assert(nan_max.size == max_matches)
    assert(nan_min.size == min_matches)


def test_ndarray_broadcasting_with_decorator():
    nd = hl.nd.array([[1, 4, 9], [16, 25, 36]])
    nd_sqrt = hl.eval(hl.nd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    nd = hl.eval(hl.sqrt(nd))
    assert(np.array_equal(nd, nd_sqrt))

    nd = hl.nd.array([[10, 100, 1000], [10000, 100000, 1000000]])
    nd_log10 = hl.eval(hl.nd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    nd = hl.eval(hl.log10(nd))
    assert(np.array_equal(nd, nd_log10))

    nd = hl.nd.array([[1.2, 2.3, 3.3], [4.3, 5.3, 6.3]])
    nd_floor = hl.eval(hl.nd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    nd = hl.eval(hl.floor(nd))
    assert(np.array_equal(nd, nd_floor))


def test_ndarray_indices_aggregations():
    ht = hl.utils.range_table(1)
    ht = ht.annotate_globals(g = hl.nd.ones((2, 2)))
    ht = ht.annotate(x = hl.nd.ones((2, 2)))
    ht = ht.annotate(a = hl.nd.solve(ht.x, 2 * ht.g))
    ht = ht.annotate(b = hl.nd.solve(2 * ht.g, ht.x))
    ht = ht.annotate(c = hl.nd.solve_triangular(2 * ht.g, hl.nd.eye(2)))
    ht = ht.annotate(d = hl.nd.solve_triangular(hl.nd.eye(2), 2 * ht.g))
    ht = ht.annotate(e = hl.nd.svd(ht.x))
    ht = ht.annotate(f = hl.nd.inv(ht.x))
    ht = ht.annotate(h = hl.nd.concatenate((ht.x, ht.g)))
    ht = ht.annotate(i = hl.nd.concatenate((ht.g, ht.x)))


def test_ndarray_log_broadcasting():
    expected = np.array([math.log(x) for x in [5, 10, 15, 20]]).reshape(2, 2)
    actual = hl.eval(hl.log(hl.nd.array([[5, 10], [15, 20]])))
    assert np.array_equal(actual, expected)
