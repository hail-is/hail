from functools import reduce

import hail as hl
from hail.expr.functions import _ndarray
from hail.expr.functions import array as aarray
from hail.expr.types import HailType, tfloat64, ttuple, tndarray
from hail.typecheck import typecheck, nullable, oneof, tupleof, sequenceof
from hail.expr.expressions import (
    expr_int32, expr_int64, expr_tuple, expr_any, expr_array, expr_ndarray,
    expr_numeric, Int64Expression, cast_expr, construct_expr)
from hail.expr.expressions.typed_expressions import NDArrayNumericExpression
from hail.ir import NDArrayQR, NDArrayInv, NDArrayConcat, NDArraySVD, Apply

tsequenceof_nd = oneof(sequenceof(expr_ndarray()), expr_array(expr_ndarray()))
shape_type = oneof(expr_int64, tupleof(expr_int64), expr_tuple())


def array(input_array, dtype=None):
    """Construct an :class:`.NDArrayExpression`

    Examples
    --------

    >>> hl.eval(hl.nd.array([1, 2, 3, 4]))
    array([1, 2, 3, 4], dtype=int32)

    >>> hl.eval(hl.nd.array([[1, 2, 3], [4, 5, 6]]))
    array([[1, 2, 3],
       [4, 5, 6]], dtype=int32)

    >>> hl.eval(hl.nd.array(np.identity(3)))
    array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

    >>> hl.eval(hl.nd.array(hl.range(10, 20)))
    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=int32)

    Parameters
    ----------
    input_array : :class:`.ArrayExpression`, numpy ndarray, or nested python lists/tuples
    dtype : :class:`.HailType`
        Desired hail type.  Default: `float64`.

    Returns
    -------
    :class:`.NDArrayExpression`
        An ndarray based on the input array.
    """
    return _ndarray(input_array, dtype=dtype)


@typecheck(a=expr_array(), shape=shape_type)
def from_column_major(a, shape):
    assert len(shape) == 2
    return array(a).reshape(tuple(reversed(shape))).T


@typecheck(start=expr_int32, stop=nullable(expr_int32), step=expr_int32)
def arange(start, stop=None, step=1) -> NDArrayNumericExpression:
    """Returns a 1-dimensions ndarray of integers from `start` to `stop` by `step`.

    Examples
    --------

    >>> hl.eval(hl.nd.arange(10))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)

    >>> hl.eval(hl.nd.arange(3, 10))
    array([3, 4, 5, 6, 7, 8, 9], dtype=int32)

    >>> hl.eval(hl.nd.arange(0, 10, step=3))
    array([0, 3, 6, 9], dtype=int32)

    Notes
    -----
    The range includes `start`, but excludes `stop`.

    If provided exactly one argument, the argument is interpreted as `stop` and
    `start` is set to zero. This matches the behavior of Python's ``range``.

    Parameters
    ----------
    start : int or :class:`.Expression` of type :py:data:`.tint32`
        Start of range.
    stop : int or :class:`.Expression` of type :py:data:`.tint32`
        End of range.
    step : int or :class:`.Expression` of type :py:data:`.tint32`
        Step of range.

    Returns
    -------
    :class:`.NDArrayNumericExpression`
        A 1-dimensional ndarray from `start` to `stop` by `step`.
    """
    return array(hl.range(start, stop, step))


@typecheck(shape=shape_type, value=expr_any, dtype=nullable(HailType))
def full(shape, value, dtype=None):
    """Creates a hail :class:`.NDArrayNumericExpression` full of the specified value.

    Examples
    --------

    Create a 5 by 7 NDArray of type :py:data:`.tfloat64` 9s.

    >>> hl.nd.full((5, 7), 9)

    It is possible to specify a type other than :py:data:`.tfloat64` with the `dtype` argument.

    >>> hl.nd.full((5, 7), 9, dtype=hl.tint32)

    Parameters
    ----------
    shape : `tuple` or :class:`.TupleExpression`
            Desired shape.
    value : :class:`.Expression` or python value
            Value to fill ndarray with.
    dtype : :class:`.HailType`
            Desired hail type.

    Returns
    -------
    :class:`.NDArrayNumericExpression`
        An ndarray of the specified shape filled with the specified value.
    """
    if isinstance(shape, Int64Expression):
        shape_product = shape
    else:
        shape_product = reduce(lambda a, b: a * b, shape)
    return arange(hl.int32(shape_product)).map(lambda x: cast_expr(value, dtype)).reshape(shape)


@typecheck(shape=shape_type, dtype=HailType)
def zeros(shape, dtype=tfloat64):
    """Creates a hail :class:`.NDArrayNumericExpression` full of zeros.

       Examples
       --------

       Create a 5 by 7 NDArray of type :py:data:`.tfloat64` zeros.

       >>> hl.nd.zeros((5, 7))

       It is possible to specify a type other than :py:data:`.tfloat64` with the `dtype` argument.

       >>> hl.nd.zeros((5, 7), dtype=hl.tfloat32)


       Parameters
       ----------
       shape : `tuple` or :class:`.TupleExpression`
            Desired shape.
       dtype : :class:`.HailType`
            Desired hail type.  Default: `float64`.

       See Also
       --------
       :func:`.full`

       Returns
       -------
       :class:`.NDArrayNumericExpression`
           ndarray of the specified size full of zeros.
       """
    return full(shape, 0, dtype)


@typecheck(shape=shape_type, dtype=HailType)
def ones(shape, dtype=tfloat64):
    """Creates a hail :class:`.NDArrayNumericExpression` full of ones.

       Examples
       --------

       Create a 5 by 7 NDArray of type :py:data:`.tfloat64` ones.

       >>> hl.nd.ones((5, 7))

       It is possible to specify a type other than :py:data:`.tfloat64` with the `dtype` argument.

       >>> hl.nd.ones((5, 7), dtype=hl.tfloat32)


       Parameters
       ----------
       shape : `tuple` or :class:`.TupleExpression`
            Desired shape.
       dtype : :class:`.HailType`
            Desired hail type.  Default: `float64`.


       See Also
       --------
       :func:`.full`

       Returns
       -------
       :class:`.NDArrayNumericExpression`
           ndarray of the specified size full of ones.
       """
    return full(shape, 1, dtype)


@typecheck(nd=expr_ndarray())
def diagonal(nd):
    """Gets the diagonal of a 2 dimensional NDArray.

    Examples
    --------

    >>> hl.eval(hl.nd.diagonal(hl.nd.array([[1, 2], [3, 4]])))
    array([1, 4], dtype=int32)

    :param nd: A 2 dimensional NDArray, shape(M, N).
    :return: A 1 dimension NDArray of length min (M, N), containing the diagonal of `nd`.
    """
    assert nd.ndim == 2, "diagonal requires 2 dimensional ndarray"
    shape_min = hl.min(nd.shape[0], nd.shape[1])
    return hl.nd.array(hl.range(hl.int32(shape_min)).map(lambda i: nd[i, i]))


@typecheck(a=expr_ndarray(), b=expr_ndarray())
def solve(a, b):
    """Solve a linear system.

    Parameters
    ----------
    a : :class:`.NDArrayNumericExpression`, (N, N)
        Coefficient matrix.
    b : :class:`.NDArrayNumericExpression`, (N,) or (N, K)
        Dependent variables.

    Returns
    -------
    :class:`.NDArrayNumericExpression`, (N,) or (N, K)
        Solution to the system Ax = B. Shape is same as shape of B.

    """
    assert a.ndim == 2
    assert b.ndim == 1 or b.ndim == 2

    b_ndim_orig = b.ndim

    if b_ndim_orig == 1:
        b = b.reshape((-1, 1))

    if a.dtype.element_type != hl.tfloat64:
        a = a.map(lambda e: hl.float64(e))
    if b.dtype.element_type != hl.tfloat64:
        b = b.map(lambda e: hl.float64(e))

    ir = Apply("linear_solve", hl.tndarray(hl.tfloat64, 2), a._ir, b._ir)
    result = construct_expr(ir, hl.tndarray(hl.tfloat64, 2), a._indices, a._aggregations)

    if b_ndim_orig == 1:
        result = result.reshape((-1))
    return result


@typecheck(nd=expr_ndarray(), mode=str)
def qr(nd, mode="reduced"):
    """Performs a QR decomposition.

    :param nd: A 2 dimensional ndarray, shape(M, N)
    :param mode: One of "reduced", "complete", "r", or "raw".

        If K = min(M, N), then:

        - `reduced`: returns q and r with dimensions (M, K), (K, N)
        - `complete`: returns q and r with dimensions (M, M), (M, N)
        - `r`: returns only r with dimensions (K, N)
        - `raw`: returns h, tau with dimensions (N, M), (K,)

    Returns
    -------
    - q: ndarray of float64
        A matrix with orthonormal columns.
    - r: ndarray of float64
        The upper-triangular matrix R.
    - (h, tau): ndarrays of float64
        The array h contains the Householder reflectors that generate q along with r.
        The tau array contains scaling factors for the reflectors
    """

    assert nd.ndim == 2, "QR decomposition requires 2 dimensional ndarray"

    if mode not in ["reduced", "r", "raw", "complete"]:
        raise ValueError(f"Unrecognized mode '{mode}' for QR decomposition")

    float_nd = nd.map(lambda x: hl.float64(x))
    ir = NDArrayQR(float_nd._ir, mode)
    indices = nd._indices
    aggs = nd._aggregations
    if mode == "raw":
        return construct_expr(ir, ttuple(tndarray(tfloat64, 2), tndarray(tfloat64, 1)), indices, aggs)
    elif mode == "r":
        return construct_expr(ir, tndarray(tfloat64, 2), indices, aggs)
    elif mode in ["complete", "reduced"]:
        return construct_expr(ir, ttuple(tndarray(tfloat64, 2), tndarray(tfloat64, 2)), indices, aggs)


@typecheck(nd=expr_ndarray(), full_matrices=bool, compute_uv=bool)
def svd(nd, full_matrices=True, compute_uv=True):
    """Performs a singular value decomposition.

    :param nd: :class:`.NDArrayExpression`
        A 2 dimensional ndarray, shape(M, N).
    :param full_matrices: `bool`
        If True (default), u and vt have dimensions (M, M) and (N, N) respectively. Otherwise, they have dimensions
        (M, K) and (K, N), where K = min(M, N)
    :param compute_uv: `bool`
        If True (default), compute the singular vectors u and v. Otherwise, only return a single ndarray, s.

    Returns
    -------
    - u: :class:`.NDArrayExpression`
        The left singular vectors.
    - s: :class:`.NDArrayExpression`
        The singular values.
    - vt: :class:`.NDArrayExpression`
        The right singular vectors.
    """
    float_nd = nd.map(lambda x: hl.float64(x))
    ir = NDArraySVD(float_nd._ir, full_matrices, compute_uv)

    return_type = ttuple(tndarray(tfloat64, 2), tndarray(tfloat64, 1), tndarray(tfloat64, 2)) if compute_uv else tndarray(tfloat64, 1)
    return construct_expr(ir, return_type)


@typecheck(nd=expr_ndarray())
def inv(nd):
    """Performs a matrix inversion.

    :param nd: A 2 dimensional ndarray, shape(M, N)

    Returns
    -------
    - a: ndarray of float64
        The inverted matrix
    """

    assert nd.ndim == 2, "Matrix inversion requires 2 dimensional ndarray"

    float_nd = nd.map(lambda x: hl.float64(x))
    ir = NDArrayInv(float_nd._ir)
    return construct_expr(ir, tndarray(tfloat64, 2))


@typecheck(nds=tsequenceof_nd, axis=int)
def concatenate(nds, axis=0):
    """Join a sequence of arrays along an existing axis.

    Examples
    --------

    >>> x = hl.nd.array([[1., 2.], [3., 4.]])
    >>> y = hl.nd.array([[5.], [6.]])
    >>> hl.eval(hl.nd.concatenate([x, y], axis=1))
    array([[1., 2., 5.],
           [3., 4., 6.]])
    >>> x = hl.nd.array([1., 2.])
    >>> y = hl.nd.array([3., 4.])
    >>> hl.eval(hl.nd.concatenate((x, y), axis=0))
    array([1., 2., 3., 4.])

    Parameters
    ----------
    :param nds: a1, a2, …sequence of array_like
        The arrays must have the same shape, except in the dimension corresponding to axis (the first, by default).
        Note: unlike Numpy, the numerical element type of each array_like must match.
    :param axis: int, optional
        The axis along which the arrays will be joined. Default is 0.
        Note: unlike Numpy, if provided, axis cannot be None.

    Returns
    -------
    - res: ndarray
        The concatenated array
    """
    head_nd = nds[0]
    head_ndim = head_nd.ndim
    hl.case().when(hl.all(lambda a: a.ndim == head_ndim, nds), True).or_error("Mismatched ndim")

    makearr = aarray(nds)
    concat_ir = NDArrayConcat(makearr._ir, axis)

    return construct_expr(concat_ir, tndarray(head_nd._type.element_type, head_ndim))


@typecheck(N=expr_numeric, M=nullable(expr_numeric), dtype=HailType)
def eye(N, M=None, dtype=hl.tfloat64):
    """
    Construct a 2-D :class:`.NDArrayExpression` with ones on the *main* diagonal
    and zeros elsewhere.

    Parameters
    ----------
    N : :class:`.NumericExpression` or Python number
      Number of rows in the output.
    M : :class:`.NumericExpression` or Python number, optional
      Number of columns in the output. If None, defaults to `N`.
    dtype : numeric :class:`.HailType`, optional
      Element type of the returned array. Defaults to :py:data:`.tfloat64`

    Returns
    -------
    I : :class:`.NDArrayExpression` representing a Hail ndarray of shape (N,M)
      An ndarray whose elements are equal to one on the main diagonal, zeroes elsewhere.

    See Also
    --------
    :func:`.identity`
    :func:`.diagonal`

    Examples
    --------
    >>> hl.eval(hl.nd.eye(3))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> hl.eval(hl.nd.eye(2, 5, dtype=hl.tint32))
    array([[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0]], dtype=int32)
    """

    n_row = hl.int32(N)
    if M is None:
        n_col = n_row
    else:
        n_col = hl.int32(M)

    return hl.nd.array(hl.range(0, n_row * n_col).map(
        lambda i: hl.if_else((i // n_col) == (i % n_col),
                             hl.literal(1, dtype),
                             hl.literal(0, dtype))
    )).reshape((n_row, n_col))


@typecheck(N=expr_numeric, dtype=HailType)
def identity(N, dtype=hl.tfloat64):
    """
    Constructs a 2-D :class:`.NDArrayExpression` representing the identity array.
    The identity array is a square array with ones on the main diagonal.

    Parameters
    ----------
    n : :class:`.NumericExpression` or Python number
      Number of rows and columns in the output.
    dtype : numeric :class:`.HailType`, optional
      Element type of the returned array. Defaults to :py:data:`.tfloat64`

    Returns
    -------
    out : :class:`.NDArrayExpression`
        `n` x `n` ndarray with its main diagonal set to one, and all other elements 0.

    See Also
    --------
    :func:`.eye`

    Examples
    --------
    >>> hl.eval(hl.nd.identity(3))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    return eye(N, dtype=dtype)


@typecheck(arrs=tsequenceof_nd)
def vstack(arrs):
    """
    Stack arrays in sequence vertically (row wise).
    1-D arrays of shape `(N,)`, will reshaped to `(1,N)` before concatenation.
    For all other arrays, equivalent to  :func:`.concatenate` with axis=0.

    Parameters
    ----------
    arrs : sequence of :class:`.NDArrayExpression`
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : :class:`.NDArrayExpression`
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    :func:`.concatenate` : Join a sequence of arrays along an existing axis.

    Examples
    --------
    >>> a = hl.nd.array([1, 2, 3])
    >>> b = hl.nd.array([2, 3, 4])
    >>> hl.eval(hl.nd.vstack((a,b)))
    array([[1, 2, 3],
           [2, 3, 4]], dtype=int32)
    >>> a = hl.nd.array([[1], [2], [3]])
    >>> b = hl.nd.array([[2], [3], [4]])
    >>> hl.eval(hl.nd.vstack((a,b)))
    array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]], dtype=int32)
    """
    head_ndim = arrs[0].ndim

    if head_ndim == 1:
        return concatenate(hl.map(lambda a: a._broadcast(2), arrs), 0)

    return concatenate(arrs, 0)


@typecheck(arrs=tsequenceof_nd)
def hstack(arrs):
    """
    Stack arrays in sequence horizontally (column wise).
    Equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis.

    This function makes most sense for arrays with up to 3 dimensions.
    :func:`.concatenate` provides more general stacking and concatenation operations.

    Parameters
    ----------
    tup : sequence of :class:`.NDArrayExpression`
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    Returns
    -------
    stacked : :class:`.NDArrayExpression`
        The array formed by stacking the given arrays.

    See Also
    --------
    :func:`.concatenate`
    :func:`.vstack`

    Examples
    --------
    >>> a = hl.nd.array([1,2,3])
    >>> b = hl.nd.array([2,3,4])
    >>> hl.eval(hl.nd.hstack((a,b)))
    array([1, 2, 3, 2, 3, 4], dtype=int32)
    >>> a = hl.nd.array([[1],[2],[3]])
    >>> b = hl.nd.array([[2],[3],[4]])
    >>> hl.eval(hl.nd.hstack((a,b)))
    array([[1, 2],
           [2, 3],
           [3, 4]], dtype=int32)
    """
    head_ndim = arrs[0].ndim

    if head_ndim == 1:
        axis = 0
    else:
        axis = 1

    return concatenate(arrs, axis)
