from functools import reduce

import hail as hl
from hail.expr.functions import _ndarray
from hail.expr.types import HailType
from hail.typecheck import *
from hail.expr.expressions import expr_int32, expr_int64, expr_tuple, expr_any, expr_ndarray, Int64Expression, cast_expr, construct_expr
from hail.expr.expressions.typed_expressions import NDArrayNumericExpression
from hail.ir import NDArrayQR


def array(input_array):
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
    input_array : :class:`.ArrayExpression` or numpy ndarray or nested python lists

    Returns
    -------
    :class:`.NDArrayExpression`
        An ndarray based on the input array.
    """
    return _ndarray(input_array)


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


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()), value=expr_any, dtype=nullable(HailType))
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


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()), dtype=HailType)
def zeros(shape, dtype=hl.tfloat64):
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
            Desired hail type.

       See Also
       --------
       :func:`.full`

       Returns
       -------
       :class:`.NDArrayNumericExpression`
           ndarray of the specified size full of zeros.
       """
    return full(shape, 0, dtype)


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()), dtype=HailType)
def ones(shape, dtype=hl.tfloat64):
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
            Desired hail type.


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
        The array h contains the Householder reflectors that generate q along with r. The tau array contains scaling factors for the reflectors
    """

    assert nd.ndim == 2, "QR decomposition requires 2 dimensional ndarray"

    if mode not in ["reduced", "r", "raw", "complete"]:
        raise ValueError(f"Unrecognized mode '{mode}' for QR decomposition")

    float_nd = nd.map(lambda x: hl.float64(x))
    ir = NDArrayQR(float_nd._ir, mode)
    if mode == "raw":
        return construct_expr(ir, hl.ttuple(hl.tndarray(hl.tfloat64, 2), hl.tndarray(hl.tfloat64, 1)))
    elif mode == "r":
        return construct_expr(ir, hl.tndarray(hl.tfloat64, 2))
    elif mode in ["complete", "reduced"]:
        return construct_expr(ir, hl.ttuple(hl.tndarray(hl.tfloat64, 2), hl.tndarray(hl.tfloat64, 2)))
