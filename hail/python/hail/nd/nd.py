from functools import reduce

import hail as hl
from hail.expr.functions import _ndarray
from hail.expr.types import HailType
from hail.typecheck import *
from hail.expr.expressions import expr_int64, expr_tuple, expr_any, Int64Expression, cast_expr


def array(input_array):
    return _ndarray(input_array)


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()), value=expr_any, dtype=nullable(HailType))
def full(shape, value, dtype=None):
    if isinstance(shape, Int64Expression):
        shape_product = shape
    else:
        shape_product = reduce(lambda a, b: a * b, shape)
    return array(hl.range(hl.int32(shape_product)).map(lambda x: cast_expr(value, dtype))).reshape(shape)


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()), dtype=HailType)
def zeros(shape, dtype=hl.tfloat64):
    """Creates a hail :class:`.NDArrayNumericExpression` full of zeros.

       Examples
       --------

       Create a 5 by 7 NDArray of type `tint64` zeros.

       >>> hl._nd.zeros((5, 7))

       It is possible to specify a type other than `tint64` with the `dtype` argument.

       >>> hl._nd.zeros((5, 7), dtype=hl.tfloat32)


       Parameters
       ----------
       shape : `tuple` or `TupleExpression`
            Desired shape.
       dtype : `HailType`
            Desired hail type.

       Returns
       -------
       :class:`.NDArrayNumericExpression`
           NDArray of the specified size full of zeros.
       """
    return full(shape, 0, dtype)


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()), dtype=HailType)
def ones(shape, dtype=hl.tfloat64):
    return full(shape, 1, dtype)