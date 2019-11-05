from functools import reduce

import hail as hl
from hail.expr.functions import _ndarray
from hail.typecheck import *
from hail.expr.expressions import expr_int64, expr_tuple, expr_any, Int64Expression


def array(input_array):
    return _ndarray(input_array)


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()), value=expr_any)
def full(shape, value):
    if isinstance(shape, Int64Expression):
        shape_product = shape
    else:
        shape_product = reduce(lambda a, b: a * b, shape)
    return array(hl.range(hl.int32(shape_product)).map(lambda x: value)).reshape(shape)


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()))
def zeros(shape):
    return full(shape, 0)


@typecheck(shape=oneof(expr_int64, tupleof(expr_int64), expr_tuple()))
def ones(shape):
    return full(shape, 1)