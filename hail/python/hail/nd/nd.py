from functools import reduce

import hail as hl
from hail.expr.functions import _ndarray
from hail.expr import Expression, NDArrayExpression

def array(input_array):
    return _ndarray(input_array)

def full(shape, value):
    if isinstance(shape, Expression):
        return None
    else:
        shapeProduct = reduce(lambda a, b: a * b, shape)
        return array(hl.range(shapeProduct).map(lambda x: value)).reshape(shape)

def zeros(shape):
    return full(shape, 0)

def ones(shape):
    return full(shape, 1)