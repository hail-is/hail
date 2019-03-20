import hail as hl
import numpy as np
import hail.ir as ir
from hail.expr import construct_expr, to_expr

from hail.typecheck import typecheck


@typecheck(nd=np.ndarray)
def ndarray(nd):
    nd = nd.astype(np.float64)
    data = hl.array(list(nd.flat))
    shape = to_expr(nd.shape, ir.tarray(ir.tint64))
    ndir = ir.MakeNDArray(data._ir, shape._ir, ir.TrueIR())

    return construct_expr(ndir, ndir.typ)
