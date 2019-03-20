import hail as hl
import numpy as np
import hail.ir as ir
from hail.expr import construct_expr, to_expr

from hail.typecheck import typecheck


@typecheck(nd=np.ndarray)
def ndarray(nd):
    nd = nd.astype(np.float64)
    data_ir = hl.array(list(nd.flat))
    shape_ir = to_expr(list(nd.shape), ir.tarray(ir.tint64))  # Indices must be int64
    ndir = ir.MakeNDArray(data_ir._ir, shape_ir._ir, ir.TrueIR())

    return construct_expr(ndir, ndir.typ)
