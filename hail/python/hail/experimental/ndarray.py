import numpy as np
import hail.ir as ir
from hail.expr import construct_expr

from hail.typecheck import typecheck


@typecheck(collection=np.ndarray)
def ndarray(collection):
    collection = collection.astype(np.float64)
    # Explicitly creating IRs that are supported in CxxCompile
    data_ir = ir.MakeArray([ir.F64(x) for x in collection.flat], ir.tarray(ir.tfloat64))
    shape_ir = ir.MakeArray([ir.I64(dim) for dim in collection.shape], ir.tarray(ir.tint64))
    ndir = ir.MakeNDArray(data_ir, shape_ir, ir.TrueIR())

    return construct_expr(ndir, ir.tndarray(ir.tfloat64))
