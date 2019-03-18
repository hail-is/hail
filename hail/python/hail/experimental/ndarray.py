import numpy as np
import hail.ir as ir
from hail.linalg.NDArray import NDArray

from hail.typecheck import typecheck


@typecheck(data=np.ndarray)
def ndarray(data):
    data = data.astype(np.float64)
    # Just until literals are supported
    data_ir = ir.MakeArray([ir.F64(x) for x in data.flat], ir.tarray(ir.tfloat64))
    shape_ir = ir.MakeArray([ir.I64(dim) for dim in data.shape], ir.tarray(ir.tint64))
    return NDArray(ir.MakeNDArray(data_ir, shape_ir, ir.TrueIR()))
