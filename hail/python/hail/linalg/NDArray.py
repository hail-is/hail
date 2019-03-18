import hail.ir as ir
from hail.typecheck import typecheck_method, tupleof
from hail.utils.java import Env


class NDArray(object):
    def __init__(self, ndir):
        self._ir = ndir

    @typecheck_method(item=tupleof(int))
    def __getitem__(self, item):
        idxs = ir.MakeArray([ir.I64(idx) for idx in item], ir.tarray(ir.tint64))
        return Env.backend().execute(ir.NDArrayRef(self._ir, idxs))
