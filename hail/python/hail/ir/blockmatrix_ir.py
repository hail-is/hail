from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir import BlockMatrixIR, ApplyBinaryOp
from hail.utils.java import escape_str
from hail.typecheck import typecheck_method

from hail.utils.java import Env


class BlockMatrixRead(BlockMatrixIR):
    @typecheck_method(path=str)
    def __init__(self, path):
        super().__init__()
        self._path = path

    def render(self, r):
        return f'(BlockMatrixRead "{escape_str(self._path)}")'

    def _compute_type(self):
        self._type = Env.backend().blockmatrix_type(self)


class BlockMatrixAdd(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR)
    def __init__(self, left, right):
        super().__init__()
        self._left = left
        self._right = right

    def render(self, r):
        return f'(BlockMatrixAdd {r(self._left)} {r(self._right)})'

    def _compute_type(self):
        self._right.typ # force
        self._type = self._left.typ


class BlockMatrixElementWiseBinaryOp(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR, applyBinOp=ApplyBinaryOp)
    def __init__(self, left, right, applyBinOp):
        super().__init__()
        self._left = left
        self._right = right
        self._applyBinOp = applyBinOp

    def render(self, r):
        return f'(BlockMatrixElementWiseBinaryOp {r(self._left)} {r(self._right)} {r(self._applyBinOp)})'

    def _compute_type(self, ):
        self._right.typ # force
        self._type = self._left.typ


class BlockMatrixBroadcastValue(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, apply_bin_op=ApplyBinaryOp)
    def __init__(self, child, apply_bin_op):
        super().__init__()
        self._child = child
        self._apply_bin_op = apply_bin_op

    def render(self, r):
        return f'(BlockMatrixBroadcastValue {r(self._child)} {r(self._apply_bin_op)})'

    def _compute_type(self):
        self._type = self._child.typ


class JavaBlockMatrix(BlockMatrixIR):
    def __init__(self, jbm):
        super().__init__()
        self._jir = Env.hail().expr.ir.BlockMatrixLiteral(jbm)

    def render(self, r):
        return f'(JavaBlockMatrix {r.add_jir(self._jir)})'

    def _compute_type(self):
        self._type = tblockmatrix._from_java(self._jir.typ())
