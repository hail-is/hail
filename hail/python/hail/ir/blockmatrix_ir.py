from hail.expr.blockmatrix_type import tblockmatrix
from hail.expr.types import hail_type
from hail.ir import BlockMatrixIR, ApplyBinaryOp, IR
from hail.utils.java import escape_str
from hail.typecheck import typecheck_method, sequenceof

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


class BlockMatrixMap2(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR, apply_bin_op=ApplyBinaryOp)
    def __init__(self, left, right, apply_bin_op):
        super().__init__()
        self._left = left
        self._right = right
        self._apply_bin_op = apply_bin_op

    def render(self, r):
        return f'(BlockMatrixMap2 {r(self._left)} {r(self._right)} {r(self._apply_bin_op)})'

    def _compute_type(self):
        self._right.typ  # Force
        self._type = self._left.typ


class BlockMatrixMap(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, apply_bin_op=ApplyBinaryOp)
    def __init__(self, child, apply_bin_op):
        super().__init__()
        self._child = child
        self._apply_bin_op = apply_bin_op

    def render(self, r):
        return f'(BlockMatrixMap {r(self._child)} {r(self._apply_bin_op)})'

    def _compute_type(self):
        self._type = self._child.typ


class BlockMatrixBroadcast(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR,
                      broadcast_type=str,
                      shape=sequenceof(int),
                      block_size=int,
                      dims_partitioned=sequenceof(bool))
    def __init__(self, child, broadcast_type, shape, block_size, dims_partitioned):
        super().__init__()
        self._child = child
        self._broadcast_type = broadcast_type
        self._shape = shape
        self._block_size = block_size
        self._dims_partitioned = dims_partitioned

    def render(self, r):
        return '(BlockMatrixBroadcast {} ({}) {} ({}) {})'\
            .format(escape_str(self._broadcast_type),
                    ' '.join([str(x) for x in self._shape]),
                    self._block_size,
                    ' '.join([str(b) for b in self._dims_partitioned]),
                    r(self._child))

    def _compute_type(self):
        self._type = tblockmatrix(self._child.typ.element_type,
                                  self._shape,
                                  self._block_size,
                                  self._dims_partitioned)


class ValueToBlockMatrix(BlockMatrixIR):
    @typecheck_method(child=IR,
                      element_type=hail_type,
                      shape=sequenceof(int),
                      block_size=int,
                      dims_partitioned=sequenceof(bool))
    def __init__(self, child, element_type, shape, block_size, dims_partitioned):
        super().__init__()
        self._child = child
        self._element_type = element_type
        self._shape = shape
        self._block_size = block_size
        self._dims_partitioned = dims_partitioned

    def render(self, r):
        return '(ValueToBlockMatrix {} ({}) {} ({}) {})'.format(self._element_type._parsable_string(),
                                                                ' '.join([str(x) for x in self._shape]),
                                                                self._block_size,
                                                                ' '.join([str(b) for b in self._dims_partitioned]),
                                                                r(self._child))

    def _compute_type(self):
        self._type = tblockmatrix(self._element_type, self._shape, self._block_size, self._dims_partitioned)


class JavaBlockMatrix(BlockMatrixIR):
    def __init__(self, jbm):
        super().__init__()
        self._jir = Env.hail().expr.ir.BlockMatrixLiteral(jbm)

    def render(self, r):
        return f'(JavaBlockMatrix {r.add_jir(self._jir)})'

    def _compute_type(self):
        self._type = tblockmatrix._from_java(self._jir.typ())
