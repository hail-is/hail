from hail.expr.blockmatrix_type import tblockmatrix
from hail.expr.types import hail_type, tarray
from hail.ir import BlockMatrixIR, ApplyBinaryOp, IR
from hail.utils.java import escape_str
from hail.typecheck import typecheck_method, sequenceof

from hail.utils.java import Env


class BlockMatrixRead(BlockMatrixIR):
    @typecheck_method(path=str)
    def __init__(self, path):
        super().__init__()
        self.path = path

    def render(self, r):
        return f'(BlockMatrixRead "{escape_str(self.path)}")'

    def _compute_type(self):
        self._type = Env.backend().blockmatrix_type(self)


class BlockMatrixMap2(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR, apply_bin_op=ApplyBinaryOp)
    def __init__(self, left, right, apply_bin_op):
        super().__init__()
        self.left = left
        self.right = right
        self.apply_bin_op = apply_bin_op

    def render(self, r):
        return f'(BlockMatrixMap2 {r(self.left)} {r(self.right)} {r(self.apply_bin_op)})'

    def _compute_type(self):
        self.right.typ  # Force
        self._type = self.left.typ


class BlockMatrixBroadcast(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR,
                      broadcast_kind=str,
                      shape=sequenceof(int),
                      block_size=int,
                      dims_partitioned=sequenceof(bool))
    def __init__(self, child, broadcast_kind, shape, block_size, dims_partitioned):
        super().__init__()
        self.child = child
        self.broadcast_kind = broadcast_kind
        self.shape = shape
        self.block_size = block_size
        self.dims_partitioned = dims_partitioned

    def render(self, r):
        return '(BlockMatrixBroadcast {} ({}) {} ({}) {})'\
            .format(escape_str(self.broadcast_kind),
                    ' '.join([str(x) for x in self.shape]),
                    self.block_size,
                    ' '.join([str(b) for b in self.dims_partitioned]),
                    r(self.child))

    def _compute_type(self):
        self._type = tblockmatrix(self.child.typ.element_type,
                                  self.shape,
                                  self.block_size,
                                  self.dims_partitioned)


class ValueToBlockMatrix(BlockMatrixIR):
    @typecheck_method(child=IR,
                      shape=sequenceof(int),
                      block_size=int,
                      dims_partitioned=sequenceof(bool))
    def __init__(self, child, shape, block_size, dims_partitioned):
        super().__init__()
        self.child = child
        self.shape = shape
        self.block_size = block_size
        self.dims_partitioned = dims_partitioned

    def render(self, r):
        return '(ValueToBlockMatrix ({}) {} ({}) {})'.format(' '.join([str(x) for x in self.shape]),
                                                             self.block_size,
                                                             ' '.join([str(b) for b in self.dims_partitioned]),
                                                             r(self.child))

    def _compute_type(self):
        child_type = self.child.typ
        if isinstance(child_type, tarray):
            element_type = child_type._element_type
        else:
            element_type = child_type

        self._type = tblockmatrix(element_type, self.shape, self.block_size, self.dims_partitioned)


class JavaBlockMatrix(BlockMatrixIR):
    def __init__(self, jbm):
        super().__init__()
        self.jir = Env.hail().expr.ir.BlockMatrixLiteral(jbm)

    def render(self, r):
        return f'(JavaBlockMatrix {r.add_jir(self.jir)})'

    def _compute_type(self):
        self._type = tblockmatrix._from_java(self.jir.typ())
