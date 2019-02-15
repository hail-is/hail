from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir import BlockMatrixIR, ApplyBinaryOp, IR, parsable_strings, tarray
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
                      in_index_expr=sequenceof(int),
                      shape=sequenceof(int),
                      block_size=int,
                      dims_partitioned=sequenceof(bool))
    def __init__(self, child, in_index_expr, shape, block_size, dims_partitioned):
        super().__init__()
        self.child = child
        self.in_index_expr = in_index_expr
        self.shape = shape
        self.block_size = block_size
        self.dims_partitioned = dims_partitioned

    def render(self, r):
        return '(BlockMatrixBroadcast {} {} {} {} {})'\
            .format(_serialize_ints(self.in_index_expr),
                    _serialize_ints(self.shape),
                    self.block_size,
                    _serialize_ints(self.dims_partitioned),
                    r(self.child))

    def _compute_type(self):
        self._type = tblockmatrix(self.child.typ.element_type,
                                  self.shape,
                                  False,
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
        return '(ValueToBlockMatrix {} {} {} {})'.format(_serialize_ints(self.shape),
                                                         self.block_size,
                                                         _serialize_ints(self.dims_partitioned),
                                                         r(self.child))

    def _compute_type(self):
        child_type = self.child.typ
        if isinstance(child_type, tarray):
            element_type = child_type._element_type
        else:
            element_type = child_type

        tensor_shape, is_row_vector = self._matrix_shape_to_tensor_shape(self.shape)
        self._type = tblockmatrix(element_type, tensor_shape, is_row_vector, self.block_size, self.dims_partitioned)

    @staticmethod
    def _matrix_shape_to_tensor_shape(shape):
        assert len(shape) == 2

        if shape == [1, 1]:
            return [], False
        elif shape[0] == 1:
            return [shape[1]], True
        elif shape[1] == 1:
            return [shape[0]], False
        else:
            return shape, False


class JavaBlockMatrix(BlockMatrixIR):
    def __init__(self, jbm):
        super().__init__()
        self.jir = Env.hail().expr.ir.BlockMatrixLiteral(jbm)

    def render(self, r):
        return f'(JavaBlockMatrix {r.add_jir(self.jir)})'

    def _compute_type(self):
        self._type = tblockmatrix._from_java(self.jir.typ())


def _serialize_ints(ints):
    return "(" + ' '.join([str(x) for x in ints]) + ")"
