from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir import BlockMatrixIR, ApplyBinaryOp, Ref, F64
from hail.utils.java import escape_str
from hail.typecheck import typecheck_method

from hail.utils.java import Env
import collections


class BlockMatrixRead(BlockMatrixIR):
    @typecheck_method(path=str)
    def __init__(self, path):
        super().__init__()
        self._path = path

    def render(self, r):
        return f'(BlockMatrixRead "{escape_str(self._path)}")'

    def _compute_type(self):
        self._type = Env.backend().blockmatrix_type(self)


class BlockMatrixElementWiseBinaryOp(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR, applyBinOp=ApplyBinaryOp)
    def __init__(self, left, right, applyBinOp):
        super().__init__()
        self._left = left
        self._right = right
        self._applyBinOp = applyBinOp

    def render(self, r):
        return f'(BlockMatrixElementWiseBinaryOp {r(self._left)} {r(self._right)} {r(self._applyBinOp)})'

    def _compute_type(self):
        right_type = self._right.typ
        left_type = self._left.typ
        self._type = tblockmatrix(left_type.element_type,
                                  _compute_shape(left_type.shape, right_type.shape),
                                  left_type.block_size,
                                  left_type.dims_partitioned)


class BlockMatrixBroadcastValue(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, apply_bin_op=ApplyBinaryOp)
    def __init__(self, child, apply_bin_op):
        super().__init__()
        self._child = child
        self._apply_bin_op = apply_bin_op

    def render(self, r):
        return f'(BlockMatrixBroadcastValue {r(self._child)} {r(self._apply_bin_op)})'

    def _compute_type(self):
        child_type = self._child.typ
        self._type = tblockmatrix(child_type.element_type,
                                  _compute_shape(*self._get_children_shapes()),
                                  child_type.block_size,
                                  child_type.dims_partitioned)

    def _get_children_shapes(self):
        child_matrix = self._child
        left, right = self._apply_bin_op.l, self._apply_bin_op.r

        if isinstance(left, Ref):
            left_shape = child_matrix.typ.shape
            if isinstance(right, F64):
                right_shape = [1, 1]
            else:  # MakeStruct to hold vectors
                right_shape = right.fields[0][1].value
        else:
            right_shape = child_matrix.typ.shape
            if isinstance(left, F64):
                left_shape = [1, 1]
            else:  # MakeStruct to hold vectors
                left_shape = left.fields[0][1].value

        return left_shape, right_shape


class JavaBlockMatrix(BlockMatrixIR):
    def __init__(self, jbm):
        super().__init__()
        self._jir = Env.hail().expr.ir.BlockMatrixLiteral(jbm)

    def render(self, r):
        return f'(JavaBlockMatrix {r.add_jir(self._jir)})'

    def _compute_type(self):
        self._type = tblockmatrix._from_java(self._jir.typ())


def _compute_shape(left_shape, right_shape):
    left_ndim, right_ndim = len(left_shape), len(right_shape)
    new_shape = collections.deque()
    for i in range(min(left_ndim, right_ndim)):
        left_dim_length = left_shape[-i - 1]
        right_dim_length = right_shape[-i - 1]
        assert left_dim_length == right_dim_length or left_dim_length == 1 or right_dim_length == 1

        new_shape.extendleft([max(left_dim_length, right_dim_length)])

    # Extend non-shared dimensions with whatever the higher-dimensional object has
    if left_ndim < right_ndim:
        return right_shape[:(right_ndim - left_ndim)] + list(new_shape)
    elif right_ndim < left_ndim:
        return left_shape[:(left_ndim - right_ndim)] + list(new_shape)

    return list(new_shape)
