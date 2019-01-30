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
                                  _compute_shape_after_broadcast(left_type.shape, right_type.shape),
                                  left_type.block_size,
                                  left_type.dims_partitioned)


class BlockMatrixMap(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, apply_bin_op=ApplyBinaryOp)
    def __init__(self, child, apply_bin_op):
        super().__init__()
        self._child = child
        self._apply_bin_op = apply_bin_op

    def render(self, r):
        return f'(BlockMatrixMap {r(self._child)} {r(self._apply_bin_op)})'

    def _compute_type(self):
        child_type = self._child.typ
        self._type = tblockmatrix(child_type.element_type,
                                  _shape_after_broadcast(*self._get_children_shapes()),
                                  child_type.block_size,
                                  child_type.dims_partitioned)

    def _get_children_shapes(self):
        child_matrix = self._child
        left, right = self._apply_bin_op.l, self._apply_bin_op.r

        if isinstance(left, Ref):
            left_shape = child_matrix.typ.shape
            if isinstance(right, F64):
                right_shape = [1, 1]
            else:
                right_shape = right.fields[0][1].value
        else:
            right_shape = child_matrix.typ.shape
            if isinstance(left, F64):
                left_shape = [1, 1]
            else:
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


def _shape_after_broadcast(left_shape, right_shape):
    def _calc_new_dim(l_dim_length, r_dim_length):
        if not (l_dim_length == r_dim_length or l_dim_length == 1 or r_dim_length == 1):
            raise ValueError(f'Incompatible shapes for broadcasting: {left_shape}, {right_shape}')

        return max(l_dim_length, r_dim_length)

    left_ndim, right_ndim = len(left_shape), len(right_shape)
    if left_ndim < right_ndim:
        left_shape = [1 for _ in range(right_ndim - left_ndim)].extend(left_shape)
    elif right_ndim < left_ndim:
        right_shape = [1 for _ in range(left_ndim - right_ndim)].extend(right_shape)

    assert(len(left_shape) == len(right_shape))
    return map(_calc_new_dim, left_shape, right_shape)
