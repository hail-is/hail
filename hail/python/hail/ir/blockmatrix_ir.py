from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir import hl
from hail.expr.types import tarray
from hail.ir import BlockMatrixIR, IR
from hail.ir.blockmatrix_reader import BlockMatrixReader
from hail.typecheck import typecheck_method, sequenceof

from hail.utils.java import Env


class BlockMatrixRead(BlockMatrixIR):
    @typecheck_method(reader=BlockMatrixReader)
    def __init__(self, reader):
        super().__init__()
        self.reader = reader

    def render(self, r):
        return f'(BlockMatrixRead "{r(self.reader)}")'

    def _compute_type(self):
        self._type = Env.backend().blockmatrix_type(self)


class BlockMatrixMap(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, f=IR)
    def __init__(self, child, f):
        super().__init__()
        self.child = child
        self.f = f

    def render(self, r):
        return f'(BlockMatrixMap {r(self.child)} {r(self.f)})'

    def _compute_type(self):
        self._type = self.child.typ


class BlockMatrixMap2(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR, f=IR)
    def __init__(self, left, right, f):
        super().__init__()
        self.left = left
        self.right = right
        self.f = f

    def render(self, r):
        return f'(BlockMatrixMap2 {r(self.left)} {r(self.right)} {r(self.f)})'

    def _compute_type(self):
        self.right.typ  # Force
        self._type = self.left.typ


class BlockMatrixDot(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR)
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def render(self, r):
        return '(BlockMatrixDot {} {})'.format(r(self.left), r(self.right))

    def _compute_type(self):
        l_rows, l_cols = tensor_shape_to_matrix_shape(self.left)
        r_rows, r_cols = tensor_shape_to_matrix_shape(self.right)
        assert l_cols == r_rows

        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(l_rows, r_cols)
        self._type = tblockmatrix(self.left.typ.element_type,
                                  tensor_shape,
                                  is_row_vector,
                                  self.left.typ.block_size)


class BlockMatrixBroadcast(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR,
                      in_index_expr=sequenceof(int),
                      shape=sequenceof(int),
                      block_size=int)
    def __init__(self, child, in_index_expr, shape, block_size):
        super().__init__()
        self.child = child
        self.in_index_expr = in_index_expr
        self.shape = shape
        self.block_size = block_size

    def render(self, r):
        return '(BlockMatrixBroadcast {} {} {} {})'\
            .format(_serialize_list(self.in_index_expr),
                    _serialize_list(self.shape),
                    self.block_size,
                    r(self.child))

    def _compute_type(self):
        assert len(self.shape) == 2
        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(self.shape[0], self.shape[1])
        self._type = tblockmatrix(self.child.typ.element_type,
                                  tensor_shape,
                                  is_row_vector,
                                  self.block_size)


class BlockMatrixAgg(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR,
                      out_index_expr=sequenceof(int))
    def __init__(self, child, out_index_expr):
        super().__init__()
        self.child = child
        self.out_index_expr = out_index_expr

    def render(self, r):
        return '(BlockMatrixAgg {} {})' \
            .format(_serialize_list(self.out_index_expr),
                    r(self.child))

    def _compute_type(self):
        shape = [self.child.typ.shape[i] for i in self.out_index_expr]
        is_row_vector = self.out_index_expr == [1]

        self._type = tblockmatrix(self.child.typ.element_type,
                                  shape,
                                  is_row_vector,
                                  self.child.typ.block_size)


class ValueToBlockMatrix(BlockMatrixIR):
    @typecheck_method(child=IR,
                      shape=sequenceof(int),
                      block_size=int)
    def __init__(self, child, shape, block_size):
        super().__init__()
        self.child = child
        self.shape = shape
        self.block_size = block_size

    def render(self, r):
        return '(ValueToBlockMatrix {} {} {})'.format(_serialize_list(self.shape),
                                                      self.block_size,
                                                      r(self.child))

    def _compute_type(self):
        child_type = self.child.typ
        if isinstance(child_type, tarray):
            element_type = child_type._element_type
        else:
            element_type = child_type

        assert len(self.shape) == 2
        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(self.shape[0], self.shape[1])
        self._type = tblockmatrix(element_type, tensor_shape, is_row_vector, self.block_size)


class BlockMatrixRandom(BlockMatrixIR):
    @typecheck_method(seed=int,
                      gaussian=bool,
                      shape=sequenceof(int),
                      block_size=int)
    def __init__(self, seed, gaussian, shape, block_size):
        super().__init__()
        self.seed = seed
        self.gaussian = gaussian
        self.shape = shape
        self.block_size = block_size

    def render(self, r):
        return '(BlockMatrixRandom {} {} {} {})'.format(self.seed,
                                                        self.gaussian,
                                                        _serialize_list(self.shape),
                                                        self.block_size)

    def _compute_type(self):
        assert len(self.shape) == 2
        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(self.shape[0], self.shape[1])

        self._type = tblockmatrix(hl.tfloat64, tensor_shape, is_row_vector, self.block_size)


class JavaBlockMatrix(BlockMatrixIR):
    def __init__(self, jbm):
        super().__init__()
        self.jir = Env.hail().expr.ir.BlockMatrixLiteral(jbm)

    def render(self, r):
        return f'(JavaBlockMatrix {r.add_jir(self.jir)})'

    def _compute_type(self):
        self._type = tblockmatrix._from_java(self.jir.typ())


def tensor_shape_to_matrix_shape(bmir):
    shape = bmir.typ.shape
    is_row_vector = bmir.typ.is_row_vector

    assert len(shape) <= 2
    if len(shape) == 0:
        return (1, 1)
    elif len(shape) == 1:
        length = shape[0]
        return (1, length) if is_row_vector else (length, 1)
    else:
        return tuple(shape)


def _serialize_list(xs):
    return "(" + ' '.join([str(x) for x in xs]) + ")"


def _matrix_shape_to_tensor_shape(n_rows, n_cols):
    if n_rows == 1 and n_cols == 1:
        return [], False
    elif n_rows == 1:
        return [n_cols], True
    elif n_cols == 1:
        return [n_rows], False
    else:
        return [n_rows, n_cols], False
