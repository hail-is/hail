from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir import BlockMatrixIR, IR, tarray
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

        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape([l_rows, r_cols])
        self._type = tblockmatrix(self.left.typ.element_type,
                                  tensor_shape, is_row_vector,
                                  self.left.typ.block_size,
                                  [True for _ in tensor_shape])


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
        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(self.shape)
        self._type = tblockmatrix(self.child.typ.element_type,
                                  tensor_shape, is_row_vector,
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

        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(self.shape)
        self._type = tblockmatrix(element_type,
                                  tensor_shape, is_row_vector,
                                  self.block_size, self.dims_partitioned)


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


def tensor_shape_to_matrix_shape(bmir):
    shape = bmir.typ.shape
    is_row_vector = bmir.typ.is_row_vector

    assert len(shape) <= 2
    if len(shape) == 0:
        return 1, 1
    elif len(shape) == 1:
        length = shape[0]
        return (1, length) if is_row_vector else (length, 1)
    else:
        return tuple(shape)
