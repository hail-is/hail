import hail as hl
from hail.expr.blockmatrix_type import tblockmatrix
from hail.expr.types import tarray
from .blockmatrix_reader import BlockMatrixReader
from .base_ir import BlockMatrixIR, IR
from hail.typecheck import typecheck_method, sequenceof
from hail.utils.misc import escape_id

from hail.utils.java import Env


class BlockMatrixRead(BlockMatrixIR):
    @typecheck_method(reader=BlockMatrixReader)
    def __init__(self, reader):
        super().__init__()
        self.reader = reader

    def head_str(self):
        return f'"{self.reader.render()}"'

    def _eq(self, other):
        return self.reader == other.reader

    def _compute_type(self):
        self._type = Env.backend().blockmatrix_type(self)


class BlockMatrixMap(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, name=str, f=IR, needs_dense=bool)
    def __init__(self, child, name, f, needs_dense):
        super().__init__(child, f)
        self.child = child
        self.name = name
        self.f = f
        self.needs_dense = needs_dense

    def _compute_type(self):
        self._type = self.child.typ

    def head_str(self):
        return escape_id(self.name) + " " + str(self.needs_dense)

    def bindings(self, i: int, default_value=None):
        if i == 1:
            value = self.child.typ.element_type if default_value is None else default_value
            return {self.name: value}
        else:
            return {}

    def binds(self, i):
        return {self.name} if i == 1 else {}


class BlockMatrixMap2(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR, left_name=str, right_name=str, f=IR, sparsity_strategy=str)
    def __init__(self, left, right, left_name, right_name, f, sparsity_strategy):
        super().__init__(left, right, f)
        self.left = left
        self.right = right
        self.left_name = left_name
        self.right_name = right_name
        self.f = f
        self.sparsity_strategy = sparsity_strategy

    def _compute_type(self):
        self.right.typ  # Force
        self._type = self.left.typ

    def head_str(self):
        return escape_id(self.left_name) + " " + escape_id(self.right_name) + " " + self.sparsity_strategy

    def bindings(self, i: int, default_value=None):
        if i == 2:
            if default_value is None:
                l_value = self.left.typ.element_type
                r_value = self.right.typ.element_type
            else:
                (l_value, r_value) = (default_value, default_value)
            return {self.left_name: l_value, self.right_name: r_value}
        else:
            return {}

    def binds(self, i):
        return {self.left_name, self.right_name} if i == 2 else {}


class BlockMatrixDot(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR)
    def __init__(self, left, right):
        super().__init__(left, right)
        self.left = left
        self.right = right

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
        super().__init__(child)
        self.child = child
        self.in_index_expr = in_index_expr
        self.shape = shape
        self.block_size = block_size

    def head_str(self):
        return '{} {} {}'.format(_serialize_list(self.in_index_expr),
                                 _serialize_list(self.shape),
                                 self.block_size)

    def _eq(self, other):
        return self.in_index_expr == other.in_index_expr and \
            self.shape == other.shape and \
            self.block_size == other.block_size

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
        super().__init__(child)
        self.child = child
        self.out_index_expr = out_index_expr

    def head_str(self):
        return _serialize_list(self.out_index_expr)

    def _eq(self, other):
        return self.out_index_expr == other.out_index_expr

    def _compute_type(self):
        child_matrix_shape = tensor_shape_to_matrix_shape(self.child)
        if self.out_index_expr == [0, 1]:
            is_row_vector = False
            shape = []
        elif self.out_index_expr == [0]:
            is_row_vector = True
            shape = [child_matrix_shape[1]]
        elif self.out_index_expr == [1]:
            is_row_vector = False
            shape = [child_matrix_shape[0]]
        else:
            raise ValueError("Invalid out_index_expr")

        self._type = tblockmatrix(self.child.typ.element_type,
                                  shape,
                                  is_row_vector,
                                  self.child.typ.block_size)


class BlockMatrixFilter(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, indices_to_keep=sequenceof(sequenceof(int)))
    def __init__(self, child, indices_to_keep):
        super().__init__(child)
        self.child = child
        self.indices_to_keep = indices_to_keep

    def head_str(self):
        return _serialize_list([_serialize_list(idxs) for idxs in self.indices_to_keep])

    def _eq(self, other):
        return self.indices_to_keep == other.indices_to_keep

    def _compute_type(self):
        assert len(self.indices_to_keep) == 2

        child_tensor_shape = self.child.typ.shape
        child_ndim = len(child_tensor_shape)
        if child_ndim == 1:
            if self.child.typ.is_row_vector:
                child_matrix_shape = [1, child_tensor_shape[0]]
            else:
                child_matrix_shape = [child_tensor_shape[0], 1]
        else:
            child_matrix_shape = child_tensor_shape

        matrix_shape = [len(idxs) if len(idxs) != 0 else child_matrix_shape[i] for i, idxs in
                        enumerate(self.indices_to_keep)]

        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(matrix_shape[0], matrix_shape[1])
        self._type = tblockmatrix(self.child.typ.element_type,
                                  tensor_shape,
                                  is_row_vector,
                                  self.child.typ.block_size)


class BlockMatrixDensify(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR)
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _compute_type(self):
        self._type = self.child.typ


class BlockMatrixSparsifier(object):
    def head_str(self):
        return ''

    def __repr__(self):
        head_str = self.head_str()
        if head_str != '':
            head_str = f' {head_str}'
        return f'(Py{self.__class__.__name__}{head_str})'

    def _eq(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._eq(other)


class BandSparsifier(BlockMatrixSparsifier):
    @typecheck_method(blocks_only=bool)
    def __init__(self, blocks_only):
        self.blocks_only = blocks_only

    def head_str(self):
        return str(self.blocks_only)

    def _eq(self, other):
        return self.blocks_only == other.blocks_only


class RowIntervalSparsifier(BlockMatrixSparsifier):
    @typecheck_method(blocks_only=bool)
    def __init__(self, blocks_only):
        self.blocks_only = blocks_only

    def head_str(self):
        return str(self.blocks_only)

    def _eq(self, other):
        return self.blocks_only == other.blocks_only


class _RectangleSparsifier(BlockMatrixSparsifier):
    def __init__(self):
        pass

    def __repr__(self):
        return '(PyRectangleSparsifier)'


RectangleSparsifier = _RectangleSparsifier()


class PerBlockSparsifier(BlockMatrixSparsifier):
    def __init__(self):
        pass

    def __repr__(self):
        return '(PyPerBlockSparsifier)'


class BlockMatrixSparsify(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, value=IR, sparsifier=BlockMatrixSparsifier)
    def __init__(self, child, value, sparsifier):
        super().__init__(value, child)
        self.child = child
        self.value = value
        self.sparsifier = sparsifier

    def head_str(self):
        return str(self.sparsifier)

    def _eq(self, other):
        return self.sparsifier == other.sparsifier

    def _compute_type(self):
        self._type = self.child.typ


class BlockMatrixSlice(BlockMatrixIR):
    @typecheck_method(child=BlockMatrixIR, slices=sequenceof(slice))
    def __init__(self, child, slices):
        super().__init__(child)
        self.child = child
        self.slices = slices

    def head_str(self):
        return '{}'.format(_serialize_list([f'({s.start} {s.stop} {s.step})' for s in self.slices]))

    def _eq(self, other):
        return self.slices == other.slices

    def _compute_type(self):
        assert len(self.slices) == 2
        matrix_shape = [1 + (s.stop - s.start - 1) // s.step for s in self.slices]
        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(matrix_shape[0], matrix_shape[1])
        self._type = tblockmatrix(self.child.typ.element_type,
                                  tensor_shape,
                                  is_row_vector,
                                  self.child.typ.block_size)


class ValueToBlockMatrix(BlockMatrixIR):
    @typecheck_method(child=IR,
                      shape=sequenceof(int),
                      block_size=int)
    def __init__(self, child, shape, block_size):
        super().__init__(child)
        self.child = child
        self.shape = shape
        self.block_size = block_size

    def head_str(self):
        return '{} {}'.format(_serialize_list(self.shape),
                              self.block_size)

    def _eq(self, other):
        return self.shape == other.shape and \
            self.block_size == other.block_size

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

    def head_str(self):
        return '{} {} {} {}'.format(self.seed,
                                    self.gaussian,
                                    _serialize_list(self.shape),
                                    self.block_size)

    def _eq(self, other):
        return self.seed == other.seed and \
            self.gaussian == other.gaussian and \
            self.shape == other.shape and \
            self.block_size == other.block_size

    def _compute_type(self):
        assert len(self.shape) == 2
        tensor_shape, is_row_vector = _matrix_shape_to_tensor_shape(self.shape[0], self.shape[1])

        self._type = tblockmatrix(hl.tfloat64, tensor_shape, is_row_vector, self.block_size)


class JavaBlockMatrix(BlockMatrixIR):
    def __init__(self, jbm):
        super().__init__()
        self.jir = Env.hail().expr.ir.BlockMatrixLiteral(jbm)

    def render_head(self, r):
        return f'(JavaBlockMatrix {r.add_jir(self.jir)}'

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
