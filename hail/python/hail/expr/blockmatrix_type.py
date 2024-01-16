import pprint

from hail.expr.types import dtype, hail_type
from hail.typecheck import sequenceof, typecheck_method
from hail.utils.java import jiterable_to_list


class tblockmatrix(object):
    __slots__ = 'element_type', 'shape', 'is_row_vector', 'block_size'

    @staticmethod
    def _from_java(jtbm):
        return tblockmatrix(
            dtype(jtbm.elementType().toString()), jiterable_to_list(jtbm.shape()), jtbm.isRowVector(), jtbm.blockSize()
        )

    @staticmethod
    def _from_json(json):
        return tblockmatrix(
            element_type=dtype(json['element_type']),
            shape=json['shape'],
            is_row_vector=json['is_row_vector'],
            block_size=json['block_size'],
        )

    @typecheck_method(element_type=hail_type, shape=sequenceof(int), is_row_vector=bool, block_size=int)
    def __init__(self, element_type, shape, is_row_vector, block_size):
        self.element_type = element_type
        self.shape = shape
        self.is_row_vector = is_row_vector
        self.block_size = block_size

    def to_dict(self):
        return dict(
            element_type=str(self.element_type),
            shape=self.shape,
            is_row_vector=self.is_row_vector,
            block_size=self.block_size,
        )

    def __eq__(self, other):
        return (
            isinstance(other, tblockmatrix)
            and self.element_type == other.element_type
            and self.shape == other.shape
            and self.is_row_vector == other.is_row_vector
            and self.block_size == other.block_size
        )

    def __hash__(self):
        return 43 + hash(str(self))

    def __repr__(self):
        return (
            f'tblockmatrix(element_type={self.element_type!r}, shape={self.shape!r}, '
            f'is_row_vector={self.is_row_vector!r}, block_size={self.block_size!r})'
        )

    def __str__(self):
        return (
            f'blockmatrix {{element_type: {self.element_type}, shape: {self.shape}, '
            f'is_row_vector: {self.is_row_vector}, block_size: {self.block_size})'
        )

    def pretty(self, indent=0, increment=4):
        b = []
        b.append(' ' * indent)
        b.append('blockmatrix {\n')
        indent += increment

        b.append(' ' * indent)
        b.append('element_type: ')
        self.element_type._pretty(b, indent, increment)
        b.append(',\n')

        b.append(' ' * indent)
        b.append(f'shape: [{self.shape}],\n')

        b.append(' ' * indent)
        b.append('is_row_vector: ')
        self.is_row_vector._pretty(b, indent, increment)
        b.append(',\n')

        b.append(' ' * indent)
        b.append('block_size: ')
        self.block_size._pretty(b, indent, increment)
        b.append(',\n')

        indent -= increment
        b.append(' ' * indent)
        b.append('}')

        return ''.join(b)


_old_printer = pprint.PrettyPrinter


class PrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, tblockmatrix):
            stream.write(object.pretty(self._indent_per_level))
        else:
            return _old_printer._format(self, object, stream, indent, allowance, context, level)


pprint.PrettyPrinter = PrettyPrinter  # monkey-patch pprint
