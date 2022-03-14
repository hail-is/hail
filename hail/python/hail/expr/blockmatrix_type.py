import pprint
from hail.typecheck import typecheck_method, sequenceof
from hail.utils.java import jiterable_to_list
from hail.expr.types import dtype, hail_type


class tblockmatrix(object):
    @staticmethod
    def _from_java(jtbm):
        return tblockmatrix(
            dtype(jtbm.elementType().toString()),
            jiterable_to_list(jtbm.shape()),
            jtbm.isRowVector(),
            jtbm.blockSize())

    @staticmethod
    def _from_json(json):
        return tblockmatrix(dtype(json['element_type']),
                            json['shape'],
                            json['is_row_vector'],
                            json['block_size'])

    @typecheck_method(element_type=hail_type, shape=sequenceof(int), is_row_vector=bool, block_size=int)
    def __init__(self, element_type, shape, is_row_vector, block_size):
        self.element_type = element_type
        self.shape = shape
        self.is_row_vector = is_row_vector
        self.block_size = block_size

    def __eq__(self, other):
        return isinstance(other, tblockmatrix) and \
            self.element_type == other.element_type and \
            self.shape == other.shape and \
            self.is_row_vector == other.is_row_vector and \
            self.block_size == other.block_size

    def __hash__(self):
        return 43 + hash(str(self))

    def __repr__(self):
        return f'tblockmatrix(element_type={self.element_type!r}, shape={self.shape!r}, ' \
            f'is_row_vector={self.is_row_vector!r}, block_size={self.block_size!r})'

    def __str__(self):
        return f'blockmatrix {{element_type: {self.element_type}, shape: {self.shape}, ' \
            f'is_row_vector: {self.is_row_vector}, block_size: {self.block_size})'

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


def pprint_hail_blockmatrix(printer, obj, stream, indent, allowance, context, level):
    # https://stackoverflow.com/a/40828239/6823256
    stream.write(obj.pretty(printer._indent_per_level))


assert hasattr(pprint.PrettyPrinter, '_dispatch')
pprint.PrettyPrinter._dispatch[tblockmatrix.__repr__] = pprint_hail_blockmatrix  # https://stackoverflow.com/a/40828239/6823256
