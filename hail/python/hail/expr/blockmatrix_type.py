from hail.typecheck import *
from hail.utils.java import jiterable_to_list
from hail.expr.types import dtype, hail_type


class tblockmatrix(object):
    @staticmethod
    def _from_java(jtbm):
        return tblockmatrix(
            dtype(jtbm.elementType().toString()),
            jiterable_to_list(jtbm.shape()),
            jtbm.isRowVector(),
            jtbm.blockSize(),
            jiterable_to_list(jtbm.dimsPartitioned()))

    @staticmethod
    def _from_json(json):
        return tblockmatrix(dtype(json['elementType']),
                            json['shape'],
                            json['isRowVector'],
                            json['blockSize'],
                            json['dimsPartitioned'])

    @typecheck_method(element_type=hail_type, shape=sequenceof(int), is_row_vector=bool,
                      block_size=int, dims_partitioned=sequenceof(bool))
    def __init__(self, element_type, shape, is_row_vector, block_size, dims_partitioned):
        self.element_type = element_type
        self.shape = shape
        self.is_row_vector = is_row_vector
        self.block_size = block_size
        self.dims_partitioned = dims_partitioned

    def __eq__(self, other):
        return isinstance(other, tblockmatrix) and \
               self.element_type == other.element_type and \
               self.shape == other.shape and \
               self.is_row_vector == other.is_row_vector and \
               self.block_size == other.block_size and \
               self.dims_partitioned == other.dims_partitioned

    def __hash__(self):
        return 43 + hash(str(self))

    def __repr__(self):
        return f'tblockmatrix(element_type={self.element_type!r}, shape={self.shape!r}, ' \
            f'is_row_vector={self.is_row_vector!r}, block_size={self.block_size!r}, ' \
            f'dims_partitioned={self.dims_partitioned!r})'

    def __str__(self):
        return f'blockmatrix {{element_type: {self.element_type}, shape: {self.shape}, ' \
            f'is_row_vector: {self.is_row_vector}, block_size: {self.block_size}, ' \
            f'dims_partitioned: {self.dims_partitioned}}}'

    def pretty(self, indent=0, increment=4):
        l = []
        l.append(' ' * indent)
        l.append('blockmatrix {\n')
        indent += increment

        l.append(' ' * indent)
        l.append('element_type: ')
        self.element_type._pretty(l, indent, increment)
        l.append(',\n')

        l.append(' ' * indent)
        l.append(f'shape: [{self.shape}],\n')

        l.append(' ' * indent)
        l.append('is_row_vector: ')
        self.is_row_vector._pretty(l, indent, increment)
        l.append(',\n')

        l.append(' ' * indent)
        l.append('block_size: ')
        self.block_size._pretty(l, indent, increment)
        l.append(',\n')

        l.append(' ' * indent)
        l.append(f'dims_partitioned: [{self.dims_partitioned}],\n')

        indent -= increment
        l.append(' ' * indent)
        l.append('}')

        return ''.join(l)


import pprint

_old_printer = pprint.PrettyPrinter


class PrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, tblockmatrix):
            stream.write(object.pretty(self._indent_per_level))
        else:
            return _old_printer._format(self, object, stream, indent, allowance, context, level)


pprint.PrettyPrinter = PrettyPrinter  # monkey-patch pprint

