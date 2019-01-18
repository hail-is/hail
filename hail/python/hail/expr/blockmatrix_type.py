from hail.typecheck import *
from hail.utils.java import jiterable_to_list


class tblockmatrix(object):
    @staticmethod
    def _from_java(jtbm):
        return tblockmatrix(
            jiterable_to_list(jtbm.shape()),
            jtbm.blockSize(),
            jiterable_to_list(jtbm.dimsPartitioned()))

    @staticmethod
    def _from_json(json):
        return tblockmatrix(json['shape'], json['blockSize'], json['dimsPartitioned'])

    @typecheck_method(shape=sequenceof(int), block_size=int, dims_partitioned=sequenceof(bool))
    def __init__(self, shape, block_size, dims_partitioned):
        self.shape = shape
        self.block_size = block_size,
        self.dims_partitioned = dims_partitioned

    def __eq__(self, other):
        return isinstance(other, tblockmatrix) and \
               self.shape == other.shape and \
               self.block_size == other.block_size and \
               self.dims_partitioned == other.dims_partitioned

    def __hash__(self):
        return 43 + hash(str(self))

    def __repr__(self):
        return f'tblockmatrix(shape={self.shape!r}, block_size={self.block_size!r}, dims_partitioned={self.dims_partitioned!r})'

    def __str__(self):
        return f'blockmatrix {{shape: {self.shape}, block_size: {self.block_size}, dims_partitioned: {self.dims_partitioned}}}'

    def pretty(self, indent=0, increment=4):
        l = []
        l.append(' ' * indent)
        l.append('blockmatrix {\n')
        indent += increment

        l.append(' ' * indent)
        l.append(f'shape: [{self.shape}],\n')

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

