from hail.typecheck import typecheck_method
from hail.expr.types import dtype


class tblockmatrix(object):
    @staticmethod
    def _from_java(jtbm):
        return tblockmatrix(jtbm.nRows(), jtbm.nCols(), jtbm.blockSize())

    @staticmethod
    def _from_json(json):
        return tblockmatrix(
            dtype(json['nRows']),
            dtype(json['nCols']),
            dtype(json['blockSize']))

    @typecheck_method(n_rows=int, n_cols=int, block_size=int)
    def __init__(self, n_rows, n_cols, block_size):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.block_size = block_size

    def __eq__(self, other):
        return isinstance(other, tblockmatrix) and \
               self.n_rows == other.n_rows and \
               self.n_cols == other.n_cols and \
               self.block_size == other.block_size

    def __hash__(self):
        return 43 + hash(str(self))

    def __repr__(self):
        return f'tblockmatrix(n_rows={self.n_rows!r}, n_cols={self.n_cols!r}, block_size={self.block_size!r})'

    def __str__(self):
        return f'blockmatrix {{n_rows: {self.n_rows}, n_cols: {self.n_cols}, block_size: {self.block_size}}}'

    def pretty(self, indent=0, increment=4):
        l = []
        l.append(' ' * indent)
        l.append('blockmatrix {\n')
        indent += increment

        l.append(' ' * indent)
        l.append('n_rows: ')
        self.n_rows._pretty(l, indent, increment)
        l.append(',\n')

        l.append(' ' * indent)
        l.append('n_cols: ')
        self.n_cols._pretty(l, indent, increment)
        l.append(',\n')

        l.append(' ' * indent)
        l.append('block_size: ')
        self.block_size._pretty(l, indent, increment)
        l.append(',\n')

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

