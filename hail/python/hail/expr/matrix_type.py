import pprint

from hail.typecheck import typecheck_method, sequenceof
from hail.utils.java import escape_parsable
from hail.expr.types import dtype, tstruct
from hail.utils.java import jiterable_to_list


class tmatrix(object):
    __slots__ = 'global_type', 'col_type', 'col_key', 'row_type', 'row_key', 'entry_type'

    @staticmethod
    def _from_java(jtt):
        return tmatrix(
            dtype(jtt.globalType().toString()),
            dtype(jtt.colType().toString()),
            jiterable_to_list(jtt.colKey()),
            dtype(jtt.rowType().toString()),
            jiterable_to_list(jtt.rowKey()),
            dtype(jtt.entryType().toString()))

    @staticmethod
    def _from_json(json):
        return tmatrix(
            global_type=dtype(json['global_type']),
            col_type=dtype(json['col_type']),
            col_key=json['col_key'],
            row_type=dtype(json['row_type']),
            row_key=json['row_key'],
            entry_type=dtype(json['entry_type']))

    @typecheck_method(global_type=tstruct,
                      col_type=tstruct, col_key=sequenceof(str),
                      row_type=tstruct, row_key=sequenceof(str),
                      entry_type=tstruct)
    def __init__(self, global_type, col_type, col_key, row_type, row_key, entry_type):
        self.global_type = global_type
        self.col_type = col_type
        self.col_key = col_key
        self.row_type = row_type
        self.row_key = row_key
        self.entry_type = entry_type

    def to_dict(self):
        return dict(global_type=str(self.global_type),
                    col_type=str(self.col_type),
                    col_key=self.col_key,
                    row_type=str(self.row_type),
                    row_key=self.row_key,
                    entry_type=self.entry_type)

    def __eq__(self, other):
        return (isinstance(other, tmatrix)
                and self.global_type == other.global_type
                and self.col_type == other.col_type
                and self.col_key == other.col_key
                and self.row_type == other.row_type
                and self.row_key == other.row_key
                and self.entry_type == other.entry_type)

    def __hash__(self):
        return 43 + hash(str(self))

    def __repr__(self):
        return f'tmatrix(global_type={self.global_type!r}, col_type={self.col_type!r}, col_key={self.col_key!r}, row_type={self.row_type!r}, row_key={self.row_key!r}, entry_type={self.entry_type!r})'

    def _row_key_str(self):
        return ', '.join([escape_parsable(k) for k in self.row_key])

    def _col_key_str(self):
        return ', '.join([escape_parsable(k) for k in self.col_key])

    def __str__(self):
        return f'matrix {{global: {self.global_type}, col: {self.col_type}, col_key: {self._col_key_str()}, row: {self.row_type}, row_key: [{self._row_key_str()}], entry: {self.entry_type}}}'

    def pretty(self, indent=0, increment=4):
        b = []
        b.append(' ' * indent)
        b.append('matrix {\n')
        indent += increment

        b.append(' ' * indent)
        b.append('global: ')
        self.global_type._pretty(b, indent, increment)
        b.append(',\n')

        b.append(' ' * indent)
        b.append('row: ')
        self.row_type._pretty(b, indent, increment)
        b.append(',\n')

        b.append(' ' * indent)
        b.append(f'row_key: [{self._row_key_str()}],\n')

        b.append(' ' * indent)
        b.append('col: ')
        self.col_type._pretty(b, indent, increment)
        b.append(',\n')

        b.append(' ' * indent)
        b.append(f'col_key: [{self._col_key_str()}],\n')

        b.append(' ' * indent)
        b.append('entry: ')
        self.entry_type._pretty(b, indent, increment)
        b.append('\n')

        indent -= increment
        b.append(' ' * indent)
        b.append('}')

        return ''.join(b)

    @property
    def col_key_type(self):
        return self.col_type._select_fields(self.col_key)

    @property
    def col_value_type(self):
        return self.col_type._drop_fields(set(self.col_key))

    @property
    def row_key_type(self):
        return self.row_type._select_fields(self.row_key)

    @property
    def row_value_type(self):
        return self.row_type._drop_fields(set(self.row_key))

    def _rename(self, global_map, col_map, row_map, entry_map):
        return tmatrix(self.global_type._rename(global_map),
                       self.col_type._rename(col_map),
                       [col_map.get(k, k) for k in self.col_key],
                       self.row_type._rename(row_map),
                       [row_map.get(k, k) for k in self.row_key],
                       self.entry_type._rename(entry_map))

    def global_env(self, default_value=None):
        if default_value is None:
            return {'global': self.global_type}
        else:
            return {'global': default_value}

    def row_env(self, default_value=None):
        if default_value is None:
            return {'global': self.global_type,
                    'va': self.row_type}
        else:
            return {'global': default_value,
                    'va': default_value}

    def col_env(self, default_value=None):
        if default_value is None:
            return {'global': self.global_type,
                    'sa': self.col_type}
        else:
            return {'global': default_value,
                    'sa': default_value}

    def entry_env(self, default_value=None):
        if default_value is None:
            return {'global': self.global_type,
                    'va': self.row_type,
                    'sa': self.col_type,
                    'g': self.entry_type}
        else:
            return {'global': default_value,
                    'va': default_value,
                    'sa': default_value,
                    'g': default_value}


_old_printer = pprint.PrettyPrinter


class PrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, tmatrix):
            stream.write(object.pretty(self._indent_per_level))
        else:
            return _old_printer._format(self, object, stream, indent, allowance, context, level)


pprint.PrettyPrinter = PrettyPrinter  # monkey-patch pprint
