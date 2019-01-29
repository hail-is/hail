from hail.typecheck import *
from hail.utils.java import escape_parsable
from hail.expr.types import dtype, tstruct
from hail.utils.java import jiterable_to_list

class ttable(object):
    @staticmethod
    def _from_java(jtt):
        return ttable(
            dtype(jtt.globalType().toString()),
            dtype(jtt.rowType().toString()),
            jiterable_to_list(jtt.key()))

    @staticmethod
    def _from_json(json):
        return ttable(
            dtype(json['global']),
            dtype(json['row']),
            json['row_key'])

    @typecheck_method(global_type=tstruct, row_type=tstruct, row_key=sequenceof(str))
    def __init__(self, global_type, row_type, row_key):
        self.global_type = global_type
        self.row_type = row_type
        self.row_key = row_key

    def __eq__(self, other):
        return (isinstance(other, ttable)
                and self.global_type == other.global_type
                and self.row_type == other.row_type
                and self.row_key == other.row_key)

    def __hash__(self):
        return 43 + hash(str(self))

    def __repr__(self):
        return f'ttable(global_type={self.global_type!r}, row_type={self.row_type!r}, row_key={self.row_key!r})'

    def _key_str(self):
        return ', '.join([escape_parsable(k) for k in self.row_key])

    def __str__(self):
        return f'table {{global: {self.global_type}, row: {self.row_type}, row_key: [{self._key_str()}]}}'

    def pretty(self, indent=0, increment=4):
        l = []
        l.append(' ' * indent)
        l.append('table {\n')
        indent += increment
        
        l.append(' ' * indent)
        l.append('global: ')
        self.global_type._pretty(l, indent, increment)
        l.append(',\n')
        
        l.append(' ' * indent)
        l.append('row: ')
        self.row_type._pretty(l, indent, increment)
        l.append(',\n')
        
        l.append(' ' * indent)
        l.append(f'row_key: [{self._key_str()}]\n')

        indent -= increment
        l.append(' ' * indent)
        l.append('}')
        
        return ''.join(l)

    @property
    def key_type(self):
        return self.row_type._select_fields(self.row_key)

    @property
    def value_type(self):
        return self.row_type._drop_fields(set(self.row_key))

    def _rename(self, global_map, row_map):
        return ttable(self.global_type._rename(global_map),
                      self.row_type._rename(row_map),
                      [row_map.get(k, k) for k in self.row_key])

    def row_env(self):
        return {'global': self.global_type,
                'row': self.row_type}

    def global_env(self):
        return {'global': self.global_type}


import pprint

_old_printer = pprint.PrettyPrinter


class PrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, ttable):
            stream.write(object.pretty(self._indent_per_level))
        else:
            return _old_printer._format(self, object, stream, indent, allowance, context, level)


pprint.PrettyPrinter = PrettyPrinter  # monkey-patch pprint
