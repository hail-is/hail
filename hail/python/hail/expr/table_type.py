import pprint
from hail.typecheck import typecheck_method, sequenceof
from hail.utils.java import escape_parsable
from hail.expr.types import dtype, tstruct
from hail.utils.java import jiterable_to_list


class ttable(object):
    __slots__ = 'global_type', 'row_type', 'row_key'

    @staticmethod
    def _from_java(jtt):
        return ttable(dtype(jtt.globalType().toString()), dtype(jtt.rowType().toString()), jiterable_to_list(jtt.key()))

    @staticmethod
    def _from_json(json):
        return ttable(global_type=dtype(json['global_type']), row_type=dtype(json['row_type']), row_key=json['row_key'])

    @typecheck_method(global_type=tstruct, row_type=tstruct, row_key=sequenceof(str))
    def __init__(self, global_type, row_type, row_key):
        self.global_type = global_type
        self.row_type = row_type
        self.row_key = row_key

    def to_dict(self):
        return dict(global_type=str(self.global_type), row_type=str(self.row_type), row_key=self.row_key)

    def __eq__(self, other):
        return (
            isinstance(other, ttable)
            and self.global_type == other.global_type
            and self.row_type == other.row_type
            and self.row_key == other.row_key
        )

    def __hash__(self):
        return 43 + hash(str(self))

    def __repr__(self):
        return f'ttable(global_type={self.global_type!r}, row_type={self.row_type!r}, row_key={self.row_key!r})'

    def _key_str(self):
        return ', '.join([escape_parsable(k) for k in self.row_key])

    def __str__(self):
        return f'table {{global: {self.global_type}, row: {self.row_type}, row_key: [{self._key_str()}]}}'

    def pretty(self, indent=0, increment=4):
        b = []
        b.append(' ' * indent)
        b.append('table {\n')
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
        b.append(f'row_key: [{self._key_str()}]\n')

        indent -= increment
        b.append(' ' * indent)
        b.append('}')

        return ''.join(b)

    @property
    def key_type(self):
        return self.row_type._select_fields(self.row_key)

    @property
    def value_type(self):
        return self.row_type._drop_fields(set(self.row_key))

    def _rename(self, global_map, row_map):
        return ttable(
            self.global_type._rename(global_map),
            self.row_type._rename(row_map),
            [row_map.get(k, k) for k in self.row_key],
        )

    def row_env(self, default_value=None):
        if default_value is None:
            return {'global': self.global_type, 'row': self.row_type}
        else:
            return {'global': default_value, 'row': default_value}

    def global_env(self, default_value=None):
        if default_value is None:
            return {'global': self.global_type}
        else:
            return {'global': default_value}


_old_printer = pprint.PrettyPrinter


class PrettyPrinter(pprint.PrettyPrinter):
    def _format(self, object, stream, indent, allowance, context, level):
        if isinstance(object, ttable):
            stream.write(object.pretty(self._indent_per_level))
        else:
            return _old_printer._format(self, object, stream, indent, allowance, context, level)


pprint.PrettyPrinter = PrettyPrinter  # monkey-patch pprint
