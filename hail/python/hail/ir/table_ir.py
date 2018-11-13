import json

from hail.ir.base_ir import *
from hail.utils.java import escape_str, escape_id, parsable_strings


class MatrixRowsTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return '(MatrixRowsTable {})'.format(r(self.child))


class TableJoin(TableIR):
    def __init__(self, left, right, join_type, join_key):
        super().__init__()
        self.left = left
        self.right = right
        self.join_type = join_type
        self.join_key = join_key

    def render(self, r):
        return '(TableJoin {} {} {} {})'.format(
            escape_id(self.join_type), self.join_key, r(self.left), r(self.right))


class TableUnion(TableIR):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def render(self, r):
        return '(TableUnion {})'.format(' '.join([r(x) for x in self.children]))


class TableRange(TableIR):
    def __init__(self, n, n_partitions):
        super().__init__()
        self.n = n
        self.n_partitions = n_partitions

    def render(self, r):
        return '(TableRange {} {})'.format(self.n, self.n_partitions)


class TableMapGlobals(TableIR):
    def __init__(self, child, new_row):
        super().__init__()
        self.child = child
        self.new_row = new_row

    def render(self, r):
        return '(TableMapGlobals {} {})'.format(r(self.child), r(self.new_row))


class TableExplode(TableIR):
    def __init__(self, child, field):
        super().__init__()
        self.child = child
        self.field = field

    def render(self, r):
        return '(TableExplode {} {})'.format(escape_id(self.field), r(self.child))


class TableKeyBy(TableIR):
    def __init__(self, child, keys, is_sorted=False):
        super().__init__()
        self.child = child
        self.keys = keys
        self.is_sorted = is_sorted

    def render(self, r):
        return '(TableKeyBy ({}) {} {})'.format(
            ' '.join([escape_id(x) for x in self.keys]),
            self.is_sorted,
            r(self.child))


class TableMapRows(TableIR):
    def __init__(self, child, new_row):
        super().__init__()
        self.child = child
        self.new_row = new_row

    def render(self, r):
        return '(TableMapRows {} {})'.format(r(self.child), r(self.new_row))


class TableRead(TableIR):
    def __init__(self, path, drop_rows, typ):
        super().__init__()
        self.path = path
        self.drop_rows = drop_rows
        self.typ = typ

    def render(self, r):
        return '(TableRead "{}" {} {})'.format(
            escape_str(self.path),
            self.drop_rows,
            self.typ)


class TableImport(TableIR):
    def __init__(self, paths, typ, reader_options):
        super().__init__()
        self.paths = paths
        self.typ = typ
        self.reader_options = reader_options

    def render(self, r):
        return '(TableImport ({}) {} {})'.format(
            ' '.join([escape_str(path) for path in self.paths]),
            self.typ._parsable_string(),
            escape_str(json.dumps(self.reader_options)))


class MatrixEntriesTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return '(MatrixEntriesTable {})'.format(r(self.child))


class TableFilter(TableIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def render(self, r):
        return '(TableFilter {} {})'.format(r(self.child), r(self.pred))


class TableKeyByAndAggregate(TableIR):
    def __init__(self, child, expr, new_key, n_partitions, buffer_size):
        super().__init__()
        self.child = child
        self.expr = expr
        self.new_key = new_key
        self.n_partitions = n_partitions
        self.buffer_size = buffer_size

    def render(self, r):
        return '(TableKeyByAndAggregate {} {} {} {} {})'.format(self.n_partitions,
                                                                self.buffer_size,
                                                                r(self.child),
                                                                r(self.expr),
                                                                self.new_key)


class TableAggregateByKey(TableIR):
    def __init__(self, child, expr):
        super().__init__()
        self.child = child
        self.expr = expr

    def render(self, r):
        return '(TableAggregateByKey {} {})'.format(r(self.child), r(self.expr))


class MatrixColsTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return '(MatrixColsTable {})'.format(r(self.child))


class TableParallelize(TableIR):
    def __init__(self, rows, n_partitions):
        super().__init__()
        self.rows = rows
        self.n_partitions = n_partitions

    def render(self, r):
        return '(TableParallelize {} {})'.format(
            self.n_partitions,
            r(self.rows))


class TableHead(TableIR):
    def __init__(self, child, n):
        super().__init__()
        self.child = child
        self.n = n

    def render(self, r):
        return f'(TableHead {self.n} {r(self.child)})'


class TableOrderBy(TableIR):
    def __init__(self, child, sort_fields):
        super().__init__()
        self.child = child
        self.sort_fields = sort_fields

    def render(self, r):
        return '(TableOrderBy ({}) {})'.format(
            ' '.join(['{}{}'.format(order, escape_id(f)) for (f, order) in self.sort_fields]),
            r(self.child))


class TableDistinct(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return f'(TableDistinct {r(self.child)})'


class TableRepartition(TableIR):
    def __init__(self, child, n, shuffle):
        super().__init__()
        self.child = child
        self.n = n
        self.shuffle = shuffle

    def render(self, r):
        return f'(TableRepartition {self.n} {self.shuffle} {r(self.child)})'


class CastMatrixToTable(TableIR):
    def __init__(self, child, entries_field_name, cols_field_name):
        super().__init__()
        self.child = child
        self.entries_field_name = entries_field_name
        self.cols_field_name = cols_field_name

    def render(self, r):
        return f'(CastMatrixToTable ' \
               f'"{escape_str(self.entries_field_name)}" ' \
               f'"{escape_str(self.cols_field_name)}" ' \
               f'{r(self.child)})'


class TableRename(TableIR):
    def __init__(self, child, row_map, global_map):
        super().__init__()
        self.child = child
        self.row_map = row_map
        self.global_map = global_map

    def render(self, r):
        return f'(TableRename ' \
               f'{parsable_strings(self.row_map.keys())} ' \
               f'{parsable_strings(self.row_map.values())} ' \
               f'{parsable_strings(self.global_map.keys())} ' \
               f'{parsable_strings(self.global_map.values())} ' \
               f'{r(self.child)})'


class TableMultiWayZipJoin(TableIR):
    def __init__(self, childs, data_name, global_name):
        super().__init__()
        self.childs = childs
        self.data_name = data_name
        self.global_name = global_name

    def render(self, r):
        return f'(TableMultiWayZipJoin '\
               f'"{escape_str(self.data_name)}" '\
               f'"{escape_str(self.global_name)}" '\
               f'{" ".join([r(child) for child in self.childs])})'


class JavaTable(TableIR):
    def __init__(self, jir):
        self._jir = jir

    def render(self, r):
        return f'(JavaTable {r.add_jir(self._jir)})'
