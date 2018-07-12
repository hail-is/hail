import json

from hail.ir.base_ir import *
from hail.utils.java import escape_str, escape_id


class MatrixRowsTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def __str__(self):
        return '(MatrixRowsTable {})'.format(self.child)


class TableJoin(TableIR):
    def __init__(self, left, right, join_type):
        super().__init__()
        self.left = left
        self.right = right
        self.join_type = join_type

    def __str__(self):
        return '(TableJoin {} {} {})'.format(
            escape_id(self.join_type), self.left, self.right)


class TableUnion(TableIR):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def __str__(self):
        return '(TableUnion {})'.format(' '.join([str(x) for x in self.children]))


class TableRange(TableIR):
    def __init__(self, n, n_partitions):
        super().__init__()
        self.n = n
        self.n_partitions = n_partitions

    def __str__(self):
        return '(TableRange {} {})'.format(self.n, self.n_partitions)


class TableMapGlobals(TableIR):
    def __init__(self, child, new_row, value):
        super().__init__()
        self.child = child
        self.new_row = new_row
        self.value = value

    def __str__(self):
        return '(TableMapGlobals {} {} {})'.format(
            self.value, self.child, self.new_row)


class TableExplode(TableIR):
    def __init__(self, child, field):
        super().__init__()
        self.child = child
        self.field = field

    def __str__(self):
        return '(TableExplode {} {})'.format(escape_id(self.field), self.child)


class TableKeyBy(TableIR):
    def __init__(self, child, keys, n_partitions, sort):
        super().__init__()
        self.child = child
        self.keys = keys
        self.n_partitions = n_partitions
        self.sort = sort

    def __str__(self):
        return '(TableKeyBy ({}) {} {} {})'.format(
            ' '.join([escape_id(x) for x in self.keys]), self.n_partitions, self.sort,
            self.child)


class TableMapRows(TableIR):
    def __init__(self, child, new_row, new_key, preserved_key_fields):
        super().__init__()
        self.child = child
        self.new_row = new_row
        self.new_key = new_key
        self.preserved_key_fields = preserved_key_fields

    def __str__(self):
        return '(TableMapRows {} {} {} {})'.format(
            ' '.join([escape_id(x) for x in self.new_key]) if self.new_key else 'None',
            self.preserved_key_fields,
            self.child, self.new_row)


class TableUnkey(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def __str__(self):
        return '(TableUnkey {})'.format(self.child)


class TableRead(TableIR):
    def __init__(self, path, drop_rows, typ):
        super().__init__()
        self.path = path
        self.drop_rows = drop_rows
        self.typ = typ

    def __str__(self):
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

    def __str__(self):
        return '(TableImport ({}) {} {})'.format(
            ' '.join([escape_str(path) for path in self.paths]),
            self.typ._jtype.parsableString(),
            escape_str(json.dumps(self.reader_options)))


class MatrixEntriesTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def __str__(self):
        return '(MatrixEntriesTable {})'.format(self.child)


class TableFilter(TableIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def __str__(self):
        return '(TableFilter {} {})'.format(self.child, self.pred)


class TableAggregateByKey(TableIR):
    def __init__(self, child, expr):
        super().__init__()
        self.child = child
        self.expr = expr

    def __str__(self):
        return '(TableAggregateByKey {} {})'.format(self.child, self.expr)


class MatrixColsTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def __str__(self):
        return '(MatrixColsTable {})'.format(self.child)


class TableParallelize(TableIR):
    def __init__(self, typ, rows, n_partitions):
        super().__init__()
        self.typ = typ
        self.rows = rows
        self.n_partitions = n_partitions

    def __str__(self):
        return '(TableParallelize {} {} {})'.format(
            self.typ,
            self.rows,
            self.n_partitions)


class TableOrderBy(TableIR):
    def __init__(self, child, sort_fields):
        super().__init__()
        self.child = child
        self.sort_fields = sort_fields

    def __str__(self):
        return '(TableOrderBy ({}) {})'.format(
            ' '.join(['{}{}'.format(order, escape_id(f)) for (f, order) in self.sort_fields]),
            self.child)


class TableDistinct(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def __str__(self):
        return f'(TableDistinct {self.child})'
