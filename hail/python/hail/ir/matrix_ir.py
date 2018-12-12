import json
from hail.ir.base_ir import *
from hail.utils.java import escape_str, escape_id, parsable_strings

class MatrixAggregateRowsByKey(MatrixIR):
    def __init__(self, child, entry_expr, row_expr):
        super().__init__()
        self.child = child
        self.entry_expr = entry_expr
        self.row_expr = row_expr

    def render(self, r):
        return f'(MatrixAggregateRowsByKey {r(self.child)} {r(self.entry_expr)} {r(self.row_expr)})'


class MatrixRead(MatrixIR):
    def __init__(self, reader, drop_cols=False, drop_rows=False):
        super().__init__()
        self.reader = reader
        self.drop_cols = drop_cols
        self.drop_rows = drop_rows

    def render(self, r):
        return f'(MatrixRead None {self.drop_cols} {self.drop_rows} "{r(self.reader)}")'


class MatrixFilterRows(MatrixIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def render(self, r):
        return '(MatrixFilterRows {} {})'.format(r(self.child), r(self.pred))

class MatrixChooseCols(MatrixIR):
    def __init__(self, child, old_entries):
        super().__init__()
        self.child = child
        self.old_entries = old_entries

    def render(self, r):
        return '(MatrixChooseCols ({}) {})'.format(
            ' '.join([str(i) for i in self.old_entries]), r(self.child))

class MatrixMapCols(MatrixIR):
    def __init__(self, child, new_col, new_key):
        super().__init__()
        self.child = child
        self.new_col = new_col
        self.new_key = new_key

    def render(self, r):
        return '(MatrixMapCols {} {} {})'.format(
            '(' + ' '.join(f'"{escape_str(f)}"' for f in self.new_key) + ')' if self.new_key is not None else 'None',
            r(self.child), r(self.new_col))

class MatrixMapEntries(MatrixIR):
    def __init__(self, child, new_entry):
        super().__init__()
        self.child = child
        self.new_entry = new_entry

    def render(self, r):
        return '(MatrixMapEntries {} {})'.format(r(self.child), r(self.new_entry))

class MatrixFilterEntries(MatrixIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def render(self, r):
        return '(MatrixFilterEntries {} {})'.format(r(self.child), r(self.pred))

class MatrixKeyRowsBy(MatrixIR):
    def __init__(self, child, keys, is_sorted=False):
        super().__init__()
        self.child = child
        self.keys = keys
        self.is_sorted = is_sorted

    def render(self, r):
        return '(MatrixKeyRowsBy ({}) {} {})'.format(
            ' '.join([escape_id(x) for x in self.keys]),
            self.is_sorted,
            r(self.child))

class MatrixMapRows(MatrixIR):
    def __init__(self, child, new_row):
        super().__init__()
        self.child = child
        self.new_row = new_row

    def render(self, r):
        return '(MatrixMapRows {} {})'.format(r(self.child), r(self.new_row))

class MatrixMapGlobals(MatrixIR):
    def __init__(self, child, new_row):
        super().__init__()
        self.child = child
        self.new_row = new_row

    def render(self, r):
        return f'(MatrixMapGlobals {r(self.child)} {r(self.new_row)})'

class MatrixFilterCols(MatrixIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def render(self, r):
        return f'(MatrixFilterCols {r(self.child)} {r(self.pred)})'

class MatrixCollectColsByKey(MatrixIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return f'(MatrixCollectColsByKey {r(self.child)})'

class MatrixAggregateColsByKey(MatrixIR):
    def __init__(self, child, entry_expr, col_expr):
        super().__init__()
        self.child = child
        self.entry_expr = entry_expr
        self.col_expr = col_expr

    def render(self, r):
        return '(MatrixAggregateColsByKey {} {} {})'.format(r(self.child), r(self.entry_expr), r(self.col_expr))


class TableToMatrixTable(MatrixIR):
    def __init__(self, child, row_key, col_key, row_fields, col_fields, n_partitions):
        super().__init__()
        self.child = child
        self.row_key = row_key
        self.col_key = col_key
        self.row_fields = row_fields
        self.col_fields = col_fields
        self.n_partitions = n_partitions

    def render(self, r):
        return f'(TableToMatrixTable ' \
               f'{parsable_strings(self.row_key)} ' \
               f'{parsable_strings(self.col_key)} ' \
               f'{parsable_strings(self.row_fields)} ' \
               f'{parsable_strings(self.col_fields)} ' \
               f'{"None" if self.n_partitions is None else str(self.n_partitions)} ' \
               f'{r(self.child)})'


class MatrixExplodeRows(MatrixIR):
    def __init__(self, child, path):
        super().__init__()
        self.child = child
        self.path = path

    def render(self, r):
        return '(MatrixExplodeRows ({}) {})'.format(
            ' '.join([escape_id(id) for id in self.path]),
            r(self.child))

class MatrixRepartition(MatrixIR):
    def __init__(self, child, n, shuffle):
        super().__init__()
        self.child = child
        self.n = n
        self.shuffle = shuffle

    def render(self, r):
        return f'(MatrixRepartition {r(self.child)} {self.n} {self.shuffle})'


class MatrixUnionRows(MatrixIR):
    def __init__(self, *children):
        super().__init__()
        self.children = children

    def render(self, r):
        return '(MatrixUnionRows {})'.format(' '.join(map(r, self.children)))


class MatrixDistinctByRow(MatrixIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return f'(MatrixDistinctByRow {r(self.child)})'


class MatrixExplodeCols(MatrixIR):
    def __init__(self, child, path):
        super().__init__()
        self.child = child
        self.path = path

    def render(self, r):
        return '(MatrixExplodeCols ({}) {})'.format(
            ' '.join([escape_id(id) for id in self.path]),
            r(self.child))


class CastTableToMatrix(MatrixIR):
    def __init__(self, child, entries_field_name, cols_field_name, col_key):
        super().__init__()
        self.child = child
        self.entries_field_name = entries_field_name
        self.cols_field_name = cols_field_name
        self.col_key = col_key

    def render(self, r):
        return '(CastTableToMatrix {} {} ({}) {})'.format(
           escape_str(self.entries_field_name),
           escape_str(self.cols_field_name),
           ' '.join([escape_id(id) for id in self.col_key]),
           r(self.child))


class MatrixAnnotateRowsTable(MatrixIR):
    def __init__(self, child, table, root, key):
        super().__init__()
        self.child = child
        self.table = table
        self.root = root
        self.key = key

    def render(self, r):
        if self.key is None:
            key_bool = False
            key_strs = ''
        else:
            key_bool = True
            key_strs = ' '.join(str(x) for x in self.key)
        return f'(MatrixAnnotateRowsTable "{self.root}" {key_bool} {r(self.child)} {r(self.table)} {key_strs})'

class MatrixAnnotateColsTable(MatrixIR):
    def __init__(self, child, table, root):
        super().__init__()
        self.child = child
        self.table = table
        self.root = root

    def render(self, r):
        return f'(MatrixAnnotateColsTable "{self.root}" {r(self.child)} {r(self.table)})'

class JavaMatrix(MatrixIR):
    def __init__(self, jir):
        self._jir = jir

    def render(self, r):
        return f'(JavaMatrix {r.add_jir(self._jir)})'
