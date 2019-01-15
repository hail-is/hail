import json
import hail as hl
from hail.ir.base_ir import *
from hail.utils.java import escape_str, escape_id, parsable_strings, dump_json

class MatrixAggregateRowsByKey(MatrixIR):
    def __init__(self, child, entry_expr, row_expr):
        super().__init__()
        self.child = child
        self.entry_expr = entry_expr
        self.row_expr = row_expr

    def render(self, r):
        return f'(MatrixAggregateRowsByKey {r(self.child)} {r(self.entry_expr)} {r(self.row_expr)})'

    def _compute_type(self):
        child_typ = self.child.typ
        self.entry_expr._compute_type(child_typ.col_env(), child_typ.entry_env())
        self.row_expr._compute_type(child_typ.global_env(), child_typ.row_env())
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_key_type._concat(self.row_expr.typ),
            child_typ.row_key,
            self.entry_expr.typ)


class MatrixRead(MatrixIR):
    def __init__(self, reader, drop_cols=False, drop_rows=False):
        super().__init__()
        self.reader = reader
        self.drop_cols = drop_cols
        self.drop_rows = drop_rows

    def render(self, r):
        return f'(MatrixRead None {self.drop_cols} {self.drop_rows} "{r(self.reader)}")'

    def _compute_type(self):
        self._type = Env.backend().matrix_type(self)


class MatrixFilterRows(MatrixIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def render(self, r):
        return '(MatrixFilterRows {} {})'.format(r(self.child), r(self.pred))

    def _compute_type(self):
        self.pred._compute_type(self.child.typ.row_env(), None)
        self._type = self.child.typ

class MatrixChooseCols(MatrixIR):
    def __init__(self, child, old_indices):
        super().__init__()
        self.child = child
        self.old_indices = old_indices

    def render(self, r):
        return '(MatrixChooseCols ({}) {})'.format(
            ' '.join([str(i) for i in self.old_indices]), r(self.child))

    def _compute_type(self):
        self._type = self.child.typ

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

    def _compute_type(self):
        child_typ = self.child.typ
        self.new_col._compute_type(child_typ.col_env(), child_typ.entry_env())
        self._type = hl.tmatrix(
            child_typ.global_type,
            self.new_col.typ,
            self.new_key if self.new_key is not None else child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            child_typ.entry_type)

class MatrixUnionCols(MatrixIR):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def render(self, r):
        return f'(MatrixUnionCols {r(self.left)} {r(self.right)})'

    def _compute_type(self):
        self.right.typ # force
        self._type = self.left.typ

class MatrixMapEntries(MatrixIR):
    def __init__(self, child, new_entry):
        super().__init__()
        self.child = child
        self.new_entry = new_entry

    def render(self, r):
        return '(MatrixMapEntries {} {})'.format(r(self.child), r(self.new_entry))

    def _compute_type(self):
        child_typ = self.child.typ
        self.new_entry._compute_type(child_typ.entry_env(), None)
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            self.new_entry.typ)


class MatrixFilterEntries(MatrixIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def render(self, r):
        return '(MatrixFilterEntries {} {})'.format(r(self.child), r(self.pred))

    def _compute_type(self):
        self.pred._compute_type(self.child.typ.entry_env(), None)
        self._type = self.child.typ

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

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_type,
            self.keys,
            child_typ.entry_type)


class MatrixMapRows(MatrixIR):
    def __init__(self, child, new_row):
        super().__init__()
        self.child = child
        self.new_row = new_row

    def render(self, r):
        return '(MatrixMapRows {} {})'.format(r(self.child), r(self.new_row))

    def _compute_type(self):
        child_typ = self.child.typ
        self.new_row._compute_type(child_typ.row_env(), child_typ.entry_env())
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            self.new_row.typ,
            child_typ.row_key,
            child_typ.entry_type)

class MatrixMapGlobals(MatrixIR):
    def __init__(self, child, new_global):
        super().__init__()
        self.child = child
        self.new_global = new_global

    def render(self, r):
        return f'(MatrixMapGlobals {r(self.child)} {r(self.new_global)})'

    def _compute_type(self):
        child_typ = self.child.typ
        self.new_global._compute_type(child_typ.global_env(), None)
        self._type = hl.tmatrix(
            self.new_global.typ,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            child_typ.entry_type)


class MatrixFilterCols(MatrixIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def render(self, r):
        return f'(MatrixFilterCols {r(self.child)} {r(self.pred)})'

    def _compute_type(self):
        self.pred._compute_type(self.child.typ.col_env(), None)
        self._type = self.child.typ

class MatrixCollectColsByKey(MatrixIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return f'(MatrixCollectColsByKey {r(self.child)})'

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_key_type._concat(
                hl.tstruct(**{f: hl.tarray(t) for f, t in child_typ.col_value_type.items()})),
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            hl.tstruct(**{f: hl.tarray(t) for f, t in child_typ.entry_type.items()}))

class MatrixAggregateColsByKey(MatrixIR):
    def __init__(self, child, entry_expr, col_expr):
        super().__init__()
        self.child = child
        self.entry_expr = entry_expr
        self.col_expr = col_expr

    def render(self, r):
        return '(MatrixAggregateColsByKey {} {} {})'.format(r(self.child), r(self.entry_expr), r(self.col_expr))

    def _compute_type(self):
        child_typ = self.child.typ
        self.entry_expr._compute_type(child_typ.row_env(), child_typ.entry_env())
        self.col_expr._compute_type(child_typ.global_env(), child_typ.col_env())
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_key_type._concat(self.col_expr.typ),
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            self.entry_expr.typ)
            

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

    def _compute_type(self):
        child_typ = self.child.typ
        other_field_set = set(self.row_key + self.row_fields + self.col_key + self.col_fields)
        entry_fields = [f for f in list(child_typ.row_type) if f not in other_field_set]
        self._type = hl.tmatrix(
            child_typ.global_type,
            hl.tstruct(**{f: child_typ.row_type[f] for f in self.col_key + self.col_fields}),
            self.col_key,
            hl.tstruct(**{f: child_typ.row_type[f] for f in self.row_key + self.row_fields}) ,
            self.row_key,
            hl.tstruct(**{f: child_typ.row_type[f] for f in entry_fields}))


class MatrixExplodeRows(MatrixIR):
    def __init__(self, child, path):
        super().__init__()
        self.child = child
        self.path = path

    def render(self, r):
        return '(MatrixExplodeRows ({}) {})'.format(
            ' '.join([escape_id(id) for id in self.path]),
            r(self.child))

    def _compute_type(self):
        child_typ = self.child.typ
        a = child_typ.row_type._index_path(self.path)
        new_row_type = child_typ.row_type._insert(self.path, a.element_type)
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            new_row_type,
            child_typ.row_key,
            child_typ.entry_type)
            

class MatrixRepartition(MatrixIR):
    def __init__(self, child, n, strategy):
        super().__init__()
        self.child = child
        self.n = n
        self.strategy = strategy

    def render(self, r):
        return f'(MatrixRepartition {r(self.child)} {self.n} {self.strategy})'

    def _compute_type(self):
        self._type = self.child.typ


class MatrixUnionRows(MatrixIR):
    def __init__(self, *children):
        super().__init__()
        self.children = children

    def render(self, r):
        return '(MatrixUnionRows {})'.format(' '.join(map(r, self.children)))

    def _compute_type(self):
        for c in self.children:
            c.typ # force
        self._type = self.children[0].typ

class MatrixDistinctByRow(MatrixIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return f'(MatrixDistinctByRow {r(self.child)})'

    def _compute_type(self):
        self._type = self.child.typ


class MatrixExplodeCols(MatrixIR):
    def __init__(self, child, path):
        super().__init__()
        self.child = child
        self.path = path

    def render(self, r):
        return '(MatrixExplodeCols ({}) {})'.format(
            ' '.join([escape_id(id) for id in self.path]),
            r(self.child))

    def _compute_type(self):
        child_typ = self.child.typ
        a = child_typ.col_type._index_path(self.path)
        new_col_type = child_typ.col_type._insert(self.path, a.element_type)
        self._type = hl.tmatrix(
            child_typ.global_type,
            new_col_type,
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            child_typ.entry_type)


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

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type._drop_fields([self.cols_field_name]),
            child_typ.global_type[self.cols_field_name].element_type,
            self.col_key,
            child_typ.row_type._drop_fields([self.entries_field_name]),
            child_typ.row_key,
            child_typ.row_type[self.entries_field_name].element_type)


class MatrixAnnotateRowsTable(MatrixIR):
    def __init__(self, child, table, root):
        super().__init__()
        self.child = child
        self.table = table
        self.root = root

    def render(self, r):
        return f'(MatrixAnnotateRowsTable "{self.root}" {r(self.child)} {r(self.table)})'

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_type._insert_field(self.root, self.table.typ.value_type),
            child_typ.row_key,
            child_typ.entry_type)

class MatrixAnnotateColsTable(MatrixIR):
    def __init__(self, child, table, root):
        super().__init__()
        self.child = child
        self.table = table
        self.root = root

    def render(self, r):
        return f'(MatrixAnnotateColsTable "{self.root}" {r(self.child)} {r(self.table)})'

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type._insert_field(self.root, self.table.typ.value_type),
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            child_typ.entry_type)


class MatrixToMatrixApply(MatrixIR):
    def __init__(self, child, config):
        super().__init__()
        self.child = child
        self.config = config

    def render(self, r):
        return f'(MatrixToMatrixApply {dump_json(self.config)} {r(self.child)})'

    def _compute_type(self):
        name = self.config['name']
        child_typ = self.child.typ
        if name == 'MatrixFilterPartitions':
            self._type = child_typ
        else:
            assert name == 'WindowByLocus', name
            self._type = hl.tmatrix(
                child_typ.global_type,
                child_typ.col_type,
                child_typ.col_key,
                child_typ.row_type._insert_field('prev_rows', hl.tarray(child_typ.row_type)),
                child_typ.row_key,
                child_typ.entry_type._insert_field('prev_entries', hl.tarray(child_typ.entry_type)))


class JavaMatrix(MatrixIR):
    def __init__(self, jir):
        super().__init__()
        self._jir = jir

    def render(self, r):
        return f'(JavaMatrix {r.add_jir(self._jir)})'

    def _compute_type(self):
        self._type = hl.tmatrix._from_java(self._jir.typ())
